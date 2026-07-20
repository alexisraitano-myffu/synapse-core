//! T5 (SYN-114) — the LLM summary passes, ported from `dream_cycle/cycle.py`:
//! entity re-summary (SYN-89) and the living project synthesis (SYN-43, with
//! the SYN-44 "garbage collector" refinement). Prompts are DATA in
//! `prompts_dir` (`resummary.md`, `project-summary.md`,
//! `project-refinement.md`), byte-identical to the historical Python
//! constants; the HTTP path is the classifier's (`llm::post_messages`).

use rusqlite::types::Value as SqlV;
use rusqlite::{params, Connection};
use serde_json::{json, Value};

use crate::embedder::CoreError;
use crate::llm::{load_prompt, post_messages_text, LlmConfig};
use crate::routing::{
    entity_embedding_text, new_uuid, persist_project_entry, query_row_map, query_row_maps, Brain,
    ProjectSynthesis,
};

/// SYN-119 — majority language (ISO 639-1) across the atomic_notes that mention this
/// entity, or None when no mentioning note carries a detected language. `entities_mentioned`
/// is a JSON array of canonical names; we match the quoted name (LIKE wildcards escaped).
/// This is the deterministic "content language" of an entity — facts/relations alone carry
/// too little prose signal for the model to detect it reliably.
fn dominant_note_language(conn: &Connection, canonical: &str) -> Result<Option<String>, CoreError> {
    let escaped = canonical
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_");
    let pattern = format!("%\"{escaped}\"%");
    let mut stmt = conn.prepare(
        "SELECT language FROM atomic_notes \
         WHERE language IS NOT NULL AND language != '' \
           AND entities_mentioned LIKE ?1 ESCAPE '\\' \
         GROUP BY language ORDER BY COUNT(*) DESC, language ASC LIMIT 1",
    )?;
    let mut rows = stmt.query(params![pattern])?;
    match rows.next()? {
        Some(row) => Ok(Some(row.get::<_, String>(0)?)),
        None => Ok(None),
    }
}

/// Same two-step ``` fence strip as Python (drop the fence line, drop the tail).
fn strip_fences(text: &str) -> String {
    let t = text.trim();
    if !t.starts_with("```") {
        return t.to_string();
    }
    t.split_once('\n')
        .map(|(_, rest)| rest)
        .unwrap_or(t)
        .rsplit_once("```")
        .map(|(head, _)| head)
        .unwrap_or(t)
        .trim()
        .to_string()
}

/// SYN-44 — new entries since the last refinement before a from-scratch pass.
fn refinement_threshold() -> i64 {
    std::env::var("SYNAPSE_REFINEMENT_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .map(|v| v.max(1))
        .unwrap_or(20)
}

/// Python `f"{value}"` on a JSON scalar out of the row map.
fn scalar_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

impl Brain {
    /// SYN-89 — regenerate entity summaries from scratch (derived, never
    /// edited). Targets = `touched_ids` + entities flagged `summary_stale`.
    /// Rebuilt from ACTIVE facts + non-pending relations only. Returns the
    /// regenerated ids (to re-vectorize). An HTTP failure stops the pass —
    /// the stale flags survive for the next run; a per-entity content
    /// problem only skips that entity.
    pub fn resummarize(
        &self,
        touched_ids: &[String],
        config: &LlmConfig,
    ) -> Result<Vec<String>, CoreError> {
        let system = load_prompt(&config.prompts_dir, "resummary.md")?;
        let conn = self.storage.lock()?;
        let stale: Vec<String> = {
            let mut stmt = conn.prepare(
                "SELECT id FROM entities WHERE summary_stale = 1 AND merged_into_id IS NULL",
            )?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            rows.collect::<Result<_, _>>()?
        };
        let mut targets: Vec<String> = Vec::new();
        for id in touched_ids.iter().chain(stale.iter()) {
            if !targets.contains(id) {
                targets.push(id.clone());
            }
        }

        let mut regenerated: Vec<String> = Vec::new();
        for eid in &targets {
            let e = query_row_map(
                &conn,
                "SELECT id, canonical_name, type FROM entities \
                 WHERE id = ?1 AND merged_into_id IS NULL",
                &[SqlV::Text(eid.clone())],
            )?;
            let Some(e) = e else { continue };
            let facts = query_row_maps(
                &conn,
                "SELECT predicate, value FROM facts WHERE entity_id = ?1 \
                 AND obsoleted_at IS NULL AND archived_at IS NULL \
                 ORDER BY confidence DESC LIMIT 30",
                &[SqlV::Text(eid.clone())],
            )?;
            let relations = query_row_maps(
                &conn,
                "SELECT r.predicate, x.canonical_name AS target FROM relations r \
                 JOIN entities x ON x.id = r.entity_to \
                 WHERE r.entity_from = ?1 AND r.review_status != 'pending'",
                &[SqlV::Text(eid.clone())],
            )?;
            if facts.is_empty() && relations.is_empty() {
                // Nothing to derive from — keep the extraction summary, clear the flag.
                conn.execute(
                    "UPDATE entities SET summary_stale = 0 WHERE id = ?1",
                    params![eid],
                )?;
                continue;
            }

            let name = e.get("canonical_name").and_then(Value::as_str).unwrap_or("");
            let etype = e.get("type").and_then(Value::as_str).unwrap_or("");
            let mut lines = vec![format!(
                "Entité : {name}{}",
                if etype.is_empty() { String::new() } else { format!(" (type {etype})") }
            )];
            if !facts.is_empty() {
                lines.push("Faits :".to_string());
                for f in &facts {
                    lines.push(format!(
                        "- {} : {}",
                        f.get("predicate").and_then(Value::as_str).unwrap_or(""),
                        scalar_str(f.get("value").unwrap_or(&Value::Null)),
                    ));
                }
            }
            if !relations.is_empty() {
                lines.push("Relations :".to_string());
                for r in &relations {
                    lines.push(format!(
                        "- {} → {}",
                        r.get("predicate").and_then(Value::as_str).unwrap_or(""),
                        r.get("target").and_then(Value::as_str).unwrap_or(""),
                    ));
                }
            }

            // SYN-119 — content-language summary. Prefer the deterministic majority
            // language stored on the entity's source notes; fall back to letting the
            // model infer from the material when no note carries a language (facts /
            // relations are mostly interlingua, so inference alone is unreliable).
            let lang_directive = match dominant_note_language(&conn, name)? {
                Some(code) => format!(
                    "write the summary in {code} (ISO 639-1) — the dominant language of this \
                     entity's captures. Never translate to another language."
                ),
                None => "write the summary in the dominant language of the facts / relations \
                     below (the entity's content language) — never a fixed language."
                    .to_string(),
            };
            let system_e = system.replace("{language}", &lang_directive);

            let params_json = json!({
                "model": config.model,
                // SYN-124 — budget = sortie + marge pour un bloc de raisonnement. Un modèle
                // qui « pense » dépense d'abord son budget en thinking : dimensionné pour la
                // seule sortie, il rend une réponse vide tronquée à max_tokens (cas mesuré sur
                // Gemma E4B). max_tokens est un plafond, pas une cible : relever ne coûte rien
                // tant que le modèle ne génère pas plus.
                "max_tokens": 1024,
                "system": system_e,
                "messages": [{"role": "user", "content": lines.join("\n")}],
            });
            let summary = match post_messages_text(config, &params_json) {
                Ok(t) => t,
                // Infra failure — stop here; the stale flags survive.
                Err(CoreError::LlmHttp(_)) => break,
                Err(_) => continue,
            };
            // Still empty after the retry: leave summary_stale = 1 so the next
            // pass tries again rather than storing a blank fiche.
            if !summary.is_empty() {
                conn.execute(
                    "UPDATE entities SET summary = ?1, summary_stale = 0 WHERE id = ?2",
                    params![summary, eid],
                )?;
                regenerated.push(eid.clone());
            }
        }
        Ok(regenerated)
    }

    /// SYN-43 — amend (or create) a project's live synthesis after a new
    /// entry, then trigger the SYN-44 from-scratch refinement once enough
    /// entries accumulated. Failures never block the cycle (the entry is
    /// already persisted; the synthesis catches up on the next one).
    pub fn synthesize_project(
        &self,
        project_id: &str,
        project_name: &str,
        new_entry_content: &str,
        new_entry_count: i64,
        config: &LlmConfig,
    ) -> Result<Option<String>, CoreError> {
        let conn = self.storage.lock()?;
        append_project_summary(
            &conn,
            config,
            project_id,
            project_name,
            new_entry_content,
            new_entry_count,
        )
    }

    /// Host-facing project-entry write (the manual API endpoints): find or
    /// create the project entity, INSERT the entry, return the synthesis
    /// work item (the host decides whether to run the LLM synthesis).
    pub fn add_project_entry(
        &self,
        canonical: &str,
        content: &str,
        capture_id: &str,
        is_new_project: bool,
    ) -> Result<ProjectSynthesis, CoreError> {
        let conn = self.storage.lock()?;
        persist_project_entry(&conn, canonical, content, capture_id, is_new_project)
    }

    /// Port of `step6_vectorize` — embed each entity's composite text and
    /// store the vector. Per-entity failures skip (like Python); returns the
    /// count embedded. No-op without an embedder.
    pub fn vectorize_entities(&self, entity_ids: &[String]) -> Result<i64, CoreError> {
        let mut vectorized = 0i64;
        for eid in entity_ids {
            let entity = {
                let conn = self.storage.lock()?;
                query_row_map(
                    &conn,
                    "SELECT * FROM entities WHERE id = ?1",
                    &[SqlV::Text(eid.clone())],
                )?
            };
            let Some(entity) = entity else { continue };
            let Some(vec) = self.embed(&entity_embedding_text(&entity)) else {
                continue;
            };
            if self.storage.set_entity_embedding(eid, &vec).is_ok() {
                vectorized += 1;
            }
        }
        Ok(vectorized)
    }
}

/// SYN-134 — the project's ACTIVE facts (durable literal data: totals,
/// budget, choices) as a prompt block for the living synthesis. None when
/// the project carries no facts, so the historical prompt shape is
/// untouched for every project that predates project facts.
fn project_facts_block(conn: &Connection, project_id: &str) -> Result<Option<String>, CoreError> {
    let facts = query_row_maps(
        conn,
        "SELECT predicate, value FROM facts WHERE entity_id = ?1 \
         AND obsoleted_at IS NULL AND archived_at IS NULL \
         ORDER BY created_at ASC, id ASC",
        &[SqlV::Text(project_id.to_string())],
    )?;
    if facts.is_empty() {
        return Ok(None);
    }
    let lines = facts
        .iter()
        .map(|f| {
            format!(
                "- {} : {}",
                f.get("predicate").and_then(Value::as_str).unwrap_or(""),
                scalar_str(f.get("value").unwrap_or(&Value::Null)),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    Ok(Some(format!(
        "Faits actifs du projet (données durables à refléter fidèlement dans la synthèse) :\n{lines}"
    )))
}

/// Port of `_append_project_summary`. Any LLM failure → `Ok(None)` (the
/// caller keeps going; Python swallowed every exception here).
fn append_project_summary(
    conn: &Connection,
    config: &LlmConfig,
    project_id: &str,
    project_name: &str,
    new_entry_content: &str,
    new_entry_count: i64,
) -> Result<Option<String>, CoreError> {
    let system = load_prompt(&config.prompts_dir, "project-summary.md")?;
    let current = query_row_map(
        conn,
        "SELECT psv.summary_md FROM project_state ps \
         JOIN project_state_versions psv ON psv.id = ps.current_version_id \
         WHERE ps.project_id = ?1",
        &[SqlV::Text(project_id.to_string())],
    )?;
    let current_summary = current
        .as_ref()
        .and_then(|c| c.get("summary_md"))
        .and_then(Value::as_str)
        .map(String::from);

    // SYN-134 — the project's active facts ride along so the living
    // synthesis reflects the durable data, not just the entry timeline.
    let facts_block = project_facts_block(conn, project_id)?
        .map(|b| format!("\n\n{b}"))
        .unwrap_or_default();
    let user_msg = match &current_summary {
        Some(cur) => format!(
            "Projet : {project_name}\n\nSynthèse actuelle :\n---\n{cur}\n---\n\n\
             Nouvelle entrée à intégrer :\n---\n{new_entry_content}\n---{facts_block}\n\n\
             Mets à jour la synthèse pour intégrer la nouvelle entrée."
        ),
        None => format!(
            "Projet : {project_name}\n\nPremière entrée :\n---\n{new_entry_content}\n---{facts_block}\n\n\
             Écris la synthèse initiale du projet à partir de cette entrée."
        ),
    };

    let params_json = json!({
        "model": config.model,
        // SYN-124 — ~500 mots demandés + marge de raisonnement, cf. resummarize.
        "max_tokens": 2048,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_msg}],
    });
    let Ok(text) = post_messages_text(config, &params_json) else {
        return Ok(None);
    };
    let summary_md = strip_fences(&text);

    let version_id = new_uuid();
    conn.execute(
        "INSERT INTO project_state_versions \
         (id, project_id, summary_md, entry_count, trigger, kind) \
         VALUES (?1,?2,?3,?4,'passive','append')",
        params![version_id, project_id, summary_md, new_entry_count],
    )?;
    let has_state: Option<String> = {
        let mut stmt = conn.prepare("SELECT project_id FROM project_state WHERE project_id = ?1")?;
        let mut rows = stmt.query(params![project_id])?;
        match rows.next()? {
            Some(row) => Some(row.get(0)?),
            None => None,
        }
    };
    if has_state.is_some() {
        conn.execute(
            "UPDATE project_state SET current_version_id=?1, updated_at=CURRENT_TIMESTAMP, \
             entry_count_at_sync=?2 WHERE project_id=?3",
            params![version_id, new_entry_count, project_id],
        )?;
    } else {
        conn.execute(
            "INSERT INTO project_state \
             (project_id, current_version_id, entry_count_at_sync) VALUES (?1,?2,?3)",
            params![project_id, version_id, new_entry_count],
        )?;
    }

    // SYN-44: from-scratch refinement once enough new entries accumulated.
    let last_count: i64 = conn
        .query_row(
            "SELECT MAX(entry_count) FROM project_state_versions \
             WHERE project_id = ?1 AND kind = 'refinement'",
            params![project_id],
            |r| r.get::<_, Option<i64>>(0),
        )?
        .unwrap_or(0);
    if new_entry_count - last_count >= refinement_threshold() {
        refine_project_summary(conn, config, project_id, project_name)?;
    }

    Ok(Some(summary_md))
}

/// Port of `_refine_project_summary` — rebuild the synthesis from-scratch
/// from every entry (SYN-44 "garbage collector"). LLM failure → `Ok(None)`.
fn refine_project_summary(
    conn: &Connection,
    config: &LlmConfig,
    project_id: &str,
    project_name: &str,
) -> Result<Option<String>, CoreError> {
    let system = load_prompt(&config.prompts_dir, "project-refinement.md")?;
    let entries = query_row_maps(
        conn,
        "SELECT content, created_at FROM project_entries \
         WHERE project_id = ?1 ORDER BY created_at ASC LIMIT 200",
        &[SqlV::Text(project_id.to_string())],
    )?;
    if entries.is_empty() {
        return Ok(None);
    }

    let timeline = entries
        .iter()
        .map(|e| {
            format!(
                "[{}] {}",
                scalar_str(e.get("created_at").unwrap_or(&Value::Null)),
                e.get("content").and_then(Value::as_str).unwrap_or(""),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    // SYN-134 — same durable-facts block as the append pass.
    let facts_block = project_facts_block(conn, project_id)?
        .map(|b| format!("\n\n{b}"))
        .unwrap_or_default();
    let user_msg = format!(
        "Projet : {project_name}\n\n\
         Toutes les entrées dans l'ordre chronologique :\n---\n{timeline}\n---{facts_block}\n\n\
         Reconstruis from-scratch la synthèse du projet."
    );

    let params_json = json!({
        "model": config.model,
        // SYN-124 — 500-800 mots demandés + marge de raisonnement, cf. resummarize.
        "max_tokens": 3072,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_msg}],
    });
    let Ok(text) = post_messages_text(config, &params_json) else {
        return Ok(None);
    };
    let summary_md = strip_fences(&text);

    let entry_count = entries.len() as i64;
    let version_id = new_uuid();
    conn.execute(
        "INSERT INTO project_state_versions \
         (id, project_id, summary_md, entry_count, trigger, kind) \
         VALUES (?1,?2,?3,?4,'passive','refinement')",
        params![version_id, project_id, summary_md, entry_count],
    )?;
    conn.execute(
        "UPDATE project_state SET current_version_id=?1, updated_at=CURRENT_TIMESTAMP, \
         entry_count_at_sync=?2 WHERE project_id=?3",
        params![version_id, entry_count, project_id],
    )?;
    Ok(Some(summary_md))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_facts_block_filters_and_formats() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pf.db");
        crate::Storage::open(path.to_str().unwrap()).unwrap();
        let conn = Connection::open(&path).unwrap();
        conn.pragma_update(None, "foreign_keys", false).unwrap();
        conn.execute_batch(
            "INSERT INTO entities (id, type, canonical_name) VALUES ('p1', 'project', 'Terrasse');
             INSERT INTO facts (id, entity_id, predicate, value) VALUES ('f1', 'p1', 'budget', '3000 EUR');
             INSERT INTO facts (id, entity_id, predicate, value, obsoleted_at) VALUES ('f2', 'p1', 'budget', '2000 EUR', CURRENT_TIMESTAMP);
             INSERT INTO facts (id, entity_id, predicate, value, archived_at) VALUES ('f3', 'p1', 'surface', '12 m2', CURRENT_TIMESTAMP);",
        )
        .unwrap();

        let block = project_facts_block(&conn, "p1").unwrap().unwrap();
        assert!(block.starts_with("Faits actifs du projet"));
        assert!(block.contains("- budget : 3000 EUR"));
        assert!(!block.contains("2000 EUR")); // obsoleted filtered
        assert!(!block.contains("12 m2")); // archived filtered

        // A project without facts keeps the historical prompt shape.
        assert!(project_facts_block(&conn, "p-none").unwrap().is_none());
    }

    #[test]
    fn fences_strip_like_python() {
        assert_eq!(strip_fences("## titre\ncorps"), "## titre\ncorps");
        assert_eq!(strip_fences("```markdown\n## titre\ncorps\n```"), "## titre\ncorps");
        assert_eq!(strip_fences("```\nx\n```"), "x");
    }

    #[test]
    fn threshold_floor_is_one() {
        std::env::set_var("SYNAPSE_REFINEMENT_THRESHOLD", "0");
        assert_eq!(refinement_threshold(), 1);
        std::env::set_var("SYNAPSE_REFINEMENT_THRESHOLD", "garbage");
        assert_eq!(refinement_threshold(), 20);
        std::env::remove_var("SYNAPSE_REFINEMENT_THRESHOLD");
        assert_eq!(refinement_threshold(), 20);
    }

    // SYN-119 — the entity's "content language" is the majority `language` of the
    // atomic_notes that mention it (deterministic; injected into resummary.md). Also
    // proves the atomic_notes.language column exists (the INSERT would fail otherwise).
    #[test]
    fn dominant_note_language_is_majority_vote() {
        use crate::storage::Storage;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synapse.db");
        let storage = Storage::open(path.to_str().unwrap()).unwrap();
        let conn = storage.lock().unwrap();
        let mut ins = |mentioned: &str, lang: Option<&str>| {
            conn.execute(
                "INSERT INTO atomic_notes (id, content, entities_mentioned, language) \
                 VALUES (?1, 'x', ?2, ?3)",
                params![new_uuid(), mentioned, lang],
            )
            .unwrap();
        };
        ins(r#"["Marie","Paul"]"#, Some("fr"));
        ins(r#"["Marie"]"#, Some("fr"));
        ins(r#"["Marie"]"#, Some("en"));
        ins(r#"["Paul"]"#, Some("en"));
        ins(r#"["Solo"]"#, None); // no detected language

        // Marie: 2×fr vs 1×en → fr.
        assert_eq!(dominant_note_language(&conn, "Marie").unwrap().as_deref(), Some("fr"));
        // Paul: 1×fr vs 1×en → tie broken deterministically by language ASC → en.
        assert_eq!(dominant_note_language(&conn, "Paul").unwrap().as_deref(), Some("en"));
        // Solo: mentioned only by a language-less note → None (fall back to inference).
        assert_eq!(dominant_note_language(&conn, "Solo").unwrap(), None);
        // Never mentioned → None.
        assert_eq!(dominant_note_language(&conn, "Ghost").unwrap(), None);
    }
}
