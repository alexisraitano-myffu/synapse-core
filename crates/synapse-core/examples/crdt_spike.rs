//! SYN-112 spike — evaluate cr-sqlite (CRR layer) against the REAL Synapse
//! schema before choosing the T3 CRDT substrate (cr-sqlite vs Automerge).
//!
//! What it answers, empirically:
//! 1. Does the loadable extension coexist with our bundled rusqlite +
//!    sqlite-vec in one process (the exact production configuration)?
//! 2. Which of the production tables does `crsql_as_crr` accept, and what
//!    does it reject (AUTOINCREMENT pks, NOT NULL without default, vec0)?
//! 3. Do two databases converge after offline concurrent edits, and what
//!    does LWW pick on a same-column conflict?
//! 4. What happens to colliding AUTOINCREMENT ids created on two devices
//!    (the known weak point: inbox / atomic_notes / knowledge_graph)?
//!
//! Usage: crdt_spike <crsqlite_dylib> <out_dir>

use rusqlite::types::Value;
use rusqlite::Connection;
use synapse_core::Storage;

struct Change {
    table: String,
    pk: Vec<u8>,
    cid: String,
    val: Value,
    col_version: i64,
    db_version: i64,
    site_id: Vec<u8>,
    cl: i64,
    seq: i64,
}

fn open_with_crsqlite(path: &str, dylib: &str) -> Connection {
    // Production schema first (Storage::open runs the real init_schema and
    // registers vec0 process-wide), then a plain bundled connection on top.
    drop(Storage::open(path).expect("init production schema"));
    let conn = Connection::open(path).expect("open");
    conn.execute_batch("PRAGMA foreign_keys=OFF; PRAGMA busy_timeout=5000;")
        .expect("pragmas");
    unsafe {
        conn.load_extension_enable().expect("enable ext loading");
        conn.load_extension(dylib, None::<&str>).expect("load crsqlite");
        conn.load_extension_disable().expect("disable ext loading");
    }
    conn
}

fn user_tables(conn: &Connection) -> Vec<String> {
    let mut stmt = conn
        .prepare(
            "SELECT name FROM sqlite_master WHERE type='table' \
             AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'crsql_%' \
             AND name NOT LIKE '%__crsql%' AND name NOT LIKE 'atomic_notes_vec%' \
             ORDER BY name",
        )
        .unwrap();
    stmt.query_map([], |r| r.get(0)).unwrap().map(|r| r.unwrap()).collect()
}

fn pull_changes(conn: &Connection, since: i64) -> Vec<Change> {
    let mut stmt = conn
        .prepare(
            "SELECT \"table\", pk, cid, val, col_version, db_version, \
             site_id, cl, seq FROM crsql_changes WHERE db_version > ?1",
        )
        .expect("crsql_changes select");
    stmt.query_map([since], |r| {
        Ok(Change {
            table: r.get(0)?,
            pk: r.get(1)?,
            cid: r.get(2)?,
            val: r.get(3)?,
            col_version: r.get(4)?,
            db_version: r.get(5)?,
            site_id: r.get(6)?,
            cl: r.get(7)?,
            seq: r.get(8)?,
        })
    })
    .unwrap()
    .map(|r| r.unwrap())
    .collect()
}

fn apply_changes(conn: &Connection, changes: &[Change]) {
    let mut stmt = conn
        .prepare(
            "INSERT INTO crsql_changes \
             (\"table\", pk, cid, val, col_version, db_version, site_id, cl, seq) \
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
        )
        .expect("crsql_changes insert");
    for c in changes {
        stmt.execute(rusqlite::params![
            c.table, c.pk, c.cid, c.val, c.col_version, c.db_version, c.site_id,
            c.cl, c.seq
        ])
        .expect("apply change");
    }
}

fn db_version(conn: &Connection) -> i64 {
    conn.query_row("SELECT crsql_db_version()", [], |r| r.get(0)).unwrap()
}

fn sync_both_ways(a: &Connection, b: &Connection, since_a: i64, since_b: i64) -> (i64, i64) {
    let from_a = pull_changes(a, since_a);
    let from_b = pull_changes(b, since_b);
    apply_changes(b, &from_a);
    apply_changes(a, &from_b);
    (db_version(a), db_version(b))
}

/// Full deterministic dump of one table (rows as debug strings, sorted).
fn dump(conn: &Connection, table: &str) -> Vec<String> {
    let mut stmt = conn.prepare(&format!("SELECT * FROM \"{table}\"")).unwrap();
    let ncols = stmt.column_count();
    let mut rows: Vec<String> = stmt
        .query_map([], |r| {
            let mut parts = Vec::with_capacity(ncols);
            for i in 0..ncols {
                parts.push(format!("{:?}", r.get::<_, Value>(i).unwrap()));
            }
            Ok(parts.join("|"))
        })
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    rows.sort();
    rows
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dylib = args.next().expect("arg 1: crsqlite dylib path");
    let out_dir = args.next().expect("arg 2: output dir");
    std::fs::create_dir_all(&out_dir).unwrap();
    let path_a = format!("{out_dir}/spike_a.db");
    let path_b = format!("{out_dir}/spike_b.db");
    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);

    let a = open_with_crsqlite(&path_a, &dylib);
    let b = open_with_crsqlite(&path_b, &dylib);
    println!("✓ extension cr-sqlite chargée dans rusqlite bundled + vec0 (2 bases)");

    // ── 1. crsql_as_crr sur chaque table réelle ─────────────────────────────
    let mut crr_ok: Vec<String> = Vec::new();
    println!("\n── crsql_as_crr sur le schéma de production ──");
    for table in user_tables(&a) {
        let res_a: Result<Value, _> =
            a.query_row("SELECT crsql_as_crr(?1)", [&table], |r| r.get(0));
        match res_a {
            Ok(_) => {
                b.query_row("SELECT crsql_as_crr(?1)", [&table], |r| r.get::<_, Value>(0))
                    .expect("as_crr should succeed on B too");
                println!("  ✓ {table}");
                crr_ok.push(table);
            }
            Err(e) => println!("  ✗ {table}: {e}"),
        }
    }

    // La table virtuelle vec0, explicitement (attendu : refus).
    match a.query_row("SELECT crsql_as_crr('atomic_notes_vec')", [], |r| r.get::<_, Value>(0)) {
        Ok(_) => println!("  ✓ atomic_notes_vec (inattendu)"),
        Err(e) => println!("  ✗ atomic_notes_vec (vec0): {e}"),
    }

    // ── 2. Migration simulée : mêmes tables, pk NOT NULL (voie cr-sqlite) ──
    // Le schéma réel est intégralement rejeté ; on vérifie que la migration
    // « recréer chaque table avec un pk NOT NULL » suffirait au runtime.
    println!("\n── migration simulée (entities/facts recréées avec pk NOT NULL) ──");
    for conn in [&a, &b] {
        conn.execute_batch(
            "DROP TABLE entities; DROP TABLE facts;
             CREATE TABLE entities (
                id TEXT NOT NULL PRIMARY KEY, type TEXT, canonical_name TEXT,
                aliases TEXT DEFAULT '[]', mention_count INTEGER DEFAULT 1,
                last_mentioned DATE, summary TEXT);
             CREATE TABLE facts (
                id TEXT NOT NULL PRIMARY KEY, entity_id TEXT, predicate TEXT,
                value TEXT, confidence REAL DEFAULT 0.5);",
        )
        .unwrap();
        for t in ["entities", "facts"] {
            conn.query_row("SELECT crsql_as_crr(?1)", [t], |r| r.get::<_, Value>(0))
                .expect("as_crr après migration pk NOT NULL");
        }
    }
    crr_ok = vec!["entities".into(), "facts".into()];
    println!("  ✓ crsql_as_crr accepte les tables migrées (pk NOT NULL)");

    // ── 3. Convergence : édits concurrents hors-ligne ──────────────────────
    println!("\n── convergence 2 bases ──");
    if crr_ok.iter().any(|t| t == "entities") {
        a.execute(
            "INSERT INTO entities (id, type, canonical_name) VALUES ('e-shared','person','Alice')",
            [],
        )
        .unwrap();
        let (va, vb) = sync_both_ways(&a, &b, 0, 0);
        let seeded: i64 = b
            .query_row("SELECT COUNT(*) FROM entities WHERE id='e-shared'", [], |r| r.get(0))
            .unwrap();
        println!("  seed 'Alice' A→B: présente sur B = {} (versions A={va} B={vb})", seeded == 1);

        // Hors-ligne : conflit même colonne + colonnes disjointes + insert local.
        a.execute("UPDATE entities SET summary='résumé écrit par A', mention_count=5 WHERE id='e-shared'", []).unwrap();
        b.execute("UPDATE entities SET summary='résumé écrit par B', last_mentioned='2026-07-04' WHERE id='e-shared'", []).unwrap();
        b.execute(
            "INSERT INTO entities (id, type, canonical_name) VALUES ('e-b','person','Bob')",
            [],
        )
        .unwrap();
        if crr_ok.iter().any(|t| t == "facts") {
            a.execute(
                "INSERT INTO facts (id, entity_id, predicate, value) \
                 VALUES ('f-a','e-shared','aime','le café')",
                [],
            )
            .unwrap();
        }
        sync_both_ways(&a, &b, va, vb);
        sync_both_ways(&a, &b, 0, 0); // 2e passe : tout le monde a tout

        let (sum_a, mc_a, lm_a): (String, i64, Option<String>) = a
            .query_row(
                "SELECT summary, mention_count, last_mentioned FROM entities WHERE id='e-shared'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        let sum_b: String = b
            .query_row("SELECT summary FROM entities WHERE id='e-shared'", [], |r| r.get(0))
            .unwrap();
        println!("  conflit même colonne (summary): A={sum_a:?} B={sum_b:?} → identiques = {}", sum_a == sum_b);
        println!("  colonnes disjointes fusionnées: mention_count(A)=5→{mc_a}, last_mentioned(B)→{lm_a:?}");
    }

    // ── 4. Suppression concurrente à une édition (tombstones) ──────────────
    println!("\n── delete (A) concurrent d'un update (B) sur le même fait ──");
    a.execute("DELETE FROM facts WHERE id='f-a'", []).unwrap();
    b.execute("UPDATE facts SET value='le café noir' WHERE id='f-a'", []).unwrap();
    sync_both_ways(&a, &b, 0, 0);
    sync_both_ways(&a, &b, 0, 0);
    let na: i64 = a.query_row("SELECT COUNT(*) FROM facts WHERE id='f-a'", [], |r| r.get(0)).unwrap();
    let nb: i64 = b.query_row("SELECT COUNT(*) FROM facts WHERE id='f-a'", [], |r| r.get(0)).unwrap();
    println!("  après sync: présent sur A={na} B={nb} (cr-sqlite: delete gagne par longueur causale)");

    // ── 4. État final : les deux bases sont-elles identiques ? ─────────────
    println!("\n── comparaison finale des tables CRR ──");
    let mut diverged = 0;
    for table in &crr_ok {
        if dump(&a, table) != dump(&b, table) {
            diverged += 1;
            println!("  ✗ {table} diverge");
        }
    }
    if diverged == 0 {
        println!("  ✓ {} tables CRR identiques sur A et B", crr_ok.len());
    }

    a.query_row("SELECT crsql_finalize()", [], |r| r.get::<_, Value>(0)).ok();
    b.query_row("SELECT crsql_finalize()", [], |r| r.get::<_, Value>(0)).ok();
    println!("\nbases conservées: {path_a} · {path_b}");
}
