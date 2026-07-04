//! SYN-111 golden-parity runner: replay the frozen corpus through the Rust
//! routing into a fresh database. The Python side
//! (`scripts/golden/golden_compare.py` in the backend repo) normalizes and
//! diffs both databases.
//!
//! Usage:
//!   golden_replay <corpus.json> <out.db> <model_dir> <today YYYY-MM-DD>

use synapse_core::{Brain, RouteContext, SqlValue};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [corpus_path, out_db, model_dir, today] = &args[..] else {
        eprintln!("usage: golden_replay <corpus.json> <out.db> <model_dir> <today>");
        std::process::exit(2);
    };

    let corpus: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(corpus_path).expect("read corpus"))
            .expect("parse corpus");

    let brain = Brain::open(out_db, Some(model_dir)).expect("open brain");
    let sql = synapse_core::connect(out_db).expect("sql gateway");

    // Same frozen clock as the Python reference replay.
    let ctx = RouteContext {
        now: format!("{today}T00:00:00+00:00"),
        today: today.clone(),
        // Must sort BELOW SQLite's CURRENT_TIMESTAMP ("YYYY-MM-DD HH:MM:SS")
        // for rows created during the replay — like Python's real now−48h did.
        intentions_cutoff: format!("{today} 00:00:00"),
        now_sql: format!("{today} 00:00:00"),
    };

    let mut all_new_facts = Vec::new();
    let entries = corpus["entries"].as_array().expect("entries");
    for item in entries {
        // Corpus ids are historical integers; the uuid-pk schema stores
        // them as their text form (same convention as the Python replay).
        let capture_id = match &item["capture_id"] {
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::String(s) => s.clone(),
            other => panic!("capture_id: {other:?}"),
        };
        let content = item["content"].as_str().unwrap_or("");
        sql.execute(
            "INSERT INTO inbox (id, content, source, created_at, status) \
             VALUES (?, ?, ?, ?, 'queued')",
            &[
                SqlValue::Text(capture_id.clone()),
                SqlValue::Text(content.to_string()),
                SqlValue::Text("golden".to_string()),
                SqlValue::Text(item["created_at"].as_str().unwrap_or("").to_string()),
            ],
        )
        .expect("insert inbox");

        let entry = serde_json::json!({"id": capture_id, "content": content});
        let report = brain
            .route_capture(&entry, &item["classified"], &ctx)
            .unwrap_or_else(|e| panic!("route_capture id={capture_id}: {e}"));
        all_new_facts.extend(report.new_facts);
        println!(
            "  routed id={capture_id} ({} entité(s))",
            report.entity_ids.len()
        );
    }

    let promoted = brain.validate_pending(&all_new_facts).expect("step5");
    println!("step5: {promoted} pending fact(s) promoted");
    println!("rust replay → {out_db}");
}
