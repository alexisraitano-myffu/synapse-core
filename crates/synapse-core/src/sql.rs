//! Generic SQL gateway (SYN-110 / T1).
//!
//! Rationale: two SQLite libraries in one process (the host's own binding +
//! the core's bundled SQLite) do NOT isolate each other — POSIX advisory
//! locks are per-process, so cross-library transactions silently interleave
//! and corrupt the database (observed: `invalid page number` after a vec0
//! write inside an apsw transaction). The fix is architectural: the core's
//! bundled SQLite is the ONLY one in the process, and hosts run all their
//! SQL through [`SqlConnection`]s. Multiple simultaneous connections are
//! fine — that's ordinary same-library SQLite locking.
//!
//! The surface intentionally mirrors what the Python host uses of apsw:
//! `execute(sql, params)` returning columns + all rows, `last_insert_rowid`,
//! and transactions driven by plain `BEGIN`/`COMMIT`/`SAVEPOINT` statements.

use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use rusqlite::fallible_iterator::FallibleIterator;
use rusqlite::types::{Value, ValueRef};
use rusqlite::{params_from_iter, Connection};

use crate::embedder::CoreError;
use crate::storage::register_vec_extension;

/// One SQLite value crossing the FFI boundary, in either direction.
#[derive(Debug, Clone, PartialEq)]
pub enum SqlValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl From<&SqlValue> for Value {
    fn from(v: &SqlValue) -> Value {
        match v {
            SqlValue::Null => Value::Null,
            SqlValue::Integer(i) => Value::Integer(*i),
            SqlValue::Real(f) => Value::Real(*f),
            SqlValue::Text(s) => Value::Text(s.clone()),
            SqlValue::Blob(b) => Value::Blob(b.clone()),
        }
    }
}

impl From<ValueRef<'_>> for SqlValue {
    fn from(v: ValueRef<'_>) -> SqlValue {
        match v {
            ValueRef::Null => SqlValue::Null,
            ValueRef::Integer(i) => SqlValue::Integer(i),
            ValueRef::Real(f) => SqlValue::Real(f),
            ValueRef::Text(t) => SqlValue::Text(String::from_utf8_lossy(t).into_owned()),
            ValueRef::Blob(b) => SqlValue::Blob(b.to_vec()),
        }
    }
}

/// Result of an `execute`: `columns` is None for statements that return no
/// result set (INSERT/UPDATE/DDL), mirroring a cursor with no description.
#[derive(Debug, Clone, PartialEq)]
pub struct SqlResult {
    pub columns: Option<Vec<String>>,
    pub rows: Vec<Vec<SqlValue>>,
}

/// Open a plain SQL connection (no schema init — that's [`crate::Storage`]'s
/// job) with the vec0 extension available and a 5s busy timeout.
pub fn connect(db_path: &str) -> Result<SqlConnection, CoreError> {
    register_vec_extension();
    let conn = Connection::open(db_path)?;
    conn.busy_timeout(Duration::from_secs(5))?;
    // Parity with the apsw setup this replaces: SQLite's bundled build here
    // defaults foreign_keys ON (SQLITE_DEFAULT_FOREIGN_KEYS=1) while apsw
    // left it OFF, and the backend's soft-link/tombstone patterns (merged
    // entities, obsoleted_by, dangling provenance) rely on OFF.
    conn.pragma_update(None, "foreign_keys", false)?;
    Ok(SqlConnection {
        conn: Mutex::new(conn),
    })
}

pub struct SqlConnection {
    conn: Mutex<Connection>,
}

impl SqlConnection {
    fn lock(&self) -> Result<MutexGuard<'_, Connection>, CoreError> {
        self.conn
            .lock()
            .map_err(|_| CoreError::Storage("sql connection mutex poisoned".into()))
    }

    /// Execute SQL. Without params, multi-statement strings are allowed (the
    /// result is the last statement's); with params, exactly one statement.
    pub fn execute(&self, sql: &str, params: &[SqlValue]) -> Result<SqlResult, CoreError> {
        let conn = self.lock()?;
        if params.is_empty() {
            let mut result = SqlResult {
                columns: None,
                rows: Vec::new(),
            };
            let mut batch = rusqlite::Batch::new(&conn, sql);
            while let Some(mut stmt) = batch.next()? {
                result = run_stmt(&mut stmt, params)?;
            }
            Ok(result)
        } else {
            let mut stmt = conn.prepare(sql)?;
            run_stmt(&mut stmt, params)
        }
    }

    pub fn last_insert_rowid(&self) -> Result<i64, CoreError> {
        Ok(self.lock()?.last_insert_rowid())
    }
}

fn run_stmt(
    stmt: &mut rusqlite::Statement<'_>,
    params: &[SqlValue],
) -> Result<SqlResult, CoreError> {
    let values = params_from_iter(params.iter().map(Value::from));
    if stmt.column_count() == 0 {
        stmt.execute(values)?;
        return Ok(SqlResult {
            columns: None,
            rows: Vec::new(),
        });
    }

    let columns: Vec<String> = stmt.column_names().iter().map(|c| c.to_string()).collect();
    let n = columns.len();
    let mut rows_out = Vec::new();
    let mut rows = stmt.query(values)?;
    while let Some(row) = rows.next()? {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(SqlValue::from(row.get_ref(i)?));
        }
        rows_out.push(out);
    }
    Ok(SqlResult {
        columns: Some(columns),
        rows: rows_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executes_dml_and_queries_with_types() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sql.db");
        let c = connect(path.to_str().unwrap()).unwrap();

        let r = c
            .execute("CREATE TABLE t (i INTEGER, f REAL, s TEXT, b BLOB)", &[])
            .unwrap();
        assert!(r.columns.is_none());

        c.execute(
            "INSERT INTO t VALUES (?1, ?2, ?3, ?4)",
            &[
                SqlValue::Integer(7),
                SqlValue::Real(1.5),
                SqlValue::Text("été".into()),
                SqlValue::Blob(vec![0, 255]),
            ],
        )
        .unwrap();
        assert_eq!(c.last_insert_rowid().unwrap(), 1);

        let r = c.execute("SELECT i, f, s, b, NULL FROM t", &[]).unwrap();
        assert_eq!(
            r.columns.as_deref().unwrap(),
            ["i", "f", "s", "b", "NULL"]
        );
        assert_eq!(
            r.rows[0],
            vec![
                SqlValue::Integer(7),
                SqlValue::Real(1.5),
                SqlValue::Text("été".into()),
                SqlValue::Blob(vec![0, 255]),
                SqlValue::Null,
            ]
        );
    }

    #[test]
    fn multi_statement_without_params_returns_last_result() {
        let dir = tempfile::tempdir().unwrap();
        let c = connect(dir.path().join("m.db").to_str().unwrap()).unwrap();
        let r = c
            .execute(
                "CREATE TABLE a (x); INSERT INTO a VALUES (1); SELECT x FROM a",
                &[],
            )
            .unwrap();
        assert_eq!(r.rows, vec![vec![SqlValue::Integer(1)]]);
    }

    #[test]
    fn transactions_and_vec0_work_through_the_gateway() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("txn.db");
        // Schema via Storage (the owner), SQL via the gateway — the T1 shape.
        let storage = crate::Storage::open(path.to_str().unwrap()).unwrap();
        let c = connect(path.to_str().unwrap()).unwrap();

        c.execute("BEGIN", &[]).unwrap();
        c.execute(
            "INSERT INTO atomic_notes (id, content) VALUES (?1, ?2)",
            &[
                SqlValue::Text("note-uuid-1".into()),
                SqlValue::Text("pensée".into()),
            ],
        )
        .unwrap();
        c.execute("COMMIT", &[]).unwrap();

        // Core vector write after commit, then read back through raw SQL.
        let blob: Vec<u8> = (0..384u32)
            .flat_map(|i| (if i == 0 { 1.0f32 } else { 0.0 }).to_le_bytes())
            .collect();
        storage.upsert_note_vector("note-uuid-1", &blob).unwrap();
        let r = c
            .execute(
                "SELECT note_id FROM atomic_notes_vec WHERE embedding MATCH ?1 AND k = 1",
                &[SqlValue::Blob(blob)],
            )
            .unwrap();
        assert_eq!(r.rows, vec![vec![SqlValue::Text("note-uuid-1".into())]]);
    }
}
