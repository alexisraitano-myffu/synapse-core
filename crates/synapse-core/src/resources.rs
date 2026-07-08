//! SYN-21 — resource pipeline (T5 port of `dream_cycle/resources.py`): fetch
//! a URL, extract readable text, summarise it with the LLM, store it
//! (searchable via its embedded summary).
//!
//! HTML extraction is a dependency-free tag stripper (the Python original was
//! stdlib `html.parser`): skip script/style/nav/…, grab `<title>`, decode
//! common entities. The text is rough, but the LLM summarises it well.
//! Fetch failures and per-URL errors are non-fatal, like Python: a capture's
//! routing never blocks on a dead link.
//!
//! Everything runs on the Brain's OWN connection (network + LLM happen before
//! the DB write, no lock held) — hosts call it outside their transactions.

use std::collections::HashSet;
use std::time::Duration;

use rusqlite::{params, OptionalExtension};
use serde_json::json;

use crate::embedder::CoreError;
use crate::llm::{load_prompt, post_messages, response_text, LlmConfig};
use crate::routing::{new_uuid, Brain};

const SKIP_TAGS: [&str; 8] =
    ["script", "style", "noscript", "head", "nav", "footer", "header", "svg"];
const MAX_CONTENT: usize = 50_000; // cap stored text (chars) — articles can be huge
const FETCH_TIMEOUT: Duration = Duration::from_secs(10);

/// All http(s) URLs in a capture, de-duplicated, order-preserving. Port of
/// `URL_RE` (`https?://[^\s<>"'\)\]]+` + rstrip of trailing punctuation).
pub fn extract_urls(text: &str) -> Vec<String> {
    const STOP: &[char] = &['<', '>', '"', '\'', ')', ']'];
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    let mut last_end = 0usize;
    for (idx, _) in text.match_indices("http") {
        if idx < last_end {
            continue; // inside the previous match, like a regex scan
        }
        let tail = &text[idx..];
        let scheme_len = if tail.starts_with("https://") {
            8
        } else if tail.starts_with("http://") {
            7
        } else {
            continue;
        };
        let body = &tail[scheme_len..];
        let end = body
            .find(|c: char| c.is_whitespace() || STOP.contains(&c))
            .unwrap_or(body.len());
        if end == 0 {
            continue;
        }
        last_end = idx + scheme_len + end;
        let url = tail[..scheme_len + end].trim_end_matches(['.', ',', ';', ')']);
        if !url.is_empty() && seen.insert(url.to_string()) {
            out.push(url.to_string());
        }
    }
    out
}

pub struct PageText {
    pub title: String,
    pub text: String,
}

/// Decode the common HTML entities (named subset + numeric) — the Python
/// parser ran with `convert_charrefs=True`.
fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(p) = rest.find('&') {
        out.push_str(&rest[..p]);
        rest = &rest[p..];
        let semi = rest
            .as_bytes()
            .iter()
            .take(32)
            .position(|&b| b == b';');
        if let Some(semi) = semi {
            let ent = &rest[1..semi];
            let decoded = match ent {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some('\u{a0}'),
                _ if ent.starts_with("#x") || ent.starts_with("#X") => {
                    u32::from_str_radix(&ent[2..], 16).ok().and_then(char::from_u32)
                }
                _ if ent.starts_with('#') => {
                    ent[1..].parse::<u32>().ok().and_then(char::from_u32)
                }
                _ => None,
            };
            if let Some(c) = decoded {
                out.push(c);
                rest = &rest[semi + 1..];
                continue;
            }
        }
        out.push('&');
        rest = &rest[1..];
    }
    out.push_str(rest);
    out
}

/// Port of `_TextExtractor`: title + visible text, skip-tag subtrees dropped.
/// `<title>` wins over the skip counter (it lives inside `<head>`, which is a
/// skip tag). script/style bodies are raw text until their explicit end tag.
pub fn extract_page(html: &str) -> PageText {
    let mut title = String::new();
    let mut chunks: Vec<String> = Vec::new();
    let mut in_title = false;
    let mut skip = 0usize;

    let bytes = html.as_bytes();
    let mut i = 0usize;
    let mut data_start = 0usize;

    fn flush(
        html: &str,
        from: usize,
        to: usize,
        in_title: bool,
        skip: usize,
        title: &mut String,
        chunks: &mut Vec<String>,
    ) {
        if from >= to {
            return;
        }
        let decoded = decode_entities(&html[from..to]);
        if in_title {
            title.push_str(&decoded);
        } else if skip == 0 {
            let t = decoded.trim();
            if !t.is_empty() {
                chunks.push(t.to_string());
            }
        }
    }

    while i < bytes.len() {
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }
        flush(html, data_start, i, in_title, skip, &mut title, &mut chunks);

        if html[i..].starts_with("<!--") {
            i = html[i..].find("-->").map(|p| i + p + 3).unwrap_or(bytes.len());
            data_start = i;
            continue;
        }
        // Scan to the closing '>' honoring quoted attribute values.
        let mut j = i + 1;
        let mut quote: Option<u8> = None;
        while j < bytes.len() {
            let c = bytes[j];
            match quote {
                Some(q) => {
                    if c == q {
                        quote = None;
                    }
                }
                None => {
                    if c == b'"' || c == b'\'' {
                        quote = Some(c);
                    } else if c == b'>' {
                        break;
                    }
                }
            }
            j += 1;
        }
        if j >= bytes.len() {
            data_start = bytes.len();
            break; // unterminated tag — drop the tail
        }
        let inner = &html[i + 1..j];
        let closing = inner.starts_with('/');
        let name: String = inner
            .trim_start_matches('/')
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        let self_closing = !closing && inner.trim_end().ends_with('/');
        i = j + 1;
        data_start = i;
        if name.is_empty() {
            continue; // <!DOCTYPE …>, processing instructions…
        }
        if closing {
            if SKIP_TAGS.contains(&name.as_str()) {
                skip = skip.saturating_sub(1);
            } else if name == "title" {
                in_title = false;
            }
        } else {
            if SKIP_TAGS.contains(&name.as_str()) && !self_closing {
                skip += 1;
            } else if name == "title" && !self_closing {
                in_title = true;
            }
            // Raw-text elements: `if (a<b)` inside a script must not be
            // parsed as markup — jump straight to the explicit end tag.
            if (name == "script" || name == "style") && !self_closing {
                let close = format!("</{name}");
                match html[i..].to_ascii_lowercase().find(&close) {
                    Some(p) => {
                        i += p;
                        data_start = i;
                    }
                    None => {
                        i = bytes.len();
                        data_start = i;
                    }
                }
            }
        }
    }
    flush(html, data_start, bytes.len(), in_title, skip, &mut title, &mut chunks);

    PageText {
        title,
        text: chunks.join("\n"),
    }
}

/// GET the URL and return {title, text}. None on any network/HTTP/parse
/// failure — the caller treats a fetch miss as non-fatal.
pub fn fetch_and_extract(url: &str, timeout: Duration) -> Option<PageText> {
    let resp = ureq::get(url)
        .timeout(timeout)
        .set("User-Agent", "SynapseBot/1.0 (personal memory)")
        .call()
        .ok()?;
    let content_type = resp.header("content-type").unwrap_or("text/html").to_string();
    let body = resp.into_string().ok()?;
    if !content_type.contains("html") {
        // non-HTML (PDF, etc.) — out of scope for V1, store raw text if textual
        let text: String = body.chars().take(MAX_CONTENT).collect();
        if text.is_empty() {
            return None;
        }
        return Some(PageText {
            title: url.to_string(),
            text,
        });
    }
    let page = extract_page(&body);
    let text: String = page.text.chars().take(MAX_CONTENT).collect();
    if text.is_empty() {
        return None;
    }
    let title = page.title.trim().to_string();
    Some(PageText {
        title: if title.is_empty() { url.to_string() } else { title },
        text,
    })
}

/// LLM summary of the extracted text (prompt = data `resource-summary.md`).
/// Falls back to a truncated snippet without a config (offline) or on any
/// LLM/prompt failure — a resource is always storable.
fn summarize(config: Option<&LlmConfig>, title: &str, text: &str) -> String {
    let snippet: String = text.chars().take(300).collect();
    let Some(config) = config else {
        return snippet;
    };
    let Ok(system) = load_prompt(&config.prompts_dir, "resource-summary.md") else {
        return snippet;
    };
    let head: String = text.chars().take(8000).collect();
    let params_json = json!({
        "model": config.model,
        "max_tokens": 300,
        "system": system,
        "messages": [{"role": "user", "content": format!("Titre : {title}\n\n{head}")}],
    });
    match post_messages(config, &params_json) {
        Ok(body) => {
            let t = response_text(&body);
            if t.is_empty() {
                snippet
            } else {
                t
            }
        }
        Err(_) => snippet,
    }
}

impl Brain {
    /// Fetch → extract → summarise → store one URL. Idempotent on the URL (an
    /// already-stored link returns its id). Returns None if the fetch failed.
    /// Network + LLM happen BEFORE the DB write (no lock held).
    pub fn process_resource(
        &self,
        url: &str,
        capture_id: Option<&str>,
        config: Option<&LlmConfig>,
    ) -> Result<Option<String>, CoreError> {
        {
            let conn = self.storage.lock()?;
            let existing: Option<String> = conn
                .query_row("SELECT id FROM resources WHERE url = ?1", params![url], |r| {
                    r.get(0)
                })
                .optional()?;
            if existing.is_some() {
                return Ok(existing);
            }
        }

        let Some(page) = fetch_and_extract(url, FETCH_TIMEOUT) else {
            return Ok(None);
        };
        let summary = summarize(config, &page.title, &page.text);
        let embedding = self.embed(&format!("{}\n{}", page.title, summary));

        let rid = new_uuid();
        let now = crate::decay::resolve_now(None)
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let conn = self.storage.lock()?;
        conn.execute(
            "INSERT INTO resources (id, type, source, url, title, content, summary, \
             embedding, fetched_at) VALUES (?1,'url',?2,?3,?4,?5,?6,?7,?8)",
            params![rid, capture_id, url, page.title, page.text, summary, embedding, now],
        )?;
        Ok(Some(rid))
    }

    /// Process every URL found in a capture. Each is independent — one
    /// failure never blocks the others (or the rest of the cycle).
    pub fn process_capture_resources(
        &self,
        content: &str,
        capture_id: Option<&str>,
        config: Option<&LlmConfig>,
    ) -> Result<Vec<String>, CoreError> {
        let mut ids = Vec::new();
        for url in extract_urls(content) {
            if let Ok(Some(rid)) = self.process_resource(&url, capture_id, config) {
                ids.push(rid);
            }
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    #[test]
    fn extract_urls_dedups_and_strips_punctuation() {
        let urls = extract_urls(
            "voir https://exemple.fr/article. puis https://exemple.fr/article encore, \
             et http://x.io/y)",
        );
        assert_eq!(
            urls,
            ["https://exemple.fr/article", "http://x.io/y"]
        );
        assert_eq!(extract_urls("rien ici, http:// non plus"), Vec::<String>::new());
    }

    #[test]
    fn extract_page_skips_script_grabs_title() {
        let html = "<html><head><title>Mon Titre</title><style>x{}</style></head>\
                    <body><script>if (a<b) { bad() }</script><p>Bonjour &amp; le monde</p>\
                    <nav>menu</nav><!-- caché --></body></html>";
        let page = extract_page(html);
        assert_eq!(page.title.trim(), "Mon Titre");
        assert!(page.text.contains("Bonjour & le monde"));
        assert!(!page.text.contains("bad()"));
        assert!(!page.text.contains("menu"));
        assert!(!page.text.contains("caché"));
    }

    /// Single-thread HTTP stub good enough for ureq: read the request head,
    /// answer a fixed body, close.
    fn spawn_stub(status: &'static str, content_type: &'static str, body: &'static str) -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { break };
                let mut buf = [0u8; 2048];
                let _ = s.read(&mut buf);
                let resp = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = s.write_all(resp.as_bytes());
            }
        });
        port
    }

    #[test]
    fn process_resource_stores_and_is_idempotent() {
        let port = spawn_stub(
            "200 OK",
            "text/html; charset=utf-8",
            "<html><head><title>Article Exemple</title></head>\
             <body><p>Un texte sur les pandas roux.</p></body></html>",
        );
        let dir = tempfile::tempdir().unwrap();
        let brain =
            Brain::open(dir.path().join("r.db").to_str().unwrap(), None).unwrap();
        let url = format!("http://127.0.0.1:{port}/article");
        let rid1 = brain.process_resource(&url, Some("c1"), None).unwrap();
        let rid2 = brain.process_resource(&url, Some("c1"), None).unwrap();
        assert!(rid1.is_some());
        assert_eq!(rid1, rid2, "same URL must not be stored twice");
        let conn = brain.storage.lock().unwrap();
        let (n, title, summary): (i64, String, String) = conn
            .query_row(
                "SELECT COUNT(*), MAX(title), MAX(summary) FROM resources",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(title, "Article Exemple");
        // no LLM config → snippet fallback
        assert!(summary.contains("pandas roux"));
    }

    #[test]
    fn failed_fetch_stores_nothing() {
        let port = spawn_stub("404 Not Found", "text/html", "nope");
        let dir = tempfile::tempdir().unwrap();
        let brain =
            Brain::open(dir.path().join("r.db").to_str().unwrap(), None).unwrap();
        let rid = brain
            .process_resource(&format!("http://127.0.0.1:{port}/x"), None, None)
            .unwrap();
        assert!(rid.is_none());
        let conn = brain.storage.lock().unwrap();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM resources", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }
}
