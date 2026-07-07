//! On-device smoke/parity CLI: `embed_cli <model_dir> [text]`.
//! Prints timing and the first components of the L2-normalized vector.
//!
//! Debug aid (unix only): a watchdog thread sends SIGUSR1 to the main thread
//! if model loading takes more than 15s; the handler prints the main thread's
//! backtrace (this is how the Android ort init hang was diagnosed).

#[cfg(unix)]
use std::os::raw::c_int;

#[cfg(unix)]
extern "C" fn dump_backtrace(_sig: c_int) {
    // Not async-signal-safe, but good enough for a diagnostic tool.
    eprintln!("=== main thread backtrace (watchdog) ===");
    backtrace::trace(|frame| {
        backtrace::resolve_frame(frame, |symbol| {
            let name = symbol
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|| "?".into());
            eprintln!("  {name}");
        });
        true
    });
    eprintln!("=== end backtrace ===");
    std::process::exit(42);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().expect("usage: embed_cli <model_dir> [text]");
    let text = args
        .next()
        .unwrap_or_else(|| "bonjour le monde".to_string());

    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGUSR1, dump_backtrace as usize);
        let main_thread = libc::pthread_self();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(15));
            eprintln!("[watchdog] still loading after 15s, dumping stack...");
            libc::pthread_kill(main_thread, libc::SIGUSR1);
        });
    }

    // Mirror ort's dlopen in THIS heavyweight process to surface the real
    // error that ort's deadlocking error path swallows.
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        match unsafe { libloading::Library::new(&p) } {
            Ok(_) => eprintln!("[embed_cli] pre-dlopen({p}) OK"),
            Err(e) => eprintln!("[embed_cli] pre-dlopen({p}) FAILED: {e}"),
        }
    }

    eprintln!("[embed_cli] loading model from {model_dir}...");
    let t0 = std::time::Instant::now();
    let embedder = synapse_core::Embedder::new(&model_dir).expect("model load failed");
    eprintln!("[embed_cli] model loaded in {:?}", t0.elapsed());

    let t1 = std::time::Instant::now();
    let vec = embedder.embed(&text).expect("embed failed");
    eprintln!("[embed_cli] embedded in {:?}", t1.elapsed());

    println!(
        "dim={} first8={:?}",
        vec.len(),
        &vec[..8.min(vec.len())]
    );

    // Parity mode: if reference_vectors.json sits in the model dir, compare
    // every reference vector (Python backend semantics) against this core.
    let ref_path = std::path::Path::new(&model_dir).join("reference_vectors.json");
    if let Ok(raw) = std::fs::read_to_string(&ref_path) {
        let refs: serde_json::Value = serde_json::from_str(&raw).expect("bad reference json");
        let mut worst = 1.0f64;
        for case in refs.as_array().expect("expected array") {
            let text = case["text"].as_str().unwrap();
            let expected: Vec<f64> = case["vector"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let got = embedder.embed(text).expect("embed failed");
            let cos: f64 = expected
                .iter()
                .zip(got.iter())
                .map(|(a, b)| a * *b as f64)
                .sum();
            if cos < worst {
                worst = cos;
            }
            println!("cos={cos:.7} {:?}", &text[..text.len().min(48)]);
        }
        println!("PARITY worst cosine = {worst:.7} ({})", if worst > 0.9999 { "OK" } else { "FAIL" });
    }
}
