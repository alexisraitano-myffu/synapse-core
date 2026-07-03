//! Bisect probe for the Android hang: exercises ort init step by step.
//! Usage: ort_probe <model.onnx>

fn main() {
    let model = std::env::args().nth(1).expect("usage: ort_probe <model.onnx>");

    eprintln!("[probe] 1: Session::builder() (dlopen + CreateEnv)...");
    let builder = ort::session::Session::builder().expect("builder failed");

    eprintln!("[probe] 2: builder ok; with_intra_threads(1)...");
    let mut builder = builder.with_intra_threads(1).expect("intra threads failed");

    eprintln!("[probe] 3: commit_from_file({model})...");
    let session = builder.commit_from_file(&model).expect("commit failed");

    eprintln!("[probe] 4: session ready; inputs={}", session.inputs().len());
    println!("OK");
}
