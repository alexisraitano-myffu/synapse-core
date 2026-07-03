//! Finest-grain probe: dlopen libonnxruntime.so and call OrtGetApiBase by hand.
//! Usage: dlopen_probe <libonnxruntime.so>

use std::os::raw::c_void;

#[repr(C)]
struct OrtApiBase {
    get_api: unsafe extern "C" fn(u32) -> *const c_void,
    get_version_string: unsafe extern "C" fn() -> *const std::os::raw::c_char,
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: dlopen_probe <lib>");

    eprintln!("[dlopen] 1: dlopen({path})...");
    let lib = unsafe { libloading::Library::new(&path) }.expect("dlopen failed");

    eprintln!("[dlopen] 2: resolving OrtGetApiBase...");
    let get_api_base: libloading::Symbol<unsafe extern "C" fn() -> *const OrtApiBase> =
        unsafe { lib.get(b"OrtGetApiBase\0") }.expect("symbol not found");

    eprintln!("[dlopen] 3: calling OrtGetApiBase()...");
    let base = unsafe { get_api_base() };
    assert!(!base.is_null(), "OrtGetApiBase returned null");

    eprintln!("[dlopen] 4: version string...");
    let ver = unsafe { std::ffi::CStr::from_ptr(((*base).get_version_string)()) };
    eprintln!("[dlopen] onnxruntime version: {}", ver.to_string_lossy());

    for api_version in [23u32, 22, 21, 20, 19, 18, 17] {
        let api = unsafe { ((*base).get_api)(api_version) };
        eprintln!("[dlopen] GetApi({api_version}) -> {}", if api.is_null() { "NULL" } else { "ok" });
    }
    println!("OK");
}
