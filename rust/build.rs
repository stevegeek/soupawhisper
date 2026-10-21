// transcribe-cpp-sys 0.4.0 reconstructs the link line from its CMake manifest, which
// records the system libraries for Metal, Vulkan and ROCm but not CUDA. Link the CUDA
// runtime, cuBLAS and the driver ourselves when building with the `cuda` feature.
fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let cuda = std::env::var("CUDA_PATH").unwrap_or_else(|_| "/opt/cuda".into());
    for dir in ["lib64", "lib", "targets/x86_64-linux/lib"] {
        println!("cargo:rustc-link-search=native={cuda}/{dir}");
    }
    for lib in ["cudart", "cublas", "cublasLt", "cuda"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
