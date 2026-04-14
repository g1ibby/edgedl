fn main() {
    // Ensure linker retains all sections (app descriptor, metadata, etc.)
    println!("cargo:rustc-link-arg=-Tlinkall.x");
}
