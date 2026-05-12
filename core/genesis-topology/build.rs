use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(genesis_const_layer0_codec)");

    let const_requested = std::env::var("GENESIS_ENABLE_CONST_LAYER0_CODEC")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

    if const_requested && rustc_minor() >= Some(83) {
        println!("cargo:rustc-cfg=genesis_const_layer0_codec");
    }
}

fn rustc_minor() -> Option<u32> {
    let rustc = std::env::var("RUSTC").ok()?;
    let output = Command::new(rustc).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let mut parts = text.split_whitespace();
    let _ = parts.next();
    let ver = parts.next()?;
    let mut seg = ver.split('.');
    let major = seg.next();
    let minor = seg.next();

    match (major, minor) {
        (Some(_), Some(m)) => m.parse().ok(),
        _ => None,
    }
}
