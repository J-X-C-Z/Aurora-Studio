fn main() {
    println!("cargo:rerun-if-changed=app.rc");
    embed_resource::compile("app.rc", embed_resource::NONE)
        .manifest_optional()
        .expect("embed app.rc version resource");
}
