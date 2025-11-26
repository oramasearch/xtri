

#[cfg(feature = "serde_json")]
fn main() {
    let v: Vec<String> = serde_json::from_slice(std::fs::read("/Users/allevo/repos/oramacore_lib/bar_100000.json").unwrap().as_slice()).unwrap();

    for _ in 0..100 {
        let mut tree = xtri::RadixTree::new();
        for (i, s) in v.iter().enumerate() {
            tree.insert(s, i);
        }
    }
}

#[cfg(not(feature = "serde_json"))]
fn main() {
    eprintln!("The serde_json feature is not enabled.");
}