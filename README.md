# xtri

A Rust library that implements a radix tree (compressed trie) data structure for efficient string storage and prefix-based searching.

## Features

- Generic radix tree structure `RadixTree<T>` supporting any value type
- Insert key-value pairs with automatic node compression and splitting
- Prefix-based search returning all matching key-value pairs (`search_prefix` and `search_iter`)
- Mutable value access through closures for safe modification (`mut_value`)
- Alphabetical ordering of search results
- Full UTF-8 support including emojis and international characters

## Usage

```rust
use xtri::{RadixTree, SearchMode};

fn main() {
    let mut tree = RadixTree::new();

    // Insert key-value pairs
    tree.insert("application", "main app");
    tree.insert("app", "short form");
    tree.insert("apple", "fruit");
    tree.insert("apply", "verb");
    tree.insert("approach", "method");
    tree.insert("appropriate", "suitable");

    // Search by prefix - returns all keys starting with "app"
    let iter = tree.search_iter("app", SearchMode::Prefix);
    for (key, value) in iter {
        println!("  {} -> {}", String::from_utf8_lossy(&key), value);
    }
    // Output:
    //   apple: fruit
    //   application: main app
    //   apply: verb

    // Update existing value or create a new one
    tree.mut_value("app", |value| {
        *value = Some("short form");
    });

    let iter = tree.search_iter("app", SearchMode::Exact);
    for (key, value) in iter {
        println!("  {} -> {}", String::from_utf8_lossy(&key), value);
    }
    // Output:
    //   app -> short form
}
```
