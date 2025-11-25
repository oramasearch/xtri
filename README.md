# xtri

[![Crates.io](https://img.shields.io/crates/v/xtri)](https://crates.io/crates/xtri)
[![Documentation](https://docs.rs/xtri/badge.svg)](https://docs.rs/xtri)
[![CI](https://github.com/oramasearch/xtri/actions/workflows/ci.yml/badge.svg)](https://github.com/oramasearch/xtri/actions/workflows/ci.yml)

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

    println!("Search 'app' by prefix");
    let iter = tree.search_iter("app", SearchMode::Prefix);
    for (key, value) in iter {
        println!("  {} -> {}", String::from_utf8_lossy(&key), value);
    }
    // Output:
    // Search 'app' by prefix
    //   app -> short form
    //   apple -> fruit
    //   application -> main app
    //   apply -> verb
    //   approach -> method
    //   appropriate -> suitable

    // Update existing value or create a new one
    tree.mut_value("app", |value| {
        *value = Some("short form");
    });

    println!("Search exactly 'app'");
    let iter = tree.search_iter("app", SearchMode::Exact);
    for (key, value) in iter {
        println!("  {} -> {}", String::from_utf8_lossy(&key), value);
    }
    // Output:
    //   app -> short form

    println!("Search 'app' with tolerance");
    let iter = tree.search_with_tolerance("apl", 1);
    for (key, value, distance) in iter {
        println!("  {} -> {} (distance={})", key, value, distance);
    }
    // Output:
    // Search 'app' with tolerance
    //   app -> short form (distance=1)
    //   apple -> fruit (distance=1)
    //   application -> main app (distance=1)
    //   apply -> verb (distance=1)
    //   approach -> method (distance=1)
    //   appropriate -> suitable (distance=1)
}
```
