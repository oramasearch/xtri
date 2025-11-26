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
- **Tree merging** with customizable conflict resolution
- **Parallel sorted bulk insertion** for high-performance data loading (optional `parallel` feature)

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
        let key = String::from_utf8_lossy(&key).to_string();
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

### Merging Trees

Efficiently combine two radix trees with customizable conflict resolution:

```rust
use xtri::{RadixTree, SearchMode};

let mut tree1 = RadixTree::new();
tree1.insert("apple", 1);
tree1.insert("banana", 2);

let mut tree2 = RadixTree::new();
tree2.insert("banana", 20);  // Conflicting key
tree2.insert("cherry", 3);

// Merge trees, keeping the second value on conflict
let merged = tree1.merge(tree2, |_v1, v2| v2);

assert_eq!(*merged.search_prefix("banana", SearchMode::Exact)[0].1, 20);
```

### Parallel Sorted Bulk Insertion

For large datasets that are already sorted, use the parallel build feature for significant performance improvements (requires `parallel` feature):

```toml
[dependencies]
xtri = { version = "0.1", features = ["parallel"] }
```

```rust,ignore
use xtri::RadixTree;

// Generate sorted data (or use your own pre-sorted data)
let items: Vec<(String, usize)> = (0..10000)
    .map(|i| (format!("key_{:08}", i), i))
    .collect();

// Build tree in parallel (5-20x faster than sequential insertion for large datasets)
let tree = RadixTree::from_sorted_parallel(items, Some(1000));

// Use the tree normally
assert_eq!(tree.len(), 10000);
```

**Performance:** For 10,000+ sorted keys on multi-core systems, parallel build provides 5-20x speedup compared to sequential insertion.
