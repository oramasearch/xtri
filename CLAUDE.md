# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust library called "xtri" that implements a radix tree (compressed trie) data structure. The library is fully functional with:

**Implemented Features:**
- Generic radix tree structure `RadixTree<T>` supporting any value type
- Insert method with automatic node compression and splitting
- Prefix-based search returning all matching key-value pairs (`search_prefix`)
- Lazy iterator for memory-efficient traversal (`search_iter`)
- Mutable value access through closures for safe modification (`mut_value`)
- Tree merging with customizable conflict resolution (`merge`)
- Parallel sorted bulk insertion for high-performance data loading (`from_sorted_parallel`, requires `parallel` feature)
- Alphabetical ordering of search results
- Full UTF-8 support including emojis and international characters
- Binary search optimization for fast lookups
- Memory-efficient O(tree depth) lazy iteration
- MIT license

**Pending Features:**
- Exact key search method
- Delete key-value pairs method
- Key existence checking method

## Development Commands

### Building and Testing
```bash
# Build the library
cargo build

# Build with parallel feature
cargo build --features parallel

# Run all tests
cargo test

# Run tests with parallel feature
cargo test --features parallel

# Run a specific test
cargo test test_non_ascii_characters

# Run tests with output
cargo test -- --nocapture

# Build with release optimizations
cargo build --release

# Check code without building
cargo check

# Run benchmarks
cargo bench

# Run benchmarks with parallel feature
cargo bench --features parallel
```

### Code Quality
```bash
# Format code
cargo fmt

# Run clippy linter
cargo clippy

# Run clippy with all targets and features
cargo clippy --all-targets --all-features

# Generate documentation
cargo doc --no-deps

# Generate documentation and open in browser
cargo doc --no-deps --open
```

## Architecture

The library consists of:
- `src/lib.rs`: Complete radix tree implementation with comprehensive tests
- `RadixTree<T>`: Main public struct with generic value support
- `RadixNode<T>`: Internal node structure for the tree
- Standard Cargo project structure using Rust 2024 edition

### Core Components

**RadixTree<T>:**
- `new()`: Creates empty radix tree
- `insert(&mut self, key: &str, value: T)`: Inserts key-value pair with automatic node splitting
- `search_prefix(&self, prefix: &str) -> Vec<(String, &T)>`: Returns all keys matching prefix
- `search_iter(&self, prefix: &str) -> SearchPrefixIterator<T>`: Returns lazy iterator for memory-efficient traversal
- `mut_value<F>(&mut self, key: &str, f: F) where F: FnOnce(&mut Option<T>)`: Provides mutable access to values through closures
- `merge<F>(self, other: Self, conflict_fn: F) -> Self`: Merges two trees with customizable conflict resolution (O(n + m) complexity)
- `from_sorted_parallel<K>(items: Vec<(K, T)>, chunk_size: Option<usize>) -> Self` (requires `parallel` feature): Builds tree from pre-sorted data using parallel construction (5-20x faster for 10K+ keys)
- `clear(&mut self)`: Removes all entries from the tree

**RadixNode<T>:**
- `value: Option<T>`: Optional value stored at this node
- `children: Vec<(u8, RadixNode<T>)>`: Child nodes as sorted vector for binary search
- `key: Vec<u8>`: Compressed path stored as bytes for UTF-8 support

**SearchPrefixIterator<T>:**
- Lazy iterator using stack-based traversal
- Memory usage: O(tree depth)
- Returns results in alphabetical order

## Implementation Details

### UTF-8 Handling
- Keys are stored as `Vec<u8>` internally for proper UTF-8 byte handling
- Search traversal maintains byte vectors until final string conversion
- Supports emojis, international characters, and all Unicode text

### Node Compression and Splitting
- Automatic path compression when nodes have single children
- Dynamic node splitting when inserting keys with partial matches
- Common prefix detection using byte-level comparison

### Memory Layout
- Uses `Vec<(u8, RadixNode<T>)>` sorted by byte for O(log n) binary search lookups
- Compressed paths reduce memory overhead for sparse trees
- Stack-based iterator provides O(tree depth) memory usage during traversal
- No external dependencies - uses only std library

### Performance Optimizations
- Binary search on sorted children for fast lookups
- Path compression reduces tree height and memory usage
- Lazy iteration prevents loading all results into memory at once
- Search results returned in alphabetical order without explicit sorting
- Structural tree merging with O(n + m) complexity using two-pointer algorithm
- Parallel sorted bulk insertion using rayon with tournament-style merging (optional `parallel` feature)

### Tree Merging
- Uses structural merge algorithm that recursively combines tree nodes
- Handles 4 cases: exact key match, prefix relationships, and partial matches
- Two-pointer merge for sorted children vectors maintains O(n + m) complexity
- Customizable conflict resolution via closure parameter
- Preserves all tree invariants (sorted children, path compression)

### Parallel Build (optional `parallel` feature)
- Chunks sorted data into fixed-size segments (default 1000 keys per chunk)
- Builds subtrees in parallel using rayon's parallel iterators
- Tournament-style pairwise merging in parallel rounds until one tree remains
- Expected 5-20x speedup for 10,000+ sorted keys on multi-core systems
- Requires `T: Send` trait bound for parallel operations
- Feature-gated to keep default builds lightweight

## Testing

Comprehensive test suite covers:
- Basic insert and search operations
- Iterator functionality and lazy evaluation
- Closure-based mutable value operations
- Complex node splitting scenarios including edge cases with mut_value
- **Tree merging operations**: empty trees, conflict resolution, disjoint keys, overlapping keys, UTF-8 support, correctness validation
- **Parallel build (with `parallel` feature)**: empty input, single chunk, multiple chunks, uneven chunks, correctness vs sequential, large datasets, UTF-8 support
- Non-ASCII character handling (UTF-8, emojis, Cyrillic)
- Edge cases: empty keys, overlapping prefixes, single characters
- Special characters and whitespace handling
- Numeric string prefixes
- Stress testing with multiple insertions
- Alphabetical ordering verification
- Memory efficiency testing
- Comprehensive output comparison between search_prefix and search_iter
- Mixed operations between insert and mut_value
- Documentation examples (README code is automatically tested)

Run `cargo test` to verify all functionality works correctly.
Run `cargo test --features parallel` to test parallel build functionality.

## Documentation
- README.md includes comprehensive usage examples
- All public APIs have detailed documentation with examples
- Documentation includes the README content via `#![doc = include_str!("../README.md")]`
- Examples in README are automatically tested as doc tests