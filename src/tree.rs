use serde::Serialize;

use crate::{LeavesIterator, SearchIterator, SearchMode, TypoTolerantSearchIterator};

#[derive(Debug, Serialize)]
pub struct RadixTree<T> {
    root: RadixNode<T>,
}

#[derive(Debug, Serialize)]
pub struct RadixNode<T> {
    pub value: Option<T>,
    pub children: Vec<(u8, RadixNode<T>)>,
    pub key: Vec<u8>,
}

impl<T> RadixNode<T> {
    fn new() -> Self {
        Self {
            value: None,
            children: Vec::new(),
            key: Vec::new(),
        }
    }

    fn new_with_key(key: Vec<u8>) -> Self {
        Self {
            value: None,
            children: Vec::new(),
            key,
        }
    }
}

impl<T> Default for RadixTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> RadixTree<T> {
    pub fn new() -> Self {
        Self {
            root: RadixNode::new(),
        }
    }

    /// Inserts a key-value pair into the radix tree.
    ///
    /// # Arguments
    /// * `key` - The string key to insert (supports full UTF-8 including emojis)
    /// * `value` - The value to associate with the key
    ///
    /// # Examples
    /// ```
    /// use xtri::RadixTree;
    /// let mut tree = RadixTree::new();
    /// tree.insert("hello", "world");
    /// tree.insert("help", "assistance");
    /// tree.insert("🚀", "rocket");
    /// ```
    pub fn insert(&mut self, key: &str, value: T) {
        self.mut_value(key, |existing_value| {
            *existing_value = Some(value);
        });
    }

    /// Searches for key-value pairs based on the specified search mode.
    ///
    /// Returns keys that match according to the search mode.
    /// Keys are processed at the byte level to properly handle UTF-8 encoded strings.
    ///
    /// # Arguments
    /// * `key` - The key string to search for (supports full UTF-8)
    /// * `mode` - The search mode (Prefix or Exact)
    ///
    /// # Returns
    /// A vector of tuples containing (key, value_reference) for all matches.
    /// Keys are returned as owned Strings, values as references.
    ///
    /// # Examples
    /// ```
    /// use xtri::{RadixTree, SearchMode};
    /// let mut tree = RadixTree::new();
    /// tree.insert("hello", 1);
    /// tree.insert("help", 2);
    /// tree.insert("world", 3);
    ///
    /// // Prefix search
    /// let results = tree.search_prefix("hel", SearchMode::Prefix);
    /// assert_eq!(results.len(), 2); // Returns: [("hello", &1), ("help", &2)]
    ///
    /// // Exact search
    /// let exact_results = tree.search_prefix("hello", SearchMode::Exact);
    /// assert_eq!(exact_results.len(), 1); // Returns: [("hello", &1)]
    /// ```
    pub fn search_prefix(&self, key: &str, mode: SearchMode) -> Vec<(String, &T)> {
        self.search_iter(key, mode)
            .map(|(key_bytes, value)| (String::from_utf8_lossy(&key_bytes).to_string(), value))
            .collect()
    }

    /// Returns an iterator over key-value pairs based on the specified search mode.
    ///
    /// This provides a lazy alternative to `search_prefix`, yielding results one at a time
    /// as the tree is traversed. Memory usage is O(tree depth) instead of O(number of results).
    ///
    /// # Arguments
    /// * `key` - The key string to search for (supports full UTF-8)
    /// * `mode` - The search mode (Prefix or Exact)
    ///
    /// # Returns
    /// An iterator that yields (key, value_reference) tuples for all matches.
    ///
    /// # Examples
    /// ```
    /// use xtri::{RadixTree, SearchMode};
    /// let mut tree = RadixTree::new();
    /// tree.insert("hello", 1);
    /// tree.insert("help", 2);
    /// tree.insert("world", 3);
    ///
    /// // Prefix search
    /// let results: Vec<_> = tree.search_iter("hel", SearchMode::Prefix).collect();
    /// assert_eq!(results.len(), 2);
    ///
    /// // Exact search
    /// let exact_results: Vec<_> = tree.search_iter("hello", SearchMode::Exact).collect();
    /// assert_eq!(exact_results.len(), 1);
    /// ```
    pub fn search_iter(&self, key: &str, mode: SearchMode) -> SearchIterator<T> {
        SearchIterator::new(&self.root, key.as_bytes(), mode)
    }

    /// Returns an iterator over values only based on the specified search mode.
    ///
    /// This provides a lazy alternative that yields only values, not keys.
    /// Internally uses `search_iter` and maps to extract only the value references.
    /// Memory usage is O(tree depth) instead of O(number of results).
    ///
    /// # Arguments
    /// * `key` - The key string to search for (supports full UTF-8)
    /// * `mode` - The search mode (Prefix or Exact)
    ///
    /// # Returns
    /// An iterator that yields value references for all matches.
    ///
    /// # Examples
    /// ```
    /// use xtri::{RadixTree, SearchMode};
    /// let mut tree = RadixTree::new();
    /// tree.insert("hello", 1);
    /// tree.insert("help", 2);
    /// tree.insert("world", 3);
    ///
    /// // Prefix search - get only values
    /// let values: Vec<_> = tree.search_iter_value("hel", SearchMode::Prefix).collect();
    /// assert_eq!(values.len(), 2);
    /// assert!(values.contains(&&1));
    /// assert!(values.contains(&&2));
    ///
    /// // Exact search - get only values
    /// let exact_values: Vec<_> = tree.search_iter_value("hello", SearchMode::Exact).collect();
    /// assert_eq!(exact_values.len(), 1);
    /// assert_eq!(*exact_values[0], 1);
    /// ```
    pub fn search_iter_value(&self, key: &str, mode: SearchMode) -> impl Iterator<Item = &T> {
        self.search_iter(key, mode).map(|(_, value)| value)
    }

    /// Returns an iterator over only the leaf nodes of the radix tree.
    ///
    /// A leaf node is defined as a node that has a value but no children,
    /// representing the end of a key path with no further extensions.
    /// This provides lazy traversal with memory usage of O(tree depth).
    /// Results are returned in alphabetical order.
    ///
    /// # Returns
    /// An iterator that yields (key, value_reference) tuples for all leaf nodes.
    ///
    /// # Examples
    /// ```
    /// use xtri::RadixTree;
    /// let mut tree = RadixTree::new();
    /// tree.insert("hello", 1);
    /// tree.insert("help", 2);
    /// tree.insert("world", 3);
    /// tree.insert("he", 4);  // Not a leaf (has children: hello, help)
    ///
    /// let leaves: Vec<_> = tree.iter_leaves().collect();
    /// // Returns only nodes with no children: [("hello", &1), ("help", &2), ("world", &3)]
    /// // "he" is not included because it has children
    /// ```
    pub fn iter_leaves(&self) -> LeavesIterator<T> {
        LeavesIterator::new(&self.root)
    }

    /// Searches for keys where the search term is approximately a prefix within the given tolerance.
    ///
    /// Uses Levenshtein distance to allow for typos and small variations. Returns results
    /// in alphabetical order with their edit distances. This performs fuzzy prefix matching,
    /// meaning the search term should approximately match the beginning of keys.
    ///
    /// # Arguments
    /// * `key` - The search term
    /// * `tolerance` - Maximum edit distance allowed (recommend 1-2 for performance)
    ///
    /// # Returns
    /// An iterator yielding (key, value_reference, distance) tuples where:
    /// - key is the matched key as a String
    /// - value_reference is a reference to the stored value
    /// - distance is the Levenshtein distance (0 means exact prefix match)
    ///
    /// # Note on UTF-8
    /// Distance is calculated at the byte level. Multi-byte UTF-8 characters
    /// may result in distances > 1 (e.g., "café" vs "cafe" has distance 2).
    ///
    /// # Examples
    /// ```
    /// use xtri::RadixTree;
    ///
    /// let mut tree = RadixTree::new();
    /// tree.insert("hello", 1);
    /// tree.insert("help", 2);
    /// tree.insert("hell", 3);
    /// tree.insert("hero", 4);
    ///
    /// // Search with typo: "helo" instead of "hel"
    /// let results: Vec<_> = tree.search_with_tolerance("helo", 1).collect();
    /// // Returns: [("hell", &3, 1), ("hello", &1, 1)]
    /// // "help" and "hero" have distance 2, so they're excluded
    ///
    /// // Exact matches have distance 0
    /// let results: Vec<_> = tree.search_with_tolerance("hel", 1).collect();
    /// // Returns: [("hell", &3, 0), ("hello", &1, 0), ("help", &2, 0)]
    /// ```
    pub fn search_with_tolerance(&self, key: &str, tolerance: u8) -> TypoTolerantSearchIterator<T> {
        TypoTolerantSearchIterator::new(&self.root, key.as_bytes(), tolerance)
    }

    /// Provides mutable access to the value associated with the given key through a closure.
    ///
    /// The closure receives a `&mut Option<T>` allowing you to read, modify, or create the value.
    /// If the key doesn't exist, the Option will be None initially. The closure's return value
    /// is returned by this method.
    ///
    /// # Arguments
    /// * `key` - The string key to look up (supports full UTF-8)
    /// * `f` - A closure that receives `&mut Option<T>` for the value at this key and returns a value
    ///
    /// # Returns
    /// The value returned by the closure
    ///
    /// # Examples
    /// ```
    /// use xtri::RadixTree;
    /// let mut tree: RadixTree<i32> = RadixTree::new();
    ///
    /// // Create a new value and return confirmation
    /// let created = tree.mut_value("counter", |value| {
    ///     *value = Some(1);
    ///     true // Return whether we created a new value
    /// });
    /// assert!(created);
    ///
    /// // Modify existing value and return the new value
    /// let new_value = tree.mut_value("counter", |value| {
    ///     if let Some(v) = value {
    ///         *v += 1;
    ///         *v
    ///     } else {
    ///         0
    ///     }
    /// });
    /// assert_eq!(new_value, 2);
    ///
    /// // Get current value without modifying
    /// let current = tree.mut_value("counter", |value| {
    ///     value.unwrap_or(0)
    /// });
    /// assert_eq!(current, 2);
    /// ```
    pub fn mut_value<F, R>(&mut self, key: &str, f: F) -> R
    where
        F: FnOnce(&mut Option<T>) -> R,
    {
        let key_bytes = key.as_bytes().to_vec();
        Self::mut_value_recursive(&mut self.root, key_bytes, f)
    }

    fn mut_value_recursive<F, R>(node: &mut RadixNode<T>, key: Vec<u8>, f: F) -> R
    where
        F: FnOnce(&mut Option<T>) -> R,
    {
        if key.is_empty() {
            // We've reached the target node - call the closure with the value
            return f(&mut node.value);
        }

        let first_byte = key[0];

        // Try to find existing child
        if let Ok(index) = node
            .children
            .binary_search_by_key(&first_byte, |(byte, _)| *byte)
        {
            let (_, child) = &mut node.children[index];
            let common_len = common_prefix_length(&child.key, &key);

            if common_len == child.key.len() {
                // Child's key is fully consumed, continue with remaining key
                let remaining_key = key[common_len..].to_vec();
                Self::mut_value_recursive(child, remaining_key, f)
            } else if common_len == key.len() && child.key.starts_with(&key) {
                // The key is a prefix of the child's key - need to split the child
                let old_key = child.key.clone();
                let old_value = child.value.take();
                let old_children = std::mem::take(&mut child.children);

                // Set the child to represent our key
                child.key = key.clone();
                child.value = None;

                // Create new child for the remainder of the old key
                let remaining_key = old_key[key.len()..].to_vec();
                if !remaining_key.is_empty() {
                    let remaining_first_byte = remaining_key[0];
                    let mut remaining_node = RadixNode::new_with_key(remaining_key);
                    remaining_node.value = old_value;
                    remaining_node.children = old_children;
                    child.children.push((remaining_first_byte, remaining_node));
                    child.children.sort_by_key(|(byte, _)| *byte);
                } else {
                    child.children = old_children;
                }

                // Call the closure with the split node's value
                return f(&mut child.value);
            } else {
                // Partial match - need to split both paths
                let old_key = child.key.clone();
                let old_value = child.value.take();
                let old_children = std::mem::take(&mut child.children);

                // Update child to represent common prefix
                child.key = old_key[..common_len].to_vec();
                child.value = None;
                child.children.clear();

                // Create node for old path
                let old_remaining_key = old_key[common_len..].to_vec();
                if !old_remaining_key.is_empty() {
                    let old_first_byte = old_remaining_key[0];
                    let mut old_node = RadixNode::new_with_key(old_remaining_key);
                    old_node.value = old_value;
                    old_node.children = old_children;
                    child.children.push((old_first_byte, old_node));
                }

                // Create node for new path
                let new_remaining_key = key[common_len..].to_vec();
                if new_remaining_key.is_empty() {
                    // The key ends at the split point
                    child.children.sort_by_key(|(byte, _)| *byte);
                    return f(&mut child.value);
                } else {
                    let new_first_byte = new_remaining_key[0];
                    let mut new_node = RadixNode::new_with_key(new_remaining_key);
                    new_node.value = None;
                    child.children.push((new_first_byte, new_node));
                    child.children.sort_by_key(|(byte, _)| *byte);

                    // Find the new node we just added and call closure
                    let new_index = child
                        .children
                        .binary_search_by_key(&new_first_byte, |(byte, _)| *byte)
                        .unwrap();
                    return f(&mut child.children[new_index].1.value);
                }
            }
        } else {
            // No existing child - create new one
            let mut new_child = RadixNode::new_with_key(key);
            new_child.value = None;
            node.children.push((first_byte, new_child));
            node.children.sort_by_key(|(byte, _)| *byte);

            // Find the new child we just added and call closure
            let index = node
                .children
                .binary_search_by_key(&first_byte, |(byte, _)| *byte)
                .unwrap();
            f(&mut node.children[index].1.value)
        }
    }

    pub fn clear(&mut self) {
        self.root = RadixNode::new();
    }

    /// Returns the number of key-value pairs stored in the radix tree.
    ///
    /// # Examples
    /// ```
    /// use xtri::RadixTree;
    /// let mut tree = RadixTree::new();
    /// assert_eq!(tree.len(), 0);
    ///
    /// tree.insert("hello", 1);
    /// tree.insert("world", 2);
    /// assert_eq!(tree.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        Self::count_values(&self.root)
    }

    /// Returns `true` if the radix tree contains no key-value pairs.
    ///
    /// # Examples
    /// ```
    /// use xtri::RadixTree;
    /// let mut tree = RadixTree::new();
    /// assert!(tree.is_empty());
    ///
    /// tree.insert("hello", 1);
    /// assert!(!tree.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn count_values(node: &RadixNode<T>) -> usize {
        let mut count = if node.value.is_some() { 1 } else { 0 };
        for (_, child) in &node.children {
            count += Self::count_values(child);
        }
        count
    }

    /// Merges another RadixTree into this one, consuming both trees.
    ///
    /// Uses structural merging for O(n + m) complexity, where n and m are
    /// the sizes of the two trees. This is significantly faster than inserting
    /// all entries from one tree into another.
    ///
    /// # Arguments
    /// * `other` - The tree to merge (consumed)
    /// * `conflict_fn` - Resolves value conflicts when both trees have a value
    ///                   at the same key. Receives (self_value, other_value) and
    ///                   returns the value to keep.
    ///
    /// # Returns
    /// The merged tree
    ///
    /// # Examples
    /// ```
    /// use xtri::RadixTree;
    ///
    /// let mut tree1 = RadixTree::new();
    /// tree1.insert("hello", 1);
    /// tree1.insert("world", 2);
    ///
    /// let mut tree2 = RadixTree::new();
    /// tree2.insert("hello", 10);  // Conflict!
    /// tree2.insert("help", 3);
    ///
    /// // Keep first tree's value on conflict
    /// let merged = tree1.merge(tree2, |v1, _v2| v1);
    ///
    /// // Result: {"hello": 1, "world": 2, "help": 3}
    /// assert_eq!(merged.len(), 3);
    /// ```
    pub fn merge<F>(mut self, other: Self, conflict_fn: F) -> Self
    where
        F: Fn(T, T) -> T + Copy,
    {
        self.root = merge_nodes(self.root, other.root, conflict_fn);
        self
    }

    /// Builds a RadixTree from pre-sorted data using parallel construction.
    ///
    /// This method chunks the input data, builds subtrees in parallel using rayon,
    /// and then merges them using a tournament-style algorithm. This can be 5-20x
    /// faster than sequential insertion for large datasets (10,000+ keys).
    ///
    /// # Requirements
    /// - Data MUST be sorted by key (lexicographic order)
    /// - `T` must implement `Send` for parallel building
    /// - Requires the `parallel` feature to be enabled
    ///
    /// # Arguments
    /// * `items` - Sorted iterator of (key, value) pairs
    /// * `chunk_size` - Keys per subtree (default: 1000). Tune based on your data.
    ///
    /// # Performance
    /// - Sequential insert: O(n² log n) for sorted data
    /// - Parallel build: O(n log n / cores) + O(n) merge
    /// - Best for n > 5000 on multi-core systems
    ///
    /// # Examples
    /// ```
    /// use xtri::RadixTree;
    ///
    /// let sorted_data: Vec<_> = (0..10000)
    ///     .map(|i| (format!("key_{:08}", i), i))
    ///     .collect();
    ///
    /// #[cfg(feature = "parallel")]
    /// let tree = RadixTree::from_sorted_parallel(sorted_data, None);
    /// ```
    #[cfg(feature = "parallel")]
    pub fn from_sorted_parallel<K, I>(items: I, chunk_size: Option<usize>) -> Self
    where
        K: AsRef<str>,
        I: IntoIterator<Item = (K, T)>,
        T: Send,
    {
        use rayon::prelude::*;

        let chunk_size = chunk_size.unwrap_or(1000);
        // Convert to owned strings for parallelization
        let items: Vec<(String, T)> = items
            .into_iter()
            .map(|(k, v)| (k.as_ref().to_string(), v))
            .collect();

        if items.is_empty() {
            return RadixTree::new();
        }

        // Split into chunks for parallel processing
        let mut chunks: Vec<Vec<(String, T)>> = Vec::new();
        let mut current_chunk = Vec::with_capacity(chunk_size);

        for item in items {
            current_chunk.push(item);
            if current_chunk.len() >= chunk_size {
                chunks.push(std::mem::replace(
                    &mut current_chunk,
                    Vec::with_capacity(chunk_size),
                ));
            }
        }
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        // Build trees in parallel using rayon
        let trees: Vec<RadixTree<T>> = chunks
            .into_par_iter()
            .map(|chunk| {
                let mut tree = RadixTree::new();
                for (key, value) in chunk {
                    tree.insert(&key, value);
                }
                tree
            })
            .collect();

        // Tournament-style merge
        tournament_merge(trees)
    }
}

pub fn common_prefix_length(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Merges values from two nodes, using the conflict function when both have values.
fn merge_values<T, F>(value1: &mut Option<T>, value2: Option<T>, conflict_fn: F)
where
    F: Fn(T, T) -> T,
{
    match (value1.take(), value2) {
        (Some(v1), Some(v2)) => *value1 = Some(conflict_fn(v1, v2)),
        (None, Some(v2)) => *value1 = Some(v2),
        (Some(v1), None) => *value1 = Some(v1),
        (None, None) => {}
    }
}

/// Merges two sorted children vectors using a two-pointer algorithm.
/// Recursively merges children with matching bytes.
fn merge_children<T, F>(
    children1: Vec<(u8, RadixNode<T>)>,
    children2: Vec<(u8, RadixNode<T>)>,
    conflict_fn: F,
) -> Vec<(u8, RadixNode<T>)>
where
    F: Fn(T, T) -> T + Copy,
{
    let mut result = Vec::with_capacity(children1.len() + children2.len());
    let mut iter1 = children1.into_iter().peekable();
    let mut iter2 = children2.into_iter().peekable();

    loop {
        match (iter1.peek(), iter2.peek()) {
            (None, None) => break,
            (Some(_), None) => {
                result.extend(iter1);
                break;
            }
            (None, Some(_)) => {
                result.extend(iter2);
                break;
            }
            (Some(&(byte1, _)), Some(&(byte2, _))) => {
                if byte1 < byte2 {
                    result.push(iter1.next().unwrap());
                } else if byte2 < byte1 {
                    result.push(iter2.next().unwrap());
                } else {
                    // Same byte - merge recursively
                    let (b1, node1) = iter1.next().unwrap();
                    let (_, node2) = iter2.next().unwrap();
                    let merged = merge_nodes(node1, node2, conflict_fn);
                    result.push((b1, merged));
                }
            }
        }
    }

    result
}

/// Core recursive merge function that merges two RadixNodes.
fn merge_nodes<T, F>(
    mut node1: RadixNode<T>,
    mut node2: RadixNode<T>,
    conflict_fn: F,
) -> RadixNode<T>
where
    F: Fn(T, T) -> T + Copy,
{
    // Case 1: Both are roots (empty keys) OR keys match exactly
    if (node1.key.is_empty() && node2.key.is_empty()) || node1.key == node2.key {
        merge_values(&mut node1.value, node2.value, conflict_fn);
        node1.children = merge_children(node1.children, node2.children, conflict_fn);
        return node1;
    }

    let common_len = common_prefix_length(&node1.key, &node2.key);

    // Case 2: node1.key is prefix of node2.key
    if common_len == node1.key.len() && common_len < node2.key.len() {
        node2.key = node2.key[common_len..].to_vec();
        let first_byte = node2.key[0];

        // Find matching child or insert
        match node1
            .children
            .binary_search_by_key(&first_byte, |(b, _)| *b)
        {
            Ok(idx) => {
                let child = std::mem::replace(&mut node1.children[idx].1, RadixNode::new());
                node1.children[idx].1 = merge_nodes(child, node2, conflict_fn);
            }
            Err(idx) => {
                node1.children.insert(idx, (first_byte, node2));
            }
        }
        return node1;
    }

    // Case 3: node2.key is prefix of node1.key
    if common_len == node2.key.len() && common_len < node1.key.len() {
        node1.key = node1.key[common_len..].to_vec();
        let first_byte = node1.key[0];

        match node2
            .children
            .binary_search_by_key(&first_byte, |(b, _)| *b)
        {
            Ok(idx) => {
                let child = std::mem::replace(&mut node2.children[idx].1, RadixNode::new());
                node2.children[idx].1 = merge_nodes(node1, child, conflict_fn);
            }
            Err(idx) => {
                node2.children.insert(idx, (first_byte, node1));
            }
        }
        return node2;
    }

    // Case 4: Partial match - create common prefix node
    let mut common_node = RadixNode::new_with_key(node1.key[..common_len].to_vec());

    node1.key = node1.key[common_len..].to_vec();
    node2.key = node2.key[common_len..].to_vec();

    let byte1 = node1.key[0];
    let byte2 = node2.key[0];

    if byte1 < byte2 {
        common_node.children.push((byte1, node1));
        common_node.children.push((byte2, node2));
    } else {
        common_node.children.push((byte2, node2));
        common_node.children.push((byte1, node1));
    }

    common_node
}

/// Merges multiple trees using a tournament-style algorithm (parallel pairwise merging).
/// This is used internally by from_sorted_parallel.
#[cfg(feature = "parallel")]
fn tournament_merge<T: Send>(mut trees: Vec<RadixTree<T>>) -> RadixTree<T> {
    use rayon::prelude::*;

    // Handle base cases
    if trees.is_empty() {
        return RadixTree::new();
    }
    if trees.len() == 1 {
        return trees.pop().unwrap();
    }

    // Merge in rounds until one tree remains
    while trees.len() > 1 {
        let tree_count = trees.len();

        // Convert Vec into chunks and process pairs in parallel
        let mut next_round = Vec::with_capacity((tree_count + 1) / 2);
        let mut trees_iter = trees.into_iter();
        let mut pairs = Vec::new();

        // Group trees into pairs
        while let Some(tree1) = trees_iter.next() {
            if let Some(tree2) = trees_iter.next() {
                pairs.push((tree1, tree2));
            } else {
                // Odd one out - save for next round
                next_round.push(tree1);
            }
        }

        // Merge pairs in parallel
        let mut merged: Vec<_> = pairs
            .into_par_iter()
            .map(|(tree1, tree2)| tree1.merge(tree2, |_v1, v2| v2))
            .collect();

        // Add merged trees to next round
        next_round.append(&mut merged);
        trees = next_round;
    }

    trees.pop().unwrap()
}
