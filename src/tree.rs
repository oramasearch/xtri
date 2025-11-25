use crate::{LeavesIterator, SearchIterator, SearchMode, TypoTolerantSearchIterator};

#[derive(Debug)]
pub struct RadixTree<T> {
    root: RadixNode<T>,
}

#[derive(Debug)]
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
}

pub fn common_prefix_length(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}
