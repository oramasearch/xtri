#![doc = include_str!("../README.md")]

/// Search mode for radix tree operations.
///
/// Determines how search operations should match keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Match all keys that start with the given prefix.
    Prefix,
    /// Match only keys that exactly equal the given string.
    Exact,
}

#[derive(Debug)]
pub struct RadixTree<T> {
    root: RadixNode<T>,
}

#[derive(Debug)]
struct RadixNode<T> {
    value: Option<T>,
    children: Vec<(u8, RadixNode<T>)>,
    key: Vec<u8>,
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

    fn common_prefix_length(a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
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
            let common_len = Self::common_prefix_length(&child.key, &key);

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

/// Iterator for traversing search results from a radix tree search.
///
/// This iterator performs truly lazy traversal using a stack-based approach.
/// Memory usage is O(tree depth) storing only node references and child indices.
pub struct SearchIterator<'a, T> {
    // Stack of (node, child_index, current_key) pairs for traversal state
    stack: Vec<(&'a RadixNode<T>, usize, Vec<u8>)>,
    // Search mode to determine matching behavior
    mode: SearchMode,
    // Original search key for exact matching validation
    search_key: Vec<u8>,
}

impl<'a, T> SearchIterator<'a, T> {
    fn new(root: &'a RadixNode<T>, prefix: &[u8], mode: SearchMode) -> Self {
        let mut iterator = Self {
            stack: Vec::new(),
            mode,
            search_key: prefix.to_vec(),
        };

        // Find the starting node that matches the prefix
        if let Some((start_node, key_prefix)) = Self::find_starting_node(root, prefix, Vec::new()) {
            iterator.stack.push((start_node, 0, key_prefix));
        }

        iterator
    }

    fn find_starting_node(
        node: &'a RadixNode<T>,
        prefix: &[u8],
        current_key: Vec<u8>,
    ) -> Option<(&'a RadixNode<T>, Vec<u8>)> {
        if prefix.is_empty() {
            // Empty prefix - start from this node
            return Some((node, current_key));
        }

        let first_byte = prefix[0];
        if let Ok(index) = node
            .children
            .binary_search_by_key(&first_byte, |(byte, _)| *byte)
        {
            let (_, child) = &node.children[index];
            let common_len = RadixTree::<T>::common_prefix_length(&child.key, prefix);

            if common_len == child.key.len() {
                // Child's key is fully consumed, continue with remaining prefix
                let mut new_current_key = current_key;
                new_current_key.extend_from_slice(&child.key);
                let remaining_prefix = &prefix[common_len..];
                return Self::find_starting_node(child, remaining_prefix, new_current_key);
            } else if common_len == prefix.len() && child.key.starts_with(prefix) {
                // Prefix is fully consumed and matches - start from this child
                let mut new_current_key = current_key;
                new_current_key.extend_from_slice(&child.key);
                return Some((child, new_current_key));
            }
        }

        None
    }
}

impl<'a, T> Iterator for SearchIterator<'a, T> {
    type Item = (Vec<u8>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (node, child_index, current_key) = self.stack.pop()?;

            // If this is the first visit to this node (child_index == 0), check for value
            if child_index == 0 {
                if let Some(ref value) = node.value {
                    // Check if this value matches based on search mode
                    let matches = match self.mode {
                        SearchMode::Exact => current_key == self.search_key,
                        SearchMode::Prefix => true, // All values found are valid for prefix search
                    };

                    if matches {
                        // Push back with child_index = 1 to continue with children next time
                        if !node.children.is_empty() {
                            self.stack.push((node, 1, current_key.clone()));
                        }
                        return Some((current_key, value));
                    }
                }

                // No value or no match, start processing children
                if !node.children.is_empty() {
                    let should_continue = match self.mode {
                        SearchMode::Prefix => true, // Always continue for prefix search
                        SearchMode::Exact => current_key.len() < self.search_key.len(), // Only continue if we haven't exceeded search key length
                    };

                    if should_continue {
                        self.stack.push((node, 1, current_key));
                    }
                }
                continue;
            }

            // Processing children: child_index - 1 is the current child being processed
            let current_child_idx = child_index - 1;

            if current_child_idx < node.children.len() {
                // Push the next child index for this node
                if current_child_idx + 1 < node.children.len() {
                    let should_continue_siblings = match self.mode {
                        SearchMode::Prefix => true,
                        SearchMode::Exact => current_key.len() <= self.search_key.len(),
                    };

                    if should_continue_siblings {
                        self.stack
                            .push((node, child_index + 1, current_key.clone()));
                    }
                }

                // Push the current child to be processed
                let (_, child) = &node.children[current_child_idx];
                let mut child_key = current_key;
                child_key.extend_from_slice(&child.key);

                // Determine if we should process this child
                let should_process_child = match self.mode {
                    SearchMode::Prefix => true, // Always process for prefix search
                    SearchMode::Exact => {
                        // Only process child if it could lead to the exact key
                        child_key.len() <= self.search_key.len()
                            && self.search_key.starts_with(&child_key)
                    }
                };

                if should_process_child {
                    self.stack.push((child, 0, child_key));
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Cannot determine size without traversing, provide conservative estimate
        (0, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut tree = RadixTree::new();

        // Test insert and search with overlapping prefixes
        tree.insert("hello", 1);
        tree.insert("help", 2);
        tree.insert("world", 3);
        tree.insert("hell", 4);

        // Test prefix search
        let results = tree.search_prefix("hel", SearchMode::Prefix);
        assert_eq!(results.len(), 3);
        let mut found_keys: Vec<String> = results.iter().map(|(k, _)| k.clone()).collect();
        found_keys.sort();
        assert_eq!(found_keys, vec!["hell", "hello", "help"]);

        // Test search_iter produces same results as search_prefix
        let iter_results: Vec<_> = tree.search_iter("hel", SearchMode::Prefix).collect();
        assert_eq!(results.len(), iter_results.len());
        let mut iter_keys: Vec<String> = iter_results
            .iter()
            .map(|(k, _)| String::from_utf8_lossy(k).to_string())
            .collect();
        iter_keys.sort();
        assert_eq!(found_keys, iter_keys);

        // Test exact prefix match
        tree.insert("test", 42);
        tree.insert("testing", 84);
        let test_results = tree.search_prefix("test", SearchMode::Prefix);
        assert_eq!(test_results.len(), 2);

        // Test no matches
        let no_results = tree.search_prefix("xyz", SearchMode::Prefix);
        assert_eq!(no_results.len(), 0);
    }

    #[test]
    fn test_node_splitting_complex() {
        let mut tree = RadixTree::new();

        tree.insert("testing", 1);
        tree.insert("test", 2);
        tree.insert("tea", 3);
        tree.insert("ted", 4);
        tree.insert("ten", 5);
        tree.insert("i", 6);
        tree.insert("in", 7);
        tree.insert("inn", 8);

        let results = tree.search_prefix("te", SearchMode::Prefix);
        assert_eq!(results.len(), 5);

        let mut keys: Vec<String> = results.iter().map(|(k, _)| k.clone()).collect();
        keys.sort();
        assert_eq!(keys, vec!["tea", "ted", "ten", "test", "testing"]);

        let i_results = tree.search_prefix("i", SearchMode::Prefix);
        assert_eq!(i_results.len(), 3);
        let mut i_keys: Vec<String> = i_results.iter().map(|(k, _)| k.clone()).collect();
        i_keys.sort();
        assert_eq!(i_keys, vec!["i", "in", "inn"]);
    }

    #[test]
    fn test_utf8_and_special_characters() {
        let mut tree = RadixTree::new();

        // Test UTF-8 characters (accented, emojis, Cyrillic)
        tree.insert("café", "coffee");
        tree.insert("naïve", "innocent");
        tree.insert("résumé", "cv");
        tree.insert("🚀", "rocket");
        tree.insert("🚗", "car");
        tree.insert("🚀🌟", "rocket star");
        tree.insert("привет", "hello");
        tree.insert("приветствие", "greeting");

        // Test special ASCII characters
        tree.insert("hello@world.com", "email");
        tree.insert("hello@world.org", "org_email");
        tree.insert("/path/to/file", "filepath");
        tree.insert("/path/to/dir/", "dirpath");
        tree.insert("key:value", "config1");
        tree.insert("key:another", "config2");
        tree.insert("C++", "programming");
        tree.insert("C#", "dotnet");

        // Test whitespace and control characters
        tree.insert(" leading_space", "space1");
        tree.insert("trailing_space ", "space2");
        tree.insert("tab\there", "tab");
        tree.insert("new\nline", "newline");

        // Verify UTF-8 searches
        let cafe_results = tree.search_prefix("café", SearchMode::Prefix);
        assert_eq!(cafe_results.len(), 1);
        assert_eq!(cafe_results[0].1, &"coffee");

        let emoji_results = tree.search_prefix("🚀", SearchMode::Prefix);
        assert_eq!(emoji_results.len(), 2);
        let mut emoji_keys: Vec<String> = emoji_results.iter().map(|(k, _)| k.clone()).collect();
        emoji_keys.sort();
        assert_eq!(emoji_keys, vec!["🚀", "🚀🌟"]);

        let russian_results = tree.search_prefix("прив", SearchMode::Prefix);
        assert_eq!(russian_results.len(), 2);

        // Verify special character searches
        let email_results = tree.search_prefix("hello@world", SearchMode::Prefix);
        assert_eq!(email_results.len(), 2);

        let path_results = tree.search_prefix("/path/to", SearchMode::Prefix);
        assert_eq!(path_results.len(), 2);

        let key_results = tree.search_prefix("key:", SearchMode::Prefix);
        assert_eq!(key_results.len(), 2);

        let c_results = tree.search_prefix("C", SearchMode::Prefix);
        assert_eq!(c_results.len(), 2);

        let space_results = tree.search_prefix(" ", SearchMode::Prefix);
        assert_eq!(space_results.len(), 1); // Only " leading_space"

        // Test mut_value with UTF-8
        tree.mut_value("новый", |value| {
            assert_eq!(*value, None);
            *value = Some("new");
        });

        tree.mut_value("новый", |value| {
            assert_eq!(*value, Some("new"));
        });
    }

    #[test]
    fn test_edge_cases_and_overlapping_keys() {
        let mut tree: RadixTree<&str> = RadixTree::new();

        // Test empty key and single characters
        tree.insert("", "empty");
        tree.insert("a", "single_a");
        tree.insert("b", "single_b");
        tree.insert("ab", "double_ab");

        let empty_results = tree.search_prefix("", SearchMode::Prefix);
        assert_eq!(empty_results.len(), 4);

        let a_results = tree.search_prefix("a", SearchMode::Prefix);
        assert_eq!(a_results.len(), 2);
        let mut a_keys: Vec<String> = a_results.iter().map(|(k, _)| k.clone()).collect();
        a_keys.sort();
        assert_eq!(a_keys, vec!["a", "ab"]);

        // Test overlapping keys of different lengths
        tree.insert("aa", "double_a");
        tree.insert("aaa", "triple_a");
        tree.insert("aaaa", "quad_a");
        tree.insert("abc", "abc_val");
        tree.insert("abcd", "abcd_val");
        tree.insert("abcde", "abcde_val");

        let aa_results = tree.search_prefix("aa", SearchMode::Prefix);
        assert_eq!(aa_results.len(), 3); // aa, aaa, aaaa

        let abc_results = tree.search_prefix("abc", SearchMode::Prefix);
        assert_eq!(abc_results.len(), 3); // abc, abcd, abcde

        // Test prefix relationships (key is prefix of existing and vice versa)
        tree.insert("testing", "test_long");
        tree.insert("test", "test_short");

        let test_results = tree.search_prefix("test", SearchMode::Prefix);
        assert_eq!(test_results.len(), 2); // test, testing

        // Test mut_value with overlapping keys and empty key
        tree.mut_value("", |value| {
            assert_eq!(*value, Some("empty"));
            *value = Some("updated_empty");
        });

        tree.mut_value("new_overlap", |value| {
            assert_eq!(*value, None);
            *value = Some("overlap1");
        });

        tree.mut_value("new_overlapping", |value| {
            assert_eq!(*value, None);
            *value = Some("overlap2");
        });

        let overlap_results = tree.search_prefix("new_overlap", SearchMode::Prefix);
        assert_eq!(overlap_results.len(), 2);
    }

    #[test]
    fn test_numeric_strings() {
        let mut tree = RadixTree::new();

        tree.insert("1", "one");
        tree.insert("10", "ten");
        tree.insert("100", "hundred");
        tree.insert("101", "hundred_one");
        tree.insert("11", "eleven");
        tree.insert("2", "two");
        tree.insert("20", "twenty");

        let one_results = tree.search_prefix("1", SearchMode::Prefix);
        assert_eq!(one_results.len(), 5);

        let ten_results = tree.search_prefix("10", SearchMode::Prefix);
        assert_eq!(ten_results.len(), 3);

        let hundred_results = tree.search_prefix("100", SearchMode::Prefix);
        assert_eq!(hundred_results.len(), 1);

        let two_results = tree.search_prefix("2", SearchMode::Prefix);
        assert_eq!(two_results.len(), 2);
    }

    #[test]
    fn test_stress_many_insertions() {
        let mut tree = RadixTree::new();

        let prefixes = ["app", "test", "user", "data", "config"];
        let suffixes = ["_1", "_2", "_test", "_prod", "_dev", "_staging"];

        for prefix in &prefixes {
            for suffix in &suffixes {
                let key = format!("{prefix}{suffix}");
                tree.insert(&key, format!("value_{key}"));
            }
        }

        for prefix in &prefixes {
            let results = tree.search_prefix(prefix, SearchMode::Prefix);
            assert_eq!(results.len(), suffixes.len());

            for (key, value) in results {
                assert!(key.starts_with(prefix));
                assert_eq!(value, &format!("value_{key}"));
            }
        }

        let all_results = tree.search_prefix("", SearchMode::Prefix);
        assert_eq!(all_results.len(), prefixes.len() * suffixes.len());
    }

    #[test]
    fn test_search_iter_comprehensive() {
        let mut tree = RadixTree::new();

        // Test basic iterator functionality
        tree.insert("hello", 1);
        tree.insert("help", 2);
        tree.insert("world", 3);
        tree.insert("hell", 4);

        let results: Vec<_> = tree.search_iter("hel", SearchMode::Prefix).collect();
        assert_eq!(results.len(), 3);

        let mut found_keys: Vec<String> = results
            .iter()
            .map(|(k, _)| String::from_utf8_lossy(k).to_string())
            .collect();
        found_keys.sort();
        assert_eq!(found_keys, vec!["hell", "hello", "help"]);

        // Verify values are correct
        for (key, value) in results {
            let key_str = String::from_utf8_lossy(&key);
            match key_str.as_ref() {
                "hello" => assert_eq!(*value, 1),
                "help" => assert_eq!(*value, 2),
                "hell" => assert_eq!(*value, 4),
                _ => panic!("Unexpected key: {key_str}"),
            }
        }

        // Test search_iter vs search_prefix equivalence
        tree.insert("testing", 10);
        tree.insert("tea", 30);
        tree.insert("ted", 40);

        let prefix_results = tree.search_prefix("te", SearchMode::Prefix);
        let iter_results: Vec<_> = tree.search_iter("te", SearchMode::Prefix).collect();

        assert_eq!(prefix_results.len(), iter_results.len());

        let mut prefix_keys: Vec<String> = prefix_results.iter().map(|(k, _)| k.clone()).collect();
        let mut iter_keys: Vec<String> = iter_results
            .iter()
            .map(|(k, _)| String::from_utf8_lossy(k).to_string())
            .collect();
        prefix_keys.sort();
        iter_keys.sort();
        assert_eq!(prefix_keys, iter_keys);

        // Test empty results
        let empty_results: Vec<_> = tree.search_iter("xyz", SearchMode::Prefix).collect();
        assert_eq!(empty_results.len(), 0);

        // Test UTF-8 iterator with separate tree
        let mut utf8_tree = RadixTree::new();
        utf8_tree.insert("🚀", "rocket");
        utf8_tree.insert("🚀🌟", "rocket star");

        let utf8_results: Vec<_> = utf8_tree.search_iter("🚀", SearchMode::Prefix).collect();
        assert_eq!(utf8_results.len(), 2);

        // Test lazy evaluation with large dataset
        for i in 100..200 {
            tree.insert(&format!("large{i}"), i);
        }

        let lazy_iter = tree.search_iter("large", SearchMode::Prefix);
        let first_five: Vec<_> = lazy_iter.take(5).collect();
        assert_eq!(first_five.len(), 5);

        for (key, _) in first_five {
            let key_str = String::from_utf8_lossy(&key);
            assert!(key_str.starts_with("large"));
        }
    }

    #[test]
    fn test_comprehensive_output_comparison() {
        let mut tree = RadixTree::new();

        // Test comprehensive edge cases
        let test_cases = vec![
            // Basic cases
            ("hello", 1),
            ("help", 2),
            ("world", 3),
            ("hell", 4),
            // Empty key
            ("", 0),
            // Single characters
            ("a", 10),
            ("b", 11),
            ("z", 12),
            // Overlapping prefixes
            ("test", 20),
            ("testing", 21),
            ("tester", 22),
            ("te", 23),
            // UTF-8 cases
            ("café", 30),
            ("naïve", 31),
            ("🚀", 32),
            ("🚀🌟", 33),
            // Special characters
            ("key:value", 40),
            ("path/file", 41),
            ("user@domain", 42),
            // Numeric strings
            ("1", 50),
            ("10", 51),
            ("100", 52),
            ("101", 53),
            // Whitespace
            (" space", 60),
            ("tab\t", 61),
            ("new\nline", 62),
        ];

        for (key, value) in test_cases {
            tree.insert(key, value);
        }

        // Test prefixes that should reveal differences
        let search_prefixes = vec![
            "",
            "h",
            "he",
            "hel",
            "hell",
            "hello",
            "help",
            "world",
            "w",
            "a",
            "b",
            "z",
            "test",
            "te",
            "café",
            "🚀",
            "key",
            "path",
            "user",
            "1",
            "10",
            "100",
            " ",
            "tab",
            "new",
            "nonexistent",
            "🚗",
        ];

        for prefix in search_prefixes {
            println!("Testing prefix: '{prefix}'");

            let prefix_results = tree.search_prefix(prefix, SearchMode::Prefix);
            let iter_results: Vec<_> = tree.search_iter(prefix, SearchMode::Prefix).collect();

            // Compare lengths
            assert_eq!(
                prefix_results.len(),
                iter_results.len(),
                "Length mismatch for prefix '{}': search_prefix={}, search_iter={}",
                prefix,
                prefix_results.len(),
                iter_results.len()
            );

            // Sort both results for comparison (since order might differ)
            let mut prefix_sorted: Vec<_> =
                prefix_results.into_iter().map(|(k, v)| (k, *v)).collect();
            prefix_sorted.sort_by(|a, b| a.0.cmp(&b.0));

            let mut iter_sorted: Vec<_> = iter_results
                .into_iter()
                .map(|(k, v)| (String::from_utf8_lossy(&k).to_string(), *v))
                .collect();
            iter_sorted.sort_by(|a, b| a.0.cmp(&b.0));

            // Compare sorted results
            assert_eq!(
                prefix_sorted, iter_sorted,
                "Content mismatch for prefix '{prefix}': prefix_results={prefix_sorted:?}, iter_results={iter_sorted:?}"
            );
        }
    }

    #[test]
    fn test_edge_case_scenarios() {
        let mut tree = RadixTree::new();

        // Edge Case 1: Keys that are prefixes of other keys
        tree.insert("a", 1);
        tree.insert("ab", 2);
        tree.insert("abc", 3);
        tree.insert("abcd", 4);

        let prefix_a = tree.search_prefix("a", SearchMode::Prefix);
        let iter_a: Vec<_> = tree.search_iter("a", SearchMode::Prefix).collect();
        assert_eq!(prefix_a.len(), iter_a.len());
        assert_eq!(prefix_a.len(), 4);

        // Edge Case 2: Identical prefixes with different suffixes
        tree.clear();
        tree.insert("prefix1", 10);
        tree.insert("prefix2", 20);
        tree.insert("prefix12", 30);

        let prefix_pre = tree.search_prefix("prefix", SearchMode::Prefix);
        let iter_pre: Vec<_> = tree.search_iter("prefix", SearchMode::Prefix).collect();
        assert_eq!(prefix_pre.len(), iter_pre.len());
        assert_eq!(prefix_pre.len(), 3);

        // Edge Case 3: UTF-8 boundary issues
        tree.clear();
        tree.insert("café", 1);
        tree.insert("cafe", 2); // Similar but different

        let prefix_caf = tree.search_prefix("caf", SearchMode::Prefix);
        let iter_caf: Vec<_> = tree.search_iter("caf", SearchMode::Prefix).collect();
        assert_eq!(prefix_caf.len(), iter_caf.len());
        assert_eq!(prefix_caf.len(), 2);

        // Edge Case 4: Empty tree
        tree.clear();
        let prefix_empty = tree.search_prefix("anything", SearchMode::Prefix);
        let iter_empty: Vec<_> = tree.search_iter("anything", SearchMode::Prefix).collect();
        assert_eq!(prefix_empty.len(), iter_empty.len());
        assert_eq!(prefix_empty.len(), 0);

        // Edge Case 5: Single node tree
        tree.insert("single", 42);
        let prefix_single = tree.search_prefix("single", SearchMode::Prefix);
        let iter_single: Vec<_> = tree.search_iter("single", SearchMode::Prefix).collect();
        assert_eq!(prefix_single.len(), iter_single.len());
        assert_eq!(prefix_single.len(), 1);
        assert_eq!(prefix_single[0].1, iter_single[0].1);
    }

    #[test]
    fn test_search_iter_alphabetical_order() {
        let mut tree = RadixTree::new();

        // Insert keys in non-alphabetical order to test sorting
        tree.insert("zebra", 1);
        tree.insert("apple", 2);
        tree.insert("banana", 3);
        tree.insert("cherry", 4);
        tree.insert("date", 5);
        tree.insert("elderberry", 6);
        tree.insert("fig", 7);
        tree.insert("grape", 8);

        // Test with empty prefix (should return all keys sorted)
        let all_results: Vec<_> = tree.search_iter("", SearchMode::Prefix).collect();
        let all_keys: Vec<String> = all_results
            .iter()
            .map(|(k, _)| String::from_utf8_lossy(k).to_string())
            .collect();

        assert_eq!(
            all_keys,
            vec![
                "apple",
                "banana",
                "cherry",
                "date",
                "elderberry",
                "fig",
                "grape",
                "zebra"
            ]
        );

        // Test with specific prefix
        tree.clear();
        tree.insert("test3", 30);
        tree.insert("test1", 10);
        tree.insert("test2", 20);
        tree.insert("testing", 40);
        tree.insert("tester", 50);

        let test_results: Vec<_> = tree.search_iter("test", SearchMode::Prefix).collect();
        let test_keys: Vec<String> = test_results
            .iter()
            .map(|(k, _)| String::from_utf8_lossy(k).to_string())
            .collect();

        assert_eq!(
            test_keys,
            vec!["test1", "test2", "test3", "tester", "testing"]
        );

        // Test with complex overlapping keys
        tree.clear();
        tree.insert("a", 1);
        tree.insert("ab", 2);
        tree.insert("abc", 3);
        tree.insert("abd", 4);
        tree.insert("ac", 5);
        tree.insert("b", 6);
        tree.insert("ba", 7);

        let a_results: Vec<_> = tree.search_iter("a", SearchMode::Prefix).collect();
        let a_keys: Vec<String> = a_results
            .iter()
            .map(|(k, _)| String::from_utf8_lossy(k).to_string())
            .collect();

        assert_eq!(a_keys, vec!["a", "ab", "abc", "abd", "ac"]);

        // Test with UTF-8 characters (should be sorted by byte order)
        tree.clear();
        tree.insert("café", 1);
        tree.insert("car", 2);
        tree.insert("care", 3);
        tree.insert("cat", 4);

        let c_results: Vec<_> = tree.search_iter("c", SearchMode::Prefix).collect();
        let c_keys: Vec<String> = c_results
            .iter()
            .map(|(k, _)| String::from_utf8_lossy(k).to_string())
            .collect();

        // Note: UTF-8 "é" has different byte values than ASCII, so café might not sort where expected
        // This test verifies the actual sorting behavior
        let mut expected = c_keys.clone();
        expected.sort();
        assert_eq!(c_keys, expected, "Keys should be in alphabetical order");

        // Verify that returned keys are actually sorted
        for i in 1..c_keys.len() {
            assert!(
                c_keys[i - 1] <= c_keys[i],
                "Keys not in order: '{}' should come before '{}'",
                c_keys[i - 1],
                c_keys[i]
            );
        }
    }

    #[test]
    fn test_mut_value_comprehensive() {
        let mut tree: RadixTree<String> = RadixTree::new();

        // Test basic create and modify operations
        tree.mut_value("hello", |value| {
            assert_eq!(*value, None);
            *value = Some("world".to_string());
        });

        // Verify it's stored correctly with search
        let results = tree.search_prefix("hello", SearchMode::Prefix);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, &"world".to_string());

        // Test modifying existing value
        tree.mut_value("hello", |value| {
            assert_eq!(*value, Some("world".to_string()));
            *value = Some("universe".to_string());
        });

        // Test with integer types for different behavior
        let mut int_tree: RadixTree<i32> = RadixTree::new();
        int_tree.mut_value("counter", |value| {
            assert_eq!(*value, None);
            *value = Some(42);
        });

        int_tree.mut_value("counter", |value| {
            assert_eq!(*value, Some(42));
            if let Some(v) = value {
                *v = 100;
            }
        });

        let int_results = int_tree.search_prefix("counter", SearchMode::Prefix);
        assert_eq!(int_results.len(), 1);
        assert_eq!(*int_results[0].1, 100);

        // Test mixed with insert operations
        int_tree.insert("inserted", 999);
        int_tree.mut_value("inserted", |value| {
            assert_eq!(*value, Some(999));
            *value = Some(1000);
        });

        int_tree.mut_value("new_via_mut", |value| {
            assert_eq!(*value, None);
            *value = Some(2000);
        });

        let mixed_results = int_tree.search_prefix("", SearchMode::Prefix);
        assert_eq!(mixed_results.len(), 3); // counter, inserted, new_via_mut
    }

    #[test]
    fn test_mut_value_complex_scenarios() {
        let mut tree: RadixTree<i32> = RadixTree::new();

        // Test complex node splitting scenarios
        tree.mut_value("abcdef", |value| {
            assert_eq!(*value, None);
            *value = Some(1);
        });

        // Verify first insertion
        let results = tree.search_prefix("abcdef", SearchMode::Prefix);
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, 1);

        tree.mut_value("abc", |value| {
            // Should split "abcdef"
            assert_eq!(*value, None);
            *value = Some(2);
        });

        // Verify split worked
        let results = tree.search_prefix("abc", SearchMode::Prefix);
        assert_eq!(results.len(), 2); // abc and abcdef

        tree.mut_value("abcxyz", |value| {
            // Should create sibling to "def"
            assert_eq!(*value, None);
            *value = Some(3);
        });

        tree.mut_value("ab", |value| {
            // Should split "abc"
            assert_eq!(*value, None);
            *value = Some(4);
        });

        // Test prefix relationships
        tree.mut_value("application", |value| {
            assert_eq!(*value, None);
            *value = Some(100);
        });

        tree.mut_value("app", |value| {
            // Shorter key that's a prefix
            assert_eq!(*value, None);
            *value = Some(200);
        });

        // Test overlapping keys
        tree.mut_value("testing", |value| {
            assert_eq!(*value, None);
            *value = Some(300);
        });

        tree.mut_value("test", |value| {
            assert_eq!(*value, None);
            *value = Some(400);
        });

        // Test empty key
        tree.mut_value("", |value| {
            assert_eq!(*value, None);
            *value = Some(999);
        });

        // Verify all complex scenarios work
        tree.mut_value("abcdef", |value| assert_eq!(*value, Some(1)));
        tree.mut_value("abc", |value| assert_eq!(*value, Some(2)));
        tree.mut_value("abcxyz", |value| assert_eq!(*value, Some(3)));
        tree.mut_value("ab", |value| assert_eq!(*value, Some(4)));
        tree.mut_value("app", |value| assert_eq!(*value, Some(200)));
        tree.mut_value("application", |value| assert_eq!(*value, Some(100)));
        tree.mut_value("test", |value| assert_eq!(*value, Some(400)));
        tree.mut_value("testing", |value| assert_eq!(*value, Some(300)));
        tree.mut_value("", |value| assert_eq!(*value, Some(999)));

        // Test search functionality with value verification
        let ab_results = tree.search_prefix("ab", SearchMode::Prefix);
        assert_eq!(ab_results.len(), 4); // ab, abc, abcdef, abcxyz

        let app_results = tree.search_prefix("app", SearchMode::Prefix);
        assert_eq!(app_results.len(), 2); // app, application

        let test_results = tree.search_prefix("test", SearchMode::Prefix);
        assert_eq!(test_results.len(), 2); // test, testing

        let all_results = tree.search_prefix("", SearchMode::Prefix);
        assert_eq!(all_results.len(), 9); // All inserted keys including empty
    }

    #[test]
    fn test_mut_value_return_values() {
        let mut tree: RadixTree<i32> = RadixTree::new();

        // Test returning whether a new value was created
        let was_new = tree.mut_value("counter", |value| {
            let is_new = value.is_none();
            *value = Some(1);
            is_new
        });
        assert!(was_new);

        // Test returning the modified value
        let new_value = tree.mut_value("counter", |value| {
            if let Some(v) = value {
                *v += 5;
                *v
            } else {
                0
            }
        });
        assert_eq!(new_value, 6);

        // Test returning current value without modifying
        let current = tree.mut_value("counter", |value| value.unwrap_or(0));
        assert_eq!(current, 6);

        // Test returning computed value based on key existence
        let result = tree.mut_value("new_key", |value| {
            if value.is_none() {
                *value = Some(100);
                "created"
            } else {
                "existed"
            }
        });
        assert_eq!(result, "created");

        let result2 = tree.mut_value("new_key", |value| {
            if value.is_none() {
                *value = Some(100);
                "created"
            } else {
                "existed"
            }
        });
        assert_eq!(result2, "existed");

        // Test returning complex computations
        let stats = tree.mut_value("stats", |value| {
            *value = Some(42);
            (true, 42, "initialized")
        });
        assert_eq!(stats, (true, 42, "initialized"));

        // Test with different return types
        let mut string_tree: RadixTree<String> = RadixTree::new();

        let length = string_tree.mut_value("text", |value| {
            *value = Some("hello world".to_string());
            value.as_ref().unwrap().len()
        });
        assert_eq!(length, 11);

        let uppercase = string_tree.mut_value("text", |value| {
            if let Some(s) = value {
                s.make_ascii_uppercase();
                s.clone()
            } else {
                String::new()
            }
        });
        assert_eq!(uppercase, "HELLO WORLD");
    }

    #[test]
    fn test_exact_matching() {
        let mut tree = RadixTree::new();

        // Insert test data
        tree.insert("hello", 1);
        tree.insert("help", 2);
        tree.insert("hell", 3);
        tree.insert("world", 4);

        // Test exact matching with search_prefix
        let exact_hello = tree.search_prefix("hello", SearchMode::Exact);
        assert_eq!(exact_hello.len(), 1);
        assert_eq!(exact_hello[0].0, "hello");
        assert_eq!(*exact_hello[0].1, 1);

        // Test exact matching with search_iter
        let exact_hello_iter: Vec<_> = tree.search_iter("hello", SearchMode::Exact).collect();
        assert_eq!(exact_hello_iter.len(), 1);
        assert_eq!(String::from_utf8_lossy(&exact_hello_iter[0].0), "hello");
        assert_eq!(*exact_hello_iter[0].1, 1);

        // Test exact matching that should return no results (prefix exists but not as exact key)
        let exact_hel = tree.search_prefix("hel", SearchMode::Exact);
        assert_eq!(exact_hel.len(), 0);

        let exact_hel_iter: Vec<_> = tree.search_iter("hel", SearchMode::Exact).collect();
        assert_eq!(exact_hel_iter.len(), 0);

        // Test non-exact matching for comparison
        let prefix_hel = tree.search_prefix("hel", SearchMode::Prefix);
        assert_eq!(prefix_hel.len(), 3); // hell, hello, help

        let prefix_hel_iter: Vec<_> = tree.search_iter("hel", SearchMode::Prefix).collect();
        assert_eq!(prefix_hel_iter.len(), 3);

        // Test exact matching with non-existent key
        let exact_nonexistent = tree.search_prefix("nonexistent", SearchMode::Exact);
        assert_eq!(exact_nonexistent.len(), 0);

        let exact_nonexistent_iter: Vec<_> =
            tree.search_iter("nonexistent", SearchMode::Exact).collect();
        assert_eq!(exact_nonexistent_iter.len(), 0);

        // Test exact matching with empty key
        tree.insert("", 999);
        let exact_empty = tree.search_prefix("", SearchMode::Exact);
        assert_eq!(exact_empty.len(), 1);
        assert_eq!(exact_empty[0].0, "");
        assert_eq!(*exact_empty[0].1, 999);

        // Test exact matching with UTF-8
        tree.insert("café", 100);
        tree.insert("cafe", 200);

        let exact_cafe_utf8 = tree.search_prefix("café", SearchMode::Exact);
        assert_eq!(exact_cafe_utf8.len(), 1);
        assert_eq!(exact_cafe_utf8[0].0, "café");
        assert_eq!(*exact_cafe_utf8[0].1, 100);

        let exact_cafe_ascii = tree.search_prefix("cafe", SearchMode::Exact);
        assert_eq!(exact_cafe_ascii.len(), 1);
        assert_eq!(exact_cafe_ascii[0].0, "cafe");
        assert_eq!(*exact_cafe_ascii[0].1, 200);
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut tree = RadixTree::new();

        // Test empty tree
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());

        // Test single insertion
        tree.insert("hello", 1);
        assert_eq!(tree.len(), 1);
        assert!(!tree.is_empty());

        // Test multiple insertions
        tree.insert("world", 2);
        tree.insert("help", 3);
        assert_eq!(tree.len(), 3);
        assert!(!tree.is_empty());

        // Test overlapping prefixes
        tree.insert("hell", 4);
        tree.insert("testing", 5);
        tree.insert("test", 6);
        assert_eq!(tree.len(), 6);
        assert!(!tree.is_empty());

        // Test with UTF-8 characters
        tree.insert("café", 100);
        tree.insert("🚀", 200);
        tree.insert("привет", 300);
        assert_eq!(tree.len(), 9);

        // Test with empty key
        tree.insert("", 999);
        assert_eq!(tree.len(), 10);

        // Test with special characters
        tree.insert("key:value", 50);
        tree.insert("/path/file", 60);
        tree.insert("user@domain", 70);
        assert_eq!(tree.len(), 13);

        // Test clear operation
        tree.clear();
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());

        // Test with mut_value operations
        tree.mut_value("counter", |value| {
            *value = Some(42);
        });
        assert_eq!(tree.len(), 1);
        assert!(!tree.is_empty());

        tree.mut_value("counter", |value| {
            if let Some(v) = value {
                *v += 1;
            }
        });
        assert_eq!(tree.len(), 1); // Should still be 1, just modified

        tree.mut_value("new_key", |value| {
            *value = Some(100);
        });
        assert_eq!(tree.len(), 2);

        // Test mixed operations
        tree.insert("mixed1", 1);
        tree.mut_value("mixed2", |value| *value = Some(2));
        assert_eq!(tree.len(), 4);

        // Test with different value types
        let mut string_tree: RadixTree<String> = RadixTree::new();
        assert_eq!(string_tree.len(), 0);
        assert!(string_tree.is_empty());

        string_tree.insert("key1", "value1".to_string());
        string_tree.insert("key2", "value2".to_string());
        assert_eq!(string_tree.len(), 2);
        assert!(!string_tree.is_empty());

        // Test complex node splitting scenarios affect count correctly
        let mut complex_tree = RadixTree::new();
        complex_tree.insert("abcdef", 1);
        assert_eq!(complex_tree.len(), 1);

        complex_tree.insert("abc", 2); // Should cause node splitting
        assert_eq!(complex_tree.len(), 2);

        complex_tree.insert("abcxyz", 3); // Should add sibling
        assert_eq!(complex_tree.len(), 3);

        complex_tree.insert("ab", 4); // Should cause further splitting
        assert_eq!(complex_tree.len(), 4);
    }

    #[test]
    fn test_search_iter_value() {
        let mut tree = RadixTree::new();

        // Insert test data
        tree.insert("hello", 1);
        tree.insert("help", 2);
        tree.insert("hell", 3);
        tree.insert("world", 4);
        tree.insert("test", 5);
        tree.insert("testing", 6);

        // Test prefix search with search_iter_value
        let values: Vec<_> = tree.search_iter_value("hel", SearchMode::Prefix).collect();
        assert_eq!(values.len(), 3);
        let mut sorted_values = values.iter().cloned().cloned().collect::<Vec<_>>();
        sorted_values.sort();
        assert_eq!(sorted_values, vec![1, 2, 3]); // hello=1, help=2, hell=3

        // Test exact search with search_iter_value
        let exact_values: Vec<_> = tree.search_iter_value("hello", SearchMode::Exact).collect();
        assert_eq!(exact_values.len(), 1);
        assert_eq!(*exact_values[0], 1);

        // Test exact search that should return no results
        let no_exact_values: Vec<_> = tree.search_iter_value("hel", SearchMode::Exact).collect();
        assert_eq!(no_exact_values.len(), 0);

        // Test with non-existent prefix
        let no_values: Vec<_> = tree.search_iter_value("xyz", SearchMode::Prefix).collect();
        assert_eq!(no_values.len(), 0);

        // Test with empty prefix (should return all values)
        let all_values: Vec<_> = tree.search_iter_value("", SearchMode::Prefix).collect();
        assert_eq!(all_values.len(), 6);
        let mut all_sorted = all_values.iter().cloned().cloned().collect::<Vec<_>>();
        all_sorted.sort();
        assert_eq!(all_sorted, vec![1, 2, 3, 4, 5, 6]);

        // Test with UTF-8 characters
        tree.insert("café", 100);
        tree.insert("🚀", 200);
        tree.insert("🚀🌟", 300);

        let cafe_values: Vec<_> = tree.search_iter_value("café", SearchMode::Exact).collect();
        assert_eq!(cafe_values.len(), 1);
        assert_eq!(*cafe_values[0], 100);

        let emoji_values: Vec<_> = tree.search_iter_value("🚀", SearchMode::Prefix).collect();
        assert_eq!(emoji_values.len(), 2);
        let mut emoji_sorted = emoji_values.iter().cloned().cloned().collect::<Vec<_>>();
        emoji_sorted.sort();
        assert_eq!(emoji_sorted, vec![200, 300]); // 🚀=200, 🚀🌟=300

        // Test with different value types
        let mut string_tree: RadixTree<String> = RadixTree::new();
        string_tree.insert("key1", "value1".to_string());
        string_tree.insert("key2", "value2".to_string());
        string_tree.insert("key10", "value10".to_string());

        let string_values: Vec<_> = string_tree
            .search_iter_value("key", SearchMode::Prefix)
            .collect();
        assert_eq!(string_values.len(), 3);
        let mut string_sorted: Vec<String> = string_values.iter().cloned().cloned().collect();
        string_sorted.sort();
        assert_eq!(string_sorted, vec!["value1", "value10", "value2"]);

        // Test empty tree
        let empty_tree: RadixTree<i32> = RadixTree::new();
        let empty_values: Vec<_> = empty_tree
            .search_iter_value("anything", SearchMode::Prefix)
            .collect();
        assert_eq!(empty_values.len(), 0);

        // Test consistency with search_iter
        let iter_results: Vec<_> = tree.search_iter("test", SearchMode::Prefix).collect();
        let iter_value_results: Vec<_> =
            tree.search_iter_value("test", SearchMode::Prefix).collect();

        assert_eq!(iter_results.len(), iter_value_results.len());
        for (i, (_, expected_value)) in iter_results.iter().enumerate() {
            assert_eq!(*expected_value, iter_value_results[i]);
        }
    }
}
