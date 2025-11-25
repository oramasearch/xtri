use crate::SearchMode;
use crate::tree::{RadixNode, common_prefix_length};

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
    pub(crate) fn new(root: &'a RadixNode<T>, prefix: &[u8], mode: SearchMode) -> Self {
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
            let common_len = common_prefix_length(&child.key, prefix);

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
