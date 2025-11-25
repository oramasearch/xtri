use crate::tree::RadixNode;

/// Iterator for traversing only the leaf nodes of a radix tree.
///
/// A leaf node is defined as a node that has a value but no children,
/// representing the end of a key path with no further extensions.
/// This iterator performs lazy traversal using a stack-based approach.
/// Memory usage is O(tree depth) storing only node references and child indices.
pub struct LeavesIterator<'a, T> {
    // Stack of (node, child_index, current_key) pairs for traversal state
    stack: Vec<(&'a RadixNode<T>, usize, Vec<u8>)>,
}

impl<'a, T> LeavesIterator<'a, T> {
    pub(crate) fn new(root: &'a RadixNode<T>) -> Self {
        let mut iterator = Self { stack: Vec::new() };

        // Start traversal from the root with empty key
        iterator.stack.push((root, 0, Vec::new()));
        iterator
    }
}

impl<'a, T> Iterator for LeavesIterator<'a, T> {
    type Item = (Vec<u8>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (node, child_index, current_key) = self.stack.pop()?;

            // If this is the first visit to this node (child_index == 0), check if it's a leaf
            if child_index == 0 {
                // A leaf node has a value and no children
                if let Some(ref value) = node.value {
                    if node.children.is_empty() {
                        // This is a leaf node - return it
                        return Some((current_key.clone(), value));
                    }
                }

                // Not a leaf or no value, start processing children
                if !node.children.is_empty() {
                    self.stack.push((node, 1, current_key));
                }
                continue;
            }

            // Processing children: child_index - 1 is the current child being processed
            let current_child_idx = child_index - 1;

            if current_child_idx < node.children.len() {
                // Push the next child index for this node
                if current_child_idx + 1 < node.children.len() {
                    self.stack
                        .push((node, child_index + 1, current_key.clone()));
                }

                // Push the current child to be processed
                let (_, child) = &node.children[current_child_idx];
                let mut child_key = current_key;
                child_key.extend_from_slice(&child.key);

                self.stack.push((child, 0, child_key));
            }
        }
    }
}
