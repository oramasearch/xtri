use crate::tree::RadixNode;

#[derive(Default)]
struct BufferCache {
    prev_row: Vec<usize>,
    curr_row: Vec<usize>,
}

/// Checks if search term is approximately a prefix of target within max_distance.
///
/// A search term S is considered a fuzzy prefix of key K with distance d if there
/// exists a prefix P of K where `levenshtein_distance(S, P) <= d`.
///
/// # Arguments
/// * `search_term` - The search query bytes
/// * `target` - The key bytes to check against
/// * `max_distance` - Maximum edit distance allowed
///
/// # Returns
/// `Some(distance)` if a match is found, `None` otherwise
fn is_fuzzy_prefix_match(
    search_term: &[u8],
    target: &[u8],
    max_distance: u8,
    cache: &mut BufferCache,
) -> Option<usize> {
    if search_term.is_empty() {
        return Some(0); // Empty search matches everything with distance 0
    }

    let max_dist = max_distance as usize;
    let search_len = search_term.len();

    // Check prefixes of target from (search_len - max_dist) to (search_len + max_dist)
    // This captures all possible fuzzy prefix matches efficiently
    let min_prefix_len = search_len.saturating_sub(max_dist).max(1);
    let max_prefix_len = std::cmp::min(target.len(), search_len + max_dist);

    let mut best_distance = usize::MAX;

    for prefix_len in min_prefix_len..=max_prefix_len {
        if prefix_len > target.len() {
            continue;
        }

        let dist = levenshtein_distance(search_term, &target[0..prefix_len], cache);
        best_distance = std::cmp::min(best_distance, dist);
    }

    if best_distance <= max_dist {
        Some(best_distance)
    } else {
        None
    }
}

/// Computes the Levenshtein (edit) distance between two byte sequences.
///
/// Uses Wagner-Fischer algorithm with space optimization (two rows only).
/// This operates at the byte level, so UTF-8 multi-byte characters may have
/// distance greater than 1 (e.g., "café" vs "cafe" = distance 2).
///
/// # Arguments
/// * `a` - First byte sequence
/// * `b` - Second byte sequence
///
/// # Returns
/// The minimum number of single-byte edits (insertions, deletions, substitutions)
/// required to transform `a` into `b`.
fn levenshtein_distance(a: &[u8], b: &[u8], cache: &mut BufferCache) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }

    let m = a.len();
    let n = b.len();

    // Use two rows for space optimization O(min(m,n)) instead of O(m*n)
    let prev_row = &mut cache.prev_row;
    let curr_row = &mut cache.curr_row;
    if prev_row.len() < n + 1 {
        prev_row.resize(n + 1, 0);
    }
    if curr_row.len() < n + 1 {
        curr_row.resize(n + 1, 0);
    }

    // Initialize first row
    #[allow(clippy::needless_range_loop)]
    for j in 0..=n {
        prev_row[j] = j;
    }

    // Fill matrix row by row
    for i in 1..=m {
        curr_row[0] = i;

        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };

            curr_row[j] = std::cmp::min(
                std::cmp::min(
                    prev_row[j] + 1,     // Deletion
                    curr_row[j - 1] + 1, // Insertion
                ),
                prev_row[j - 1] + cost, // Substitution
            );
        }

        std::mem::swap(prev_row, curr_row);
    }

    prev_row[n]
}

/// Iterator for typo-tolerant fuzzy prefix search in a radix tree.
///
/// This iterator finds all keys where the search term is approximately a prefix,
/// allowing for typos within a specified edit distance tolerance. Uses Levenshtein
/// distance to measure similarity.
///
/// Results are returned in alphabetical order with their edit distances.
/// Memory usage is O(tree depth) using a stack-based traversal approach.
pub struct TypoTolerantSearchIterator<'a, T> {
    // Stack of (node, child_index, current_key, _unused) for traversal
    stack: Vec<(&'a RadixNode<T>, usize, Vec<u8>, u8)>,
    // Original search key for distance computation
    search_key: Vec<u8>,
    // Maximum Levenshtein distance allowed
    max_distance: u8,
    // Buffer cache for distance computations
    cache: BufferCache,
}

impl<'a, T> TypoTolerantSearchIterator<'a, T> {
    pub(crate) fn new(root: &'a RadixNode<T>, search_key: &[u8], max_distance: u8) -> Self {
        let mut iterator = Self {
            stack: Vec::new(),
            search_key: search_key.to_vec(),
            max_distance,
            cache: Default::default(),
        };

        // Start from root - can't optimize starting point for fuzzy search
        // Must explore all branches to find fuzzy matches
        iterator.stack.push((root, 0, Vec::new(), 0));
        iterator
    }
}

impl<'a, T> Iterator for TypoTolerantSearchIterator<'a, T> {
    type Item = (String, &'a T, u8);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (node, child_index, current_key, _) = self.stack.pop()?;

            // First visit: check if node has a value that matches
            if child_index == 0 {
                if let Some(ref value) = node.value {
                    // Check if current_key is a fuzzy match for our search
                    if let Some(distance) = is_fuzzy_prefix_match(
                        &self.search_key,
                        &current_key,
                        self.max_distance,
                        &mut self.cache,
                    ) {
                        // Found a match! Convert to String and return
                        let key_string = String::from_utf8_lossy(&current_key).to_string();

                        // Push back to continue with children
                        if !node.children.is_empty() {
                            self.stack.push((node, 1, current_key, 0));
                        }

                        return Some((key_string, value, distance as u8));
                    }
                }

                // No match or no value, start processing children
                if !node.children.is_empty() {
                    self.stack.push((node, 1, current_key, 0));
                }
                continue;
            }

            // Processing children
            let current_child_idx = child_index - 1;

            if current_child_idx < node.children.len() {
                // Push next sibling
                if current_child_idx + 1 < node.children.len() {
                    self.stack
                        .push((node, child_index + 1, current_key.clone(), 0));
                }

                // Process current child
                let (_, child) = &node.children[current_child_idx];
                let mut child_key = current_key;
                child_key.extend_from_slice(&child.key);

                // Always explore children - pruning would be too complex for fuzzy prefix matching
                // since we need to check if the search term is approximately a prefix of ANY
                // prefix of the stored keys
                self.stack.push((child, 0, child_key, 0));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_distance_basic() {
        let mut cache = BufferCache::default();

        // Empty strings
        assert_eq!(levenshtein_distance(b"", b"", &mut cache), 0);
        assert_eq!(levenshtein_distance(b"abc", b"", &mut cache), 3);
        assert_eq!(levenshtein_distance(b"", b"xyz", &mut cache), 3);

        // Identical strings
        assert_eq!(levenshtein_distance(b"abc", b"abc", &mut cache), 0);
        assert_eq!(levenshtein_distance(b"hello", b"hello", &mut cache), 0);
        // Single edits
        assert_eq!(levenshtein_distance(b"abc", b"abd", &mut cache), 1); // Substitution
        assert_eq!(levenshtein_distance(b"abc", b"abcd", &mut cache), 1); // Insertion
        assert_eq!(levenshtein_distance(b"abcd", b"abc", &mut cache), 1); // Deletion
        // Classic examples
        assert_eq!(levenshtein_distance(b"kitten", b"sitting", &mut cache), 3);
        assert_eq!(levenshtein_distance(b"saturday", b"sunday", &mut cache), 3);

        assert_eq!(levenshtein_distance(b"abdc", b"abc", &mut cache), 1);
        assert_eq!(levenshtein_distance(b"abc", b"abdc", &mut cache), 1);
        assert_eq!(levenshtein_distance(b"abc", b"ac", &mut cache), 1);
        assert_eq!(levenshtein_distance(b"ac", b"abc", &mut cache), 1);
    }

    #[test]
    fn test_levenshtein_distance_utf8() {
        let mut cache = BufferCache::default();

        // UTF-8 multi-byte characters (byte-level distance)
        // "café" has é which is 2 bytes (0xC3 0xA9), "cafe" has e which is 1 byte
        // So distance is 2 (one deletion of 0xC3, one substitution of 0xA9 to 'e')
        assert_eq!(
            levenshtein_distance("café".as_bytes(), "cafe".as_bytes(), &mut cache),
            2
        );

        // Emoji (🚀 is 4 bytes)
        assert_eq!(
            levenshtein_distance("🚀".as_bytes(), "x".as_bytes(), &mut cache),
            4
        );

        // Same multi-byte character
        assert_eq!(
            levenshtein_distance("café".as_bytes(), "café".as_bytes(), &mut cache),
            0
        );
    }

    #[test]
    fn test_fuzzy_prefix_match_basic() {
        let mut cache = BufferCache::default();

        // Exact prefix (distance 0)
        assert_eq!(
            is_fuzzy_prefix_match(b"hel", b"hello", 0, &mut cache),
            Some(0)
        );
        assert_eq!(
            is_fuzzy_prefix_match(b"hello", b"hello", 0, &mut cache),
            Some(0)
        );

        // One edit (distance 1)
        assert_eq!(
            is_fuzzy_prefix_match(b"helo", b"hello", 1, &mut cache),
            Some(1)
        ); // Missing 'l'
        assert_eq!(
            is_fuzzy_prefix_match(b"hel", b"hello", 1, &mut cache),
            Some(0)
        ); // Exact match

        // Too far (no match)
        assert_eq!(is_fuzzy_prefix_match(b"xyz", b"abc", 2, &mut cache), None);
        assert_eq!(
            is_fuzzy_prefix_match(b"hello", b"world", 2, &mut cache),
            None
        );

        // Empty search term
        assert_eq!(
            is_fuzzy_prefix_match(b"", b"anything", 1, &mut cache),
            Some(0)
        );
    }

    #[test]
    fn test_fuzzy_prefix_match_edge_cases() {
        let mut cache = BufferCache::default();

        // Search term longer than target
        assert_eq!(is_fuzzy_prefix_match(b"hello", b"hel", 1, &mut cache), None);

        // Distance 2 allows more flexibility
        assert_eq!(
            is_fuzzy_prefix_match(b"helo", b"hello", 2, &mut cache),
            Some(1)
        );
        assert_eq!(
            is_fuzzy_prefix_match(b"hllo", b"hello", 2, &mut cache),
            Some(1)
        ); // Missing 'e'
        assert_eq!(
            is_fuzzy_prefix_match(b"hlo", b"hello", 2, &mut cache),
            Some(2)
        ); // Missing 'e' and 'l'
    }
}
