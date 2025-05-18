use std::{
    collections::{HashMap, VecDeque},
    ops::Range,
};

use language::{BracketPair, BufferSnapshot};

use tree_sitter::{QueryCursor, StreamingIterator}; // Assuming this is zed_core::RopeExt

// Number of distinct colors to cycle through for bracket pairs.
pub const NUM_BRACKET_COLORS: usize = 6;

// Information about an encountered open bracket.
#[derive(Debug)]
struct OpenBracketInfo {
    // The text of the opening bracket (e.g., "(", "{").
    opener_text: String,
    // The color index assigned to this bracket.
    color_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BracketOccurrence {
    range: Range<usize>,
    text: String,
    is_open: bool,
    // Used for tie-breaking in sort for overlapping ranges.
    // Sort by start offset, then by !is_open (open before close),
    // then by a length-based criterion (longer openers first, shorter closers first).
    sort_key: (usize, bool, isize),
}

impl Ord for BracketOccurrence {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sort_key.cmp(&other.sort_key)
    }
}

impl PartialOrd for BracketOccurrence {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Identifies all bracket characters in the snapshot and assigns them a color index
/// based on their nesting level.
///
/// Returns a vector of (Range<usize>, usize) where Range is the byte offset of the bracket
/// character and usize is the color index.
pub fn colorize_bracket_pairs(snapshot: &BufferSnapshot) -> Vec<(Range<usize>, usize)> {
    let mut colored_brackets = Vec::new();
    let mut open_bracket_stack: VecDeque<OpenBracketInfo> = VecDeque::new();

    let Some(language) = snapshot.language() else {
        return colored_brackets;
    };

    let scope = language.default_scope();
    let bracket_pairs: Vec<&BracketPair> = scope
        .brackets()
        .filter_map(|(pair, enabled)| if enabled { Some(pair) } else { None })
        .collect();

    if bracket_pairs.is_empty() {
        return colored_brackets;
    }

    let mut open_bracket_texts: Vec<&str> = Vec::new();
    let mut close_bracket_texts: Vec<&str> = Vec::new();
    let mut bracket_pair_map: HashMap<&str, &BracketPair> = HashMap::new();

    for pair in &bracket_pairs {
        open_bracket_texts.push(&pair.start);
        close_bracket_texts.push(&pair.end);
        bracket_pair_map.insert(&pair.start, pair);
        bracket_pair_map.insert(&pair.end, pair); // Map closing text to its pair too
    }

    let grammar = match language.grammar() {
        Some(g) => g,
        None => return colored_brackets,
    };
    let brackets_config = match grammar.brackets_config() {
        Some(cfg) => cfg,
        None => return colored_brackets,
    };
    let brackets_query = brackets_config.query();

    let mut cursor = QueryCursor::new();
    // Use the root node from the first syntax layer, if available.
    let mut syntax_layers = snapshot.syntax_layers();
    let root_node = match syntax_layers.next() {
        Some(layer) => layer.node(),
        None => return colored_brackets,
    };
    let rope = snapshot.text.as_rope();
    let rope_string = rope.to_string();
    let mut occurrences = Vec::new();

    let mut matches = cursor.matches(brackets_query, root_node, rope_string.as_bytes());

    while let Some(m) = matches.next() {
        for capture in m.captures {
            let node = capture.node;
            let range = node.byte_range();
            // text_for_range returns an iterator of chunks, so collect and concatenate
            let text: String = snapshot.text_for_range(range.clone()).collect();

            let capture_name = &brackets_query.capture_names()[capture.index as usize];
            let is_open = match capture_name.as_ref() {
                "open" => true,
                "close" => false,
                // Other captures like "pair" or language-specific ones are ignored for basic colorization
                _ => continue,
            };

            // Ensure this bracket is part of the configured pairs for the language
            if (is_open && open_bracket_texts.contains(&text.as_str()))
                || (!is_open && close_bracket_texts.contains(&text.as_str()))
            {
                // For sorting: open brackets get negative length, close brackets get positive length.
                // This helps in prioritizing: longer opening brackets first, shorter closing brackets first
                // when start offsets are the same.
                let sort_len_criteria = if is_open {
                    -(text.len() as isize)
                } else {
                    text.len() as isize
                };
                occurrences.push(BracketOccurrence {
                    range: range.clone(),
                    text,
                    is_open,
                    sort_key: (range.start, !is_open, sort_len_criteria),
                });
            }
        }
    }

    occurrences.sort_unstable();

    for occurrence in occurrences {
        if occurrence.is_open {
            let color_index = open_bracket_stack.len() % NUM_BRACKET_COLORS;
            open_bracket_stack.push_back(OpenBracketInfo {
                opener_text: occurrence.text.clone(),
                color_index,
            });
            colored_brackets.push((occurrence.range, color_index));
        } else {
            // This is a closing bracket
            if let Some(matching_open_info) = open_bracket_stack.back() {
                // Check if the current closing bracket corresponds to the expected open bracket
                // by looking up the configured pair for the *opener* on the stack.
                if let Some(configured_pair_for_opener) =
                    bracket_pair_map.get(matching_open_info.opener_text.as_str())
                {
                    if configured_pair_for_opener.end == occurrence.text {
                        // Correctly matched pair
                        let info = open_bracket_stack.pop_back().unwrap(); // Known to exist
                        colored_brackets.push((occurrence.range, info.color_index));
                    } else {
                        // Mismatched closing bracket (e.g. `([)]`).
                        // For now, we don't color mismatched closing brackets.
                        // We also don't pop from the stack, as the open bracket is still expecting its match.
                    }
                }
                // If configured_pair_for_opener is None, it's an internal logic error or misconfiguration.
            } else {
                // Unmatched closing bracket (e.g., `)]` at the start of a file).
                // We don't color these.
            }
        }
    }
    colored_brackets
}

pub fn colorize_bracket_pairs_for_multibuffer(
    snapshot: &multi_buffer::MultiBufferSnapshot,
) -> Vec<(Range<usize>, usize)> {
    if let Some((_, _, buffer_snapshot)) = snapshot.as_singleton() {
        return colorize_bracket_pairs(buffer_snapshot);
    }

    // For non-singleton buffers, we can't currently colorize brackets
    Vec::new()
}

#[cfg(test)]
mod tests {
    // TODO: Add tests for colorize_bracket_pairs
    // - Basic matching pairs: (), {}, []
    // - Nested pairs
    // - Multiple pairs on the same level
    // - Mismatched pairs (e.g., ([)] )
    // - Unclosed pairs (e.g., ( [ )
    // - Unopened pairs (e.g., ) [ ] )
    // - Correct color cycling
    // - Empty input
    // - Input with no brackets
    // - Interaction with comments/strings (ensure brackets inside are ignored - this depends on tree-sitter query)
    // - Languages with multi-character brackets (e.g. begin/end, if query supports them)
    // - Test sort_key logic for overlapping/adjacent brackets (e.g. JSX <Foo attribute={<Bar />}>)
}
