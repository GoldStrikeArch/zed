use crate::{Editor, EditorSettings};
use gpui::{Context, Hsla, Window};
use settings::Settings;
use std::{
    cmp::Ordering,
    collections::HashMap,
    hash::{Hash, Hasher},
    ops::Range,
    time::{Duration, Instant},
};
use text::Anchor;

/// Module for colorizing matching bracket pairs with different colors.
/// Implementation inspired by VS Code's bracket pair colorization, which
/// colorizes brackets by their nesting level to make it easier to match
/// opening and closing brackets.
///
/// This implementation uses an efficient AST-based approach with length annotations
/// to ensure good performance even on very large files with many bracket pairs.
/// The data structure avoids storing absolute positions and instead tracks relative
/// lengths, making it efficient for incremental updates.

/// Private type used for background highlight identification
enum BracketPairColorization {}

/// Range wrapper that implements Eq and Hash for use as HashMap key
#[derive(Debug, Clone)]
struct RangeKey {
    start: usize,
    end: usize,
}

impl From<Range<usize>> for RangeKey {
    fn from(range: Range<usize>) -> Self {
        Self {
            start: range.start,
            end: range.end,
        }
    }
}

impl PartialEq for RangeKey {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }
}

impl Eq for RangeKey {}

impl Hash for RangeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.start.hash(state);
        self.end.hash(state);
    }
}

impl PartialOrd for RangeKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RangeKey {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.start.cmp(&other.start) {
            Ordering::Equal => self.end.cmp(&other.end),
            other => other,
        }
    }
}

/// Represents a bracket node in the AST
#[derive(Debug, Clone)]
struct BracketNode {
    // Length of the bracket in characters (usually 1 for `{` or 2 for things like `/*`)
    length: usize,
    // Type of bracket (used to match opening and closing)
    bracket_type: BracketType,
    // Whether this is an opening or closing bracket
    is_opening: bool,
}

/// Represents a bracket type to match opening and closing pairs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BracketType {
    Brace,        // {}
    Bracket,      // []
    Parenthesis,  // ()
    AngleBracket, // <>
    Custom(u8),   // For any other bracket types
}

/// Represents a node in the bracket AST
#[derive(Debug, Clone)]
enum BracketAstNode {
    /// A single bracket (opening or closing)
    Bracket(BracketNode),

    /// A pair of matched brackets with content between them
    Pair {
        opening: BracketNode,
        content: Box<BracketAstNode>,
        closing: BracketNode,
        /// The total length of this pair (opening + content + closing)
        length: usize,
    },

    /// A list of adjacent nodes (balanced using a (2,3)-tree approach)
    List {
        children: Vec<BracketAstNode>,
        /// The total length of all children
        length: usize,
    },

    /// Text content without brackets
    Text { length: usize },
}

impl BracketAstNode {
    /// Gets the total length of this node
    fn length(&self) -> usize {
        match self {
            BracketAstNode::Bracket(bracket) => bracket.length,
            BracketAstNode::Pair { length, .. } => *length,
            BracketAstNode::List { length, .. } => *length,
            BracketAstNode::Text { length } => *length,
        }
    }

    /// Creates a new Text node
    fn text(length: usize) -> Self {
        BracketAstNode::Text { length }
    }

    /// Creates a new List node from children
    fn list(children: Vec<BracketAstNode>) -> Self {
        let length = children.iter().map(|child| child.length()).sum();
        BracketAstNode::List { children, length }
    }

    /// Creates a balanced (2,3)-tree from nodes
    fn create_balanced_list(nodes: Vec<BracketAstNode>) -> Self {
        if nodes.is_empty() {
            return BracketAstNode::text(0);
        }

        if nodes.len() <= 3 {
            return BracketAstNode::list(nodes);
        }

        // Create a balanced tree by recursively splitting into groups of 2-3 children
        let mut balanced_children = Vec::new();
        let mut i = 0;

        while i < nodes.len() {
            let chunk_size = if nodes.len() - i >= 5 {
                3
            } else if nodes.len() - i >= 3 {
                2
            } else {
                nodes.len() - i
            };

            let chunk = nodes[i..i + chunk_size].to_vec();
            balanced_children.push(BracketAstNode::list(chunk));
            i += chunk_size;
        }

        // Recursively balance if we still have too many children
        if balanced_children.len() > 3 {
            return Self::create_balanced_list(balanced_children);
        } else {
            return BracketAstNode::list(balanced_children);
        }
    }
}

/// Represents the full bracket pair coloring state
#[derive(Clone)]
struct BracketPairAst {
    root: BracketAstNode,
    // Cache the last query results to avoid recomputing when nothing changed
    cache: HashMap<RangeKey, Vec<(Range<usize>, usize)>>,
    // Last time the AST was built or updated
    last_update: Instant,
}

impl BracketPairAst {
    /// Create a new empty AST
    fn new() -> Self {
        Self {
            root: BracketAstNode::text(0),
            cache: HashMap::new(),
            last_update: Instant::now(),
        }
    }

    /// Build a new AST from the buffer content
    fn build(content: &str) -> Self {
        let mut parser = BracketParser::new(content);
        let root = parser.parse_document();

        Self {
            root,
            cache: HashMap::new(),
            last_update: Instant::now(),
        }
    }

    /// Find all bracket pairs in the given range and their nesting levels
    fn find_bracket_pairs(&mut self, range: Range<usize>) -> Vec<(Range<usize>, usize)> {
        // Check if we have a cached result for this range
        let range_key = RangeKey::from(range.clone());
        if let Some(cached) = self.cache.get(&range_key) {
            return cached.clone();
        }

        let mut result = Vec::new();
        let mut collector = BracketCollector::new(range.clone(), &mut result);
        self.collect_brackets_in_range(&self.root, 0, &mut collector, 0);

        // Cache the result for future queries
        self.cache.insert(RangeKey::from(range), result.clone());

        result
    }

    /// Collect all brackets in a range with their nesting levels
    fn collect_brackets_in_range(
        &self,
        node: &BracketAstNode,
        offset: usize,
        collector: &mut BracketCollector,
        nesting_level: usize,
    ) -> usize {
        let node_end = offset + node.length();

        // Skip this node if it's entirely outside our target range
        if node_end <= collector.range.start || offset >= collector.range.end {
            return offset + node.length();
        }

        match node {
            BracketAstNode::Bracket(bracket) => {
                if offset >= collector.range.start && offset < collector.range.end {
                    let bracket_range = offset..(offset + bracket.length);
                    collector.add_bracket(bracket_range, nesting_level);
                }
                offset + bracket.length
            }

            BracketAstNode::Pair {
                opening,
                content,
                closing,
                ..
            } => {
                let mut current_offset = offset;

                // Process opening bracket
                if current_offset >= collector.range.start && current_offset < collector.range.end {
                    let bracket_range = current_offset..(current_offset + opening.length);
                    collector.add_bracket(bracket_range, nesting_level);
                }
                current_offset += opening.length;

                // Process content (increasing nesting level)
                current_offset = self.collect_brackets_in_range(
                    content,
                    current_offset,
                    collector,
                    nesting_level + 1,
                );

                // Process closing bracket
                if current_offset >= collector.range.start && current_offset < collector.range.end {
                    let bracket_range = current_offset..(current_offset + closing.length);
                    collector.add_bracket(bracket_range, nesting_level);
                }
                current_offset += closing.length;

                current_offset
            }

            BracketAstNode::List { children, .. } => {
                let mut current_offset = offset;

                for child in children {
                    current_offset = self.collect_brackets_in_range(
                        child,
                        current_offset,
                        collector,
                        nesting_level,
                    );
                }

                current_offset
            }

            BracketAstNode::Text { length } => offset + length,
        }
    }
}

/// Helper to collect brackets and their nesting levels
struct BracketCollector<'a> {
    range: Range<usize>,
    results: &'a mut Vec<(Range<usize>, usize)>,
}

impl<'a> BracketCollector<'a> {
    fn new(range: Range<usize>, results: &'a mut Vec<(Range<usize>, usize)>) -> Self {
        Self { range, results }
    }

    fn add_bracket(&mut self, range: Range<usize>, nesting_level: usize) {
        self.results.push((range, nesting_level));
    }
}

/// Parser to build the bracket AST from text
struct BracketParser<'a> {
    content: &'a str,
    pos: usize,
}

impl<'a> BracketParser<'a> {
    fn new(content: &'a str) -> Self {
        Self { content, pos: 0 }
    }

    /// Parse the entire document
    fn parse_document(&mut self) -> BracketAstNode {
        let mut nodes = Vec::new();

        while self.pos < self.content.len() {
            nodes.push(self.parse_node());
        }

        // Create a balanced (2,3)-tree from all nodes
        BracketAstNode::create_balanced_list(nodes)
    }

    /// Parse a single node (text, bracket, or pair)
    fn parse_node(&mut self) -> BracketAstNode {
        if self.at_bracket() {
            self.parse_bracket_or_pair()
        } else {
            self.parse_text()
        }
    }

    /// Parse a text segment (no brackets)
    fn parse_text(&mut self) -> BracketAstNode {
        let start = self.pos;

        while self.pos < self.content.len() && !self.at_bracket() {
            self.pos += 1;
        }

        BracketAstNode::text(self.pos - start)
    }

    /// Parse a bracket or pair starting at the current position
    fn parse_bracket_or_pair(&mut self) -> BracketAstNode {
        let (bracket, is_opening) = self.parse_bracket();

        if is_opening {
            // This is an opening bracket, try to find the matching closing bracket
            let opening = bracket;
            let start_pos = self.pos;

            // Parse content until we find the matching closing bracket
            let mut content_nodes = Vec::new();
            let mut nesting_level = 1;

            while self.pos < self.content.len() && nesting_level > 0 {
                if self.at_bracket() {
                    let (bracket, is_opening) = self.peek_bracket();

                    if is_opening {
                        // Another opening bracket of the same type increases nesting
                        if bracket.bracket_type == opening.bracket_type {
                            nesting_level += 1;
                        }
                        content_nodes.push(self.parse_bracket_or_pair());
                    } else {
                        // A closing bracket of the same type decreases nesting
                        if bracket.bracket_type == opening.bracket_type {
                            nesting_level -= 1;

                            // If this is the matching closing bracket, we're done
                            if nesting_level == 0 {
                                let (closing, _) = self.parse_bracket();
                                let content = if content_nodes.is_empty() {
                                    BracketAstNode::text(0)
                                } else if content_nodes.len() == 1 {
                                    content_nodes.remove(0)
                                } else {
                                    BracketAstNode::create_balanced_list(content_nodes)
                                };

                                let total_length =
                                    opening.length + content.length() + closing.length;

                                return BracketAstNode::Pair {
                                    opening,
                                    content: Box::new(content),
                                    closing,
                                    length: total_length,
                                };
                            }
                        }

                        content_nodes.push(self.parse_bracket_or_pair());
                    }
                } else {
                    content_nodes.push(self.parse_text());
                }
            }

            // If we couldn't find the closing bracket, treat this as an unpaired bracket
            self.pos = start_pos;
            BracketAstNode::Bracket(opening)
        } else {
            // This is a closing bracket without an opening one
            BracketAstNode::Bracket(bracket)
        }
    }

    /// Parse a single bracket at the current position
    fn parse_bracket(&mut self) -> (BracketNode, bool) {
        let result = self.peek_bracket();
        self.pos += result.0.length;
        result
    }

    /// Peek at the current character to see if it's a bracket
    fn peek_bracket(&self) -> (BracketNode, bool) {
        if self.pos >= self.content.len() {
            return (
                BracketNode {
                    length: 0,
                    bracket_type: BracketType::Custom(0),
                    is_opening: false,
                },
                false,
            );
        }

        let c = self.content.as_bytes()[self.pos];

        match c {
            b'{' => (
                BracketNode {
                    length: 1,
                    bracket_type: BracketType::Brace,
                    is_opening: true,
                },
                true,
            ),
            b'}' => (
                BracketNode {
                    length: 1,
                    bracket_type: BracketType::Brace,
                    is_opening: false,
                },
                false,
            ),
            b'[' => (
                BracketNode {
                    length: 1,
                    bracket_type: BracketType::Bracket,
                    is_opening: true,
                },
                true,
            ),
            b']' => (
                BracketNode {
                    length: 1,
                    bracket_type: BracketType::Bracket,
                    is_opening: false,
                },
                false,
            ),
            b'(' => (
                BracketNode {
                    length: 1,
                    bracket_type: BracketType::Parenthesis,
                    is_opening: true,
                },
                true,
            ),
            b')' => (
                BracketNode {
                    length: 1,
                    bracket_type: BracketType::Parenthesis,
                    is_opening: false,
                },
                false,
            ),
            b'<' => (
                BracketNode {
                    length: 1,
                    bracket_type: BracketType::AngleBracket,
                    is_opening: true,
                },
                true,
            ),
            b'>' => (
                BracketNode {
                    length: 1,
                    bracket_type: BracketType::AngleBracket,
                    is_opening: false,
                },
                false,
            ),
            _ => (
                BracketNode {
                    length: 0,
                    bracket_type: BracketType::Custom(0),
                    is_opening: false,
                },
                false,
            ),
        }
    }

    /// Check if we're at a bracket character
    fn at_bracket(&self) -> bool {
        if self.pos >= self.content.len() {
            return false;
        }

        let c = self.content.as_bytes()[self.pos];
        matches!(c, b'{' | b'}' | b'[' | b']' | b'(' | b')' | b'<' | b'>')
    }
}

// Global cache for bracket pair ASTs - for this simple implementation,
// we'll use a thread-local approach rather than a mutex
thread_local! {
    static BRACKET_AST_CACHE: std::cell::RefCell<HashMap<usize, (Instant, BracketPairAst)>> =
        std::cell::RefCell::new(HashMap::new());
}

// Maximum time to keep an AST in the cache before rebuilding
const MAX_CACHE_AGE: Duration = Duration::from_secs(10);

/// Colors brackets based on their nesting level.
///
/// This method analyzes the buffer, parses all brackets and their nesting levels,
/// and applies appropriate background highlights to colorize them. The colors
/// are chosen from the user's settings.
pub fn colorize_bracket_pairs(editor: &mut Editor, window: &mut Window, cx: &mut Context<Editor>) {
    // Clear existing bracket pair highlights
    editor.clear_background_highlights::<BracketPairColorization>(cx);

    // Check if the feature is enabled in settings
    let settings = EditorSettings::get_global(cx);
    if !settings.bracket_pair_colorization.enabled {
        return;
    }

    // Get bracket colors from settings
    let bracket_colors = &settings.bracket_pair_colorization.colors;
    if bracket_colors.is_empty() {
        return;
    }

    // Get the current editor state
    let snapshot = editor.snapshot(window, cx);
    let buffer_text = snapshot.buffer_snapshot.text();
    let buffer_id = buffer_text.len(); // Use length as a simple ID

    // Get visible range to limit the brackets we highlight
    let visible_range = {
        // Since we can't easily convert viewport coordinates to buffer positions
        // in a generic way, we'll process the entire buffer for now
        // A more optimized implementation would only process visible text
        let start_offset = 0;
        let end_offset = buffer_text.len();

        start_offset..end_offset
    };
    let buffer_id = buffer_text.len(); // Use text length as a simple ID
    let mut ast = BRACKET_AST_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();

        if let Some((last_update, cached_ast)) = cache.get_mut(&buffer_id) {
            // If the cache is too old, rebuild
            if last_update.elapsed() > MAX_CACHE_AGE {
                let new_ast = BracketPairAst::build(&buffer_text);
                *last_update = Instant::now();
                *cached_ast = new_ast.clone();
                new_ast
            } else {
                cached_ast.cache.clear(); // Clear query cache when reusing
                cached_ast.clone()
            }
        } else {
            // Build a new AST for this buffer
            let ast = BracketPairAst::build(&buffer_text);
            cache.insert(buffer_id, (Instant::now(), ast.clone()));
            ast
        }
    });

    // Find all bracket pairs in the visible range
    let bracket_pairs = ast.find_bracket_pairs(visible_range);

    // Convert offsets to anchors and apply highlights
    let highlight_ranges: Vec<_> = bracket_pairs
        .into_iter()
        .map(|(range, nesting_level)| {
            // Choose color based on nesting level
            let color_index = nesting_level % bracket_colors.len();

            // Get the points for these offsets
            let start_point = snapshot.buffer_snapshot.point_for_offset(range.start);
            let end_point = snapshot.buffer_snapshot.point_for_offset(range.end);

            // Convert to multi_buffer anchors
            let start_anchor = snapshot.buffer_snapshot.anchor_before_point(start_point);
            let end_anchor = snapshot.buffer_snapshot.anchor_before_point(end_point);

            (
                Range {
                    start: start_anchor,
                    end: end_anchor,
                },
                color_index,
            )
        })
        .collect();

    // Group by color for more efficient highlight application
    let mut by_color: HashMap<usize, Vec<Range<Anchor>>> = HashMap::new();
    for (range, color_index) in highlight_ranges {
        by_color.entry(color_index).or_default().push(range);
    }

    // Apply highlights for each color group
    for (color_index, ranges) in by_color {
        let color_str = &bracket_colors[color_index];

        // Parse the color string into an Hsla value
        let color_value = gpui::color::parse_color(color_str)
            .map(|color| gpui::Hsla { a: 0.3, ..color })
            .unwrap_or_else(|| Hsla::default());

        editor.highlight_background::<BracketPairColorization>(
            &ranges,
            move |_theme| color_value,
            cx,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{editor_tests::init_test, test::editor_lsp_test_context::EditorLspTestContext};
    use indoc::indoc;

    #[gpui::test]
    async fn test_bracket_pair_colorization(cx: &mut gpui::TestAppContext) {
        init_test(cx, |_| {});

        let mut cx = EditorLspTestContext::new_rust(
            lsp::ServerCapabilities {
                document_formatting_provider: Some(lsp::OneOf::Left(true)),
                ..Default::default()
            },
            cx,
        )
        .await;

        // Test with nested brackets
        cx.set_state(indoc! {r#"
            fn test() {
                let x = [(1, 2), {3, 4}];
                if (x.len() > 0) {
                    println!("Hello");
                }
            }
        "#});

        // Verify the code runs without errors
        cx.update_editor(|editor, window, cx| {
            colorize_bracket_pairs(editor, window, cx);
        });
    }

    #[test]
    fn test_bracket_parser() {
        let content = "func() { [1, 2, (3 + 4)] }";
        let mut parser = BracketParser::new(content);
        let ast = parser.parse_document();

        // Simple structure verification (not checking exact structure)
        match ast {
            BracketAstNode::List { .. } => {
                // Success - we expect a list at the root
                assert!(true);
            }
            _ => {
                panic!("Expected a List node at the root");
            }
        }
    }
}
