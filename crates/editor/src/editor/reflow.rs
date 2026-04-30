use super::*;

pub(crate) fn char_len_with_expanded_tabs(
    offset: usize,
    text: &str,
    tab_size: NonZeroU32,
) -> usize {
    let tab_size = tab_size.get() as usize;
    let mut width = offset;

    for ch in text.chars() {
        width += if ch == '\t' {
            tab_size - (width % tab_size)
        } else {
            1
        };
    }

    width - offset
}

/// Tokenizes a string into runs of text that should stick together, or that is whitespace.
pub(crate) struct WordBreakingTokenizer<'a> {
    input: &'a str,
}

impl<'a> WordBreakingTokenizer<'a> {
    pub(crate) fn new(input: &'a str) -> Self {
        Self { input }
    }
}

fn is_char_ideographic(ch: char) -> bool {
    use unicode_script::Script::*;
    use unicode_script::UnicodeScript;
    matches!(ch.script(), Han | Tangut | Yi)
}

fn is_grapheme_ideographic(text: &str) -> bool {
    text.chars().any(is_char_ideographic)
}

fn is_grapheme_whitespace(text: &str) -> bool {
    text.chars().any(|x| x.is_whitespace())
}

fn should_stay_with_preceding_ideograph(text: &str) -> bool {
    text.chars()
        .next()
        .is_some_and(|ch| matches!(ch, '。' | '、' | '，' | '？' | '！' | '：' | '；' | '…'))
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub(crate) enum WordBreakToken<'a> {
    Word { token: &'a str, grapheme_len: usize },
    InlineWhitespace { token: &'a str, grapheme_len: usize },
    Newline,
}

impl<'a> Iterator for WordBreakingTokenizer<'a> {
    /// Yields a span, the count of graphemes in the token, and whether it was
    /// whitespace. Note that it also breaks at word boundaries.
    type Item = WordBreakToken<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        use unicode_segmentation::UnicodeSegmentation;
        if self.input.is_empty() {
            return None;
        }

        let mut iter = self.input.graphemes(true).peekable();
        let mut offset = 0;
        let mut grapheme_len = 0;
        if let Some(first_grapheme) = iter.next() {
            let is_newline = first_grapheme == "\n";
            let is_whitespace = is_grapheme_whitespace(first_grapheme);
            offset += first_grapheme.len();
            grapheme_len += 1;
            if is_grapheme_ideographic(first_grapheme) && !is_whitespace {
                if let Some(grapheme) = iter.peek().copied()
                    && should_stay_with_preceding_ideograph(grapheme)
                {
                    offset += grapheme.len();
                    grapheme_len += 1;
                }
            } else {
                let mut words = self.input[offset..].split_word_bound_indices().peekable();
                let mut next_word_bound = words.peek().copied();
                if next_word_bound.is_some_and(|(i, _)| i == 0) {
                    next_word_bound = words.next();
                }
                while let Some(grapheme) = iter.peek().copied() {
                    if next_word_bound.is_some_and(|(i, _)| i == offset) {
                        break;
                    };
                    if is_grapheme_whitespace(grapheme) != is_whitespace
                        || (grapheme == "\n") != is_newline
                    {
                        break;
                    };
                    offset += grapheme.len();
                    grapheme_len += 1;
                    iter.next();
                }
            }
            let token = &self.input[..offset];
            self.input = &self.input[offset..];
            if token == "\n" {
                Some(WordBreakToken::Newline)
            } else if is_whitespace {
                Some(WordBreakToken::InlineWhitespace {
                    token,
                    grapheme_len,
                })
            } else {
                Some(WordBreakToken::Word {
                    token,
                    grapheme_len,
                })
            }
        } else {
            None
        }
    }
}

pub(crate) fn wrap_with_prefix(
    first_line_prefix: String,
    subsequent_lines_prefix: String,
    unwrapped_text: String,
    wrap_column: usize,
    tab_size: NonZeroU32,
    preserve_existing_whitespace: bool,
) -> String {
    let first_line_prefix_len = char_len_with_expanded_tabs(0, &first_line_prefix, tab_size);
    let subsequent_lines_prefix_len =
        char_len_with_expanded_tabs(0, &subsequent_lines_prefix, tab_size);
    let mut wrapped_text = String::new();
    let mut current_line = first_line_prefix;
    let mut is_first_line = true;

    let tokenizer = WordBreakingTokenizer::new(&unwrapped_text);
    let mut current_line_len = first_line_prefix_len;
    let mut in_whitespace = false;
    for token in tokenizer {
        let have_preceding_whitespace = in_whitespace;
        match token {
            WordBreakToken::Word {
                token,
                grapheme_len,
            } => {
                in_whitespace = false;
                let current_prefix_len = if is_first_line {
                    first_line_prefix_len
                } else {
                    subsequent_lines_prefix_len
                };
                if current_line_len + grapheme_len > wrap_column
                    && current_line_len != current_prefix_len
                {
                    wrapped_text.push_str(current_line.trim_end());
                    wrapped_text.push('\n');
                    is_first_line = false;
                    current_line = subsequent_lines_prefix.clone();
                    current_line_len = subsequent_lines_prefix_len;
                }
                current_line.push_str(token);
                current_line_len += grapheme_len;
            }
            WordBreakToken::InlineWhitespace {
                mut token,
                mut grapheme_len,
            } => {
                in_whitespace = true;
                if have_preceding_whitespace && !preserve_existing_whitespace {
                    continue;
                }
                if !preserve_existing_whitespace {
                    // Keep a single whitespace grapheme as-is
                    if let Some(first) =
                        unicode_segmentation::UnicodeSegmentation::graphemes(token, true).next()
                    {
                        token = first;
                    } else {
                        token = " ";
                    }
                    grapheme_len = 1;
                }
                let current_prefix_len = if is_first_line {
                    first_line_prefix_len
                } else {
                    subsequent_lines_prefix_len
                };
                if current_line_len + grapheme_len > wrap_column {
                    wrapped_text.push_str(current_line.trim_end());
                    wrapped_text.push('\n');
                    is_first_line = false;
                    current_line = subsequent_lines_prefix.clone();
                    current_line_len = subsequent_lines_prefix_len;
                } else if current_line_len != current_prefix_len || preserve_existing_whitespace {
                    current_line.push_str(token);
                    current_line_len += grapheme_len;
                }
            }
            WordBreakToken::Newline => {
                in_whitespace = true;
                let current_prefix_len = if is_first_line {
                    first_line_prefix_len
                } else {
                    subsequent_lines_prefix_len
                };
                if preserve_existing_whitespace {
                    wrapped_text.push_str(current_line.trim_end());
                    wrapped_text.push('\n');
                    is_first_line = false;
                    current_line = subsequent_lines_prefix.clone();
                    current_line_len = subsequent_lines_prefix_len;
                } else if have_preceding_whitespace {
                    continue;
                } else if current_line_len + 1 > wrap_column
                    && current_line_len != current_prefix_len
                {
                    wrapped_text.push_str(current_line.trim_end());
                    wrapped_text.push('\n');
                    is_first_line = false;
                    current_line = subsequent_lines_prefix.clone();
                    current_line_len = subsequent_lines_prefix_len;
                } else if current_line_len != current_prefix_len {
                    current_line.push(' ');
                    current_line_len += 1;
                }
            }
        }
    }

    if !current_line.is_empty() {
        wrapped_text.push_str(&current_line);
    }
    wrapped_text
}
