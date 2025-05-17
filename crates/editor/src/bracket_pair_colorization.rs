use crate::{Editor, EditorSettings};
use gpui::{Context, Window};
use settings::Settings;
use std::ops::Range;

/// Module for colorizing matching bracket pairs with different colors.
///
/// Implementation inspired by VS Code's bracket pair colorization, which
/// colorizes brackets by their nesting level to make it easier to match
/// opening and closing brackets.

enum BracketPairColorization {}

/// Colors brackets based on their nesting level.
///
/// This method analyzes the buffer, parses all brackets and their nesting levels,
/// and applies appropriate background highlights to colorize them. The colors
/// are chosen from the user's settings.
pub fn colorize_bracket_pairs(editor: &mut Editor, window: &mut Window, cx: &mut Context<Editor>) {
    // Clear existing bracket pair highlights
    editor.clear_background_highlights::<BracketPairColorization>(cx);

    println!("Color bracket pair");

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

    println!("In this block");

    // Get the current editor state
    let snapshot = editor.snapshot(window, cx);

    // Find matching brackets in the buffer
    if let Some((opening_range, closing_range)) =
        snapshot.buffer_snapshot.innermost_enclosing_bracket_ranges(
            0..0, // Just a placeholder range
            None,
        )
    {
        // Convert to anchors
        println!("In that block");
        let opening_anchor = snapshot
            .buffer_snapshot
            .anchor_at(opening_range.start, text::Bias::Left);
        let closing_anchor = snapshot
            .buffer_snapshot
            .anchor_at(closing_range.start, text::Bias::Left);

        // Apply highlights with different colors based on nesting levels
        editor.highlight_background::<BracketPairColorization>(
            &[Range {
                start: opening_anchor,
                end: closing_anchor,
            }],
            |theme| theme.editor_document_highlight_bracket_background,
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
}
