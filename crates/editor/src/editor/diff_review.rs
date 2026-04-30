use super::*;

impl Editor {
    pub fn show_diff_review_button(&self) -> bool {
        self.show_diff_review_button
    }

    pub fn render_diff_review_button(
        &self,
        display_row: DisplayRow,
        width: Pixels,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let text_color = cx.theme().colors().text;
        let icon_color = cx.theme().colors().icon_accent;

        h_flex()
            .id("diff_review_button")
            .cursor_pointer()
            .w(width - px(1.))
            .h(relative(0.9))
            .justify_center()
            .rounded_sm()
            .border_1()
            .border_color(text_color.opacity(0.1))
            .bg(text_color.opacity(0.15))
            .hover(|s| {
                s.bg(icon_color.opacity(0.4))
                    .border_color(icon_color.opacity(0.5))
            })
            .child(Icon::new(IconName::Plus).size(IconSize::Small))
            .tooltip(Tooltip::text("Add Review (drag to select multiple lines)"))
            .on_mouse_down(
                gpui::MouseButton::Left,
                cx.listener(move |editor, _event: &gpui::MouseDownEvent, window, cx| {
                    editor.start_diff_review_drag(display_row, window, cx);
                }),
            )
    }

    pub fn start_diff_review_drag(
        &mut self,
        display_row: DisplayRow,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let snapshot = self.snapshot(window, cx);
        let point = snapshot
            .display_snapshot
            .display_point_to_point(DisplayPoint::new(display_row, 0), Bias::Left);
        let anchor = snapshot.buffer_snapshot().anchor_before(point);
        self.diff_review_drag_state = Some(DiffReviewDragState {
            start_anchor: anchor,
            current_anchor: anchor,
        });
        cx.notify();
    }

    pub fn update_diff_review_drag(
        &mut self,
        display_row: DisplayRow,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.diff_review_drag_state.is_none() {
            return;
        }
        let snapshot = self.snapshot(window, cx);
        let point = snapshot
            .display_snapshot
            .display_point_to_point(display_row.as_display_point(), Bias::Left);
        let anchor = snapshot.buffer_snapshot().anchor_before(point);
        if let Some(drag_state) = &mut self.diff_review_drag_state {
            drag_state.current_anchor = anchor;
            cx.notify();
        }
    }

    pub fn end_diff_review_drag(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(drag_state) = self.diff_review_drag_state.take() {
            let snapshot = self.snapshot(window, cx);
            let range = drag_state.row_range(&snapshot.display_snapshot);
            self.show_diff_review_overlay(*range.start()..*range.end(), window, cx);
        }
        cx.notify();
    }

    pub fn cancel_diff_review_drag(&mut self, cx: &mut Context<Self>) {
        self.diff_review_drag_state = None;
        cx.notify();
    }

    /// Calculates the appropriate block height for the diff review overlay.
    /// Height is in lines: 2 for input row, 1 for header when comments exist,
    /// and 2 lines per comment when expanded.
    pub(super) fn calculate_overlay_height(
        &self,
        hunk_key: &DiffHunkKey,
        comments_expanded: bool,
        snapshot: &MultiBufferSnapshot,
    ) -> u32 {
        let comment_count = self.hunk_comment_count(hunk_key, snapshot);
        let base_height: u32 = 2; // Input row with avatar and buttons

        if comment_count == 0 {
            base_height
        } else if comments_expanded {
            // Header (1 line) + 2 lines per comment
            base_height + 1 + (comment_count as u32 * 2)
        } else {
            // Just header when collapsed
            base_height + 1
        }
    }

    pub fn show_diff_review_overlay(
        &mut self,
        display_range: Range<DisplayRow>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Range { start, end } = display_range.sorted();

        let buffer_snapshot = self.buffer.read(cx).snapshot(cx);
        let editor_snapshot = self.snapshot(window, cx);

        // Convert display rows to multibuffer points
        let start_point = editor_snapshot
            .display_snapshot
            .display_point_to_point(start.as_display_point(), Bias::Left);
        let end_point = editor_snapshot
            .display_snapshot
            .display_point_to_point(end.as_display_point(), Bias::Left);
        let end_multi_buffer_row = MultiBufferRow(end_point.row);

        // Create anchor range for the selected lines (start of first line to end of last line)
        let line_end = Point::new(
            end_point.row,
            buffer_snapshot.line_len(end_multi_buffer_row),
        );
        let anchor_range =
            buffer_snapshot.anchor_after(start_point)..buffer_snapshot.anchor_before(line_end);

        // Compute the hunk key for this display row
        let file_path = buffer_snapshot
            .file_at(start_point)
            .map(|file: &Arc<dyn language::File>| file.path().clone())
            .unwrap_or_else(|| Arc::from(util::rel_path::RelPath::empty()));
        let hunk_start_anchor = buffer_snapshot.anchor_before(start_point);
        let new_hunk_key = DiffHunkKey {
            file_path,
            hunk_start_anchor,
        };

        // Check if we already have an overlay for this hunk
        if let Some(existing_overlay) = self.diff_review_overlays.iter().find(|overlay| {
            Self::hunk_keys_match(&overlay.hunk_key, &new_hunk_key, &buffer_snapshot)
        }) {
            // Just focus the existing overlay's prompt editor
            let focus_handle = existing_overlay.prompt_editor.focus_handle(cx);
            window.focus(&focus_handle, cx);
            return;
        }

        // Dismiss overlays that have no comments for their hunks
        self.dismiss_overlays_without_comments(cx);

        // Get the current user's avatar URI from the project's user_store
        let user_avatar_uri = self.project.as_ref().and_then(|project| {
            let user_store = project.read(cx).user_store();
            user_store
                .read(cx)
                .current_user()
                .map(|user| user.avatar_uri.clone())
        });

        // Create anchor at the end of the last row so the block appears immediately below it
        // Use multibuffer coordinates for anchor creation
        let line_len = buffer_snapshot.line_len(end_multi_buffer_row);
        let anchor = buffer_snapshot.anchor_after(Point::new(end_multi_buffer_row.0, line_len));

        // Use the hunk key we already computed
        let hunk_key = new_hunk_key;

        // Create the prompt editor for the review input
        let prompt_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_placeholder_text("Add a review comment...", window, cx);
            editor
        });

        // Register the Newline action on the prompt editor to submit the review
        let parent_editor = cx.entity().downgrade();
        let subscription = prompt_editor.update(cx, |prompt_editor, _cx| {
            prompt_editor.register_action({
                let parent_editor = parent_editor.clone();
                move |_: &crate::actions::Newline, window, cx| {
                    if let Some(editor) = parent_editor.upgrade() {
                        editor.update(cx, |editor, cx| {
                            editor.submit_diff_review_comment(window, cx);
                        });
                    }
                }
            })
        });

        // Calculate initial height based on existing comments for this hunk
        let initial_height = self.calculate_overlay_height(&hunk_key, true, &buffer_snapshot);

        // Create the overlay block
        let prompt_editor_for_render = prompt_editor.clone();
        let hunk_key_for_render = hunk_key.clone();
        let editor_handle = cx.entity().downgrade();
        let block = BlockProperties {
            style: BlockStyle::Sticky,
            placement: BlockPlacement::Below(anchor),
            height: Some(initial_height),
            render: Arc::new(move |cx| {
                Self::render_diff_review_overlay(
                    &prompt_editor_for_render,
                    &hunk_key_for_render,
                    &editor_handle,
                    cx,
                )
            }),
            priority: 0,
        };

        let block_ids = self.insert_blocks([block], None, cx);
        let Some(block_id) = block_ids.into_iter().next() else {
            log::error!("Failed to insert diff review overlay block");
            return;
        };

        self.diff_review_overlays.push(DiffReviewOverlay {
            anchor_range,
            block_id,
            prompt_editor: prompt_editor.clone(),
            hunk_key,
            comments_expanded: true,
            inline_edit_editors: HashMap::default(),
            inline_edit_subscriptions: HashMap::default(),
            user_avatar_uri,
            _subscription: subscription,
        });

        // Focus the prompt editor
        let focus_handle = prompt_editor.focus_handle(cx);
        window.focus(&focus_handle, cx);

        cx.notify();
    }

    /// Dismisses all diff review overlays.
    pub fn dismiss_all_diff_review_overlays(&mut self, cx: &mut Context<Self>) {
        if self.diff_review_overlays.is_empty() {
            return;
        }
        let block_ids: HashSet<_> = self
            .diff_review_overlays
            .drain(..)
            .map(|overlay| overlay.block_id)
            .collect();
        self.remove_blocks(block_ids, None, cx);
        cx.notify();
    }

    /// Dismisses overlays that have no comments stored for their hunks.
    /// Keeps overlays that have at least one comment.
    pub(super) fn dismiss_overlays_without_comments(&mut self, cx: &mut Context<Self>) {
        let snapshot = self.buffer.read(cx).snapshot(cx);

        // First, compute which overlays have comments (to avoid borrow issues with retain)
        let overlays_with_comments: Vec<bool> = self
            .diff_review_overlays
            .iter()
            .map(|overlay| self.hunk_comment_count(&overlay.hunk_key, &snapshot) > 0)
            .collect();

        // Now collect block IDs to remove and retain overlays
        let mut block_ids_to_remove = HashSet::default();
        let mut index = 0;
        self.diff_review_overlays.retain(|overlay| {
            let has_comments = overlays_with_comments[index];
            index += 1;
            if !has_comments {
                block_ids_to_remove.insert(overlay.block_id);
            }
            has_comments
        });

        if !block_ids_to_remove.is_empty() {
            self.remove_blocks(block_ids_to_remove, None, cx);
            cx.notify();
        }
    }

    /// Refreshes the diff review overlay block to update its height and render function.
    /// Uses resize_blocks and replace_blocks to avoid visual flicker from remove+insert.
    pub(super) fn refresh_diff_review_overlay_height(
        &mut self,
        hunk_key: &DiffHunkKey,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Extract all needed data from overlay first to avoid borrow conflicts
        let snapshot = self.buffer.read(cx).snapshot(cx);
        let (comments_expanded, block_id, prompt_editor) = {
            let Some(overlay) = self
                .diff_review_overlays
                .iter()
                .find(|overlay| Self::hunk_keys_match(&overlay.hunk_key, hunk_key, &snapshot))
            else {
                return;
            };

            (
                overlay.comments_expanded,
                overlay.block_id,
                overlay.prompt_editor.clone(),
            )
        };

        // Calculate new height
        let snapshot = self.buffer.read(cx).snapshot(cx);
        let new_height = self.calculate_overlay_height(hunk_key, comments_expanded, &snapshot);

        // Update the block height using resize_blocks (avoids flicker)
        let mut heights = HashMap::default();
        heights.insert(block_id, new_height);
        self.resize_blocks(heights, None, cx);

        // Update the render function using replace_blocks (avoids flicker)
        let hunk_key_for_render = hunk_key.clone();
        let editor_handle = cx.entity().downgrade();
        let render: Arc<dyn Fn(&mut BlockContext) -> AnyElement + Send + Sync> =
            Arc::new(move |cx| {
                Self::render_diff_review_overlay(
                    &prompt_editor,
                    &hunk_key_for_render,
                    &editor_handle,
                    cx,
                )
            });

        let mut renderers = HashMap::default();
        renderers.insert(block_id, render);
        self.replace_blocks(renderers, None, cx);
    }

    /// Action handler for SubmitDiffReviewComment.
    pub fn submit_diff_review_comment_action(
        &mut self,
        _: &SubmitDiffReviewComment,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.submit_diff_review_comment(window, cx);
    }

    /// Stores the diff review comment locally.
    /// Comments are stored per-hunk and can later be batch-submitted to the Agent panel.
    pub fn submit_diff_review_comment(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Find the overlay that currently has focus
        let overlay_index = self
            .diff_review_overlays
            .iter()
            .position(|overlay| overlay.prompt_editor.focus_handle(cx).is_focused(window));
        let Some(overlay_index) = overlay_index else {
            return;
        };
        let overlay = &self.diff_review_overlays[overlay_index];

        let comment_text = overlay.prompt_editor.read(cx).text(cx).trim().to_string();
        if comment_text.is_empty() {
            return;
        }

        let anchor_range = overlay.anchor_range.clone();
        let hunk_key = overlay.hunk_key.clone();

        self.add_review_comment(hunk_key.clone(), comment_text, anchor_range, cx);

        // Clear the prompt editor but keep the overlay open
        if let Some(overlay) = self.diff_review_overlays.get(overlay_index) {
            overlay.prompt_editor.update(cx, |editor, cx| {
                editor.clear(window, cx);
            });
        }

        // Refresh the overlay to update the block height for the new comment
        self.refresh_diff_review_overlay_height(&hunk_key, window, cx);

        cx.notify();
    }

    /// Returns the prompt editor for the diff review overlay, if one is active.
    /// This is primarily used for testing.
    pub fn diff_review_prompt_editor(&self) -> Option<&Entity<Editor>> {
        self.diff_review_overlays
            .first()
            .map(|overlay| &overlay.prompt_editor)
    }

    /// Returns the line range for the first diff review overlay, if one is active.
    /// Returns (start_row, end_row) as physical line numbers in the underlying file.
    pub fn diff_review_line_range(&self, cx: &App) -> Option<(u32, u32)> {
        let overlay = self.diff_review_overlays.first()?;
        let snapshot = self.buffer.read(cx).snapshot(cx);
        let start_point = overlay.anchor_range.start.to_point(&snapshot);
        let end_point = overlay.anchor_range.end.to_point(&snapshot);
        let start_row = snapshot
            .point_to_buffer_point(start_point)
            .map(|(_, p)| p.row)
            .unwrap_or(start_point.row);
        let end_row = snapshot
            .point_to_buffer_point(end_point)
            .map(|(_, p)| p.row)
            .unwrap_or(end_point.row);
        Some((start_row, end_row))
    }

    /// Sets whether the comments section is expanded in the diff review overlay.
    /// This is primarily used for testing.
    pub fn set_diff_review_comments_expanded(&mut self, expanded: bool, cx: &mut Context<Self>) {
        for overlay in &mut self.diff_review_overlays {
            overlay.comments_expanded = expanded;
        }
        cx.notify();
    }

    /// Compares two DiffHunkKeys for equality by resolving their anchors.
    pub(super) fn hunk_keys_match(
        a: &DiffHunkKey,
        b: &DiffHunkKey,
        snapshot: &MultiBufferSnapshot,
    ) -> bool {
        a.file_path == b.file_path
            && a.hunk_start_anchor.to_point(snapshot) == b.hunk_start_anchor.to_point(snapshot)
    }

    /// Returns comments for a specific hunk, ordered by creation time.
    pub fn comments_for_hunk<'a>(
        &'a self,
        key: &DiffHunkKey,
        snapshot: &MultiBufferSnapshot,
    ) -> &'a [StoredReviewComment] {
        let key_point = key.hunk_start_anchor.to_point(snapshot);
        self.stored_review_comments
            .iter()
            .find(|(k, _)| {
                k.file_path == key.file_path && k.hunk_start_anchor.to_point(snapshot) == key_point
            })
            .map(|(_, comments)| comments.as_slice())
            .unwrap_or(&[])
    }

    /// Returns the total count of stored review comments across all hunks.
    pub fn total_review_comment_count(&self) -> usize {
        self.stored_review_comments
            .iter()
            .map(|(_, v)| v.len())
            .sum()
    }

    /// Returns the count of comments for a specific hunk.
    pub fn hunk_comment_count(&self, key: &DiffHunkKey, snapshot: &MultiBufferSnapshot) -> usize {
        let key_point = key.hunk_start_anchor.to_point(snapshot);
        self.stored_review_comments
            .iter()
            .find(|(k, _)| {
                k.file_path == key.file_path && k.hunk_start_anchor.to_point(snapshot) == key_point
            })
            .map(|(_, v)| v.len())
            .unwrap_or(0)
    }

    /// Adds a new review comment to a specific hunk.
    pub fn add_review_comment(
        &mut self,
        hunk_key: DiffHunkKey,
        comment: String,
        anchor_range: Range<Anchor>,
        cx: &mut Context<Self>,
    ) -> usize {
        let id = self.next_review_comment_id;
        self.next_review_comment_id += 1;

        let stored_comment = StoredReviewComment::new(id, comment, anchor_range);

        let snapshot = self.buffer.read(cx).snapshot(cx);
        let key_point = hunk_key.hunk_start_anchor.to_point(&snapshot);

        // Find existing entry for this hunk or add a new one
        if let Some((_, comments)) = self.stored_review_comments.iter_mut().find(|(k, _)| {
            k.file_path == hunk_key.file_path
                && k.hunk_start_anchor.to_point(&snapshot) == key_point
        }) {
            comments.push(stored_comment);
        } else {
            self.stored_review_comments
                .push((hunk_key, vec![stored_comment]));
        }

        cx.emit(EditorEvent::ReviewCommentsChanged {
            total_count: self.total_review_comment_count(),
        });
        cx.notify();
        id
    }

    /// Removes a review comment by ID from any hunk.
    pub fn remove_review_comment(&mut self, id: usize, cx: &mut Context<Self>) -> bool {
        for (_, comments) in self.stored_review_comments.iter_mut() {
            if let Some(index) = comments.iter().position(|c| c.id == id) {
                comments.remove(index);
                cx.emit(EditorEvent::ReviewCommentsChanged {
                    total_count: self.total_review_comment_count(),
                });
                cx.notify();
                return true;
            }
        }
        false
    }

    /// Updates a review comment's text by ID.
    pub fn update_review_comment(
        &mut self,
        id: usize,
        new_comment: String,
        cx: &mut Context<Self>,
    ) -> bool {
        for (_, comments) in self.stored_review_comments.iter_mut() {
            if let Some(comment) = comments.iter_mut().find(|c| c.id == id) {
                comment.comment = new_comment;
                comment.is_editing = false;
                cx.emit(EditorEvent::ReviewCommentsChanged {
                    total_count: self.total_review_comment_count(),
                });
                cx.notify();
                return true;
            }
        }
        false
    }

    /// Sets a comment's editing state.
    pub fn set_comment_editing(&mut self, id: usize, is_editing: bool, cx: &mut Context<Self>) {
        for (_, comments) in self.stored_review_comments.iter_mut() {
            if let Some(comment) = comments.iter_mut().find(|c| c.id == id) {
                comment.is_editing = is_editing;
                cx.notify();
                return;
            }
        }
    }

    /// Takes all stored comments from all hunks, clearing the storage.
    /// Returns a Vec of (hunk_key, comments) pairs.
    pub fn take_all_review_comments(
        &mut self,
        cx: &mut Context<Self>,
    ) -> Vec<(DiffHunkKey, Vec<StoredReviewComment>)> {
        // Dismiss all overlays when taking comments (e.g., when sending to agent)
        self.dismiss_all_diff_review_overlays(cx);
        let comments = std::mem::take(&mut self.stored_review_comments);
        // Reset the ID counter since all comments have been taken
        self.next_review_comment_id = 0;
        cx.emit(EditorEvent::ReviewCommentsChanged { total_count: 0 });
        cx.notify();
        comments
    }

    /// Removes review comments whose anchors are no longer valid or whose
    /// associated diff hunks no longer exist.
    ///
    /// This should be called when the buffer changes to prevent orphaned comments
    /// from accumulating.
    pub fn cleanup_orphaned_review_comments(&mut self, cx: &mut Context<Self>) {
        let snapshot = self.buffer.read(cx).snapshot(cx);
        let original_count = self.total_review_comment_count();

        // Remove comments with invalid hunk anchors
        self.stored_review_comments
            .retain(|(hunk_key, _)| hunk_key.hunk_start_anchor.is_valid(&snapshot));

        // Also clean up individual comments with invalid anchor ranges
        for (_, comments) in &mut self.stored_review_comments {
            comments.retain(|comment| {
                comment.range.start.is_valid(&snapshot) && comment.range.end.is_valid(&snapshot)
            });
        }

        // Remove empty hunk entries
        self.stored_review_comments
            .retain(|(_, comments)| !comments.is_empty());

        let new_count = self.total_review_comment_count();
        if new_count != original_count {
            cx.emit(EditorEvent::ReviewCommentsChanged {
                total_count: new_count,
            });
            cx.notify();
        }
    }

    /// Toggles the expanded state of the comments section in the overlay.
    pub fn toggle_review_comments_expanded(
        &mut self,
        _: &ToggleReviewCommentsExpanded,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Find the overlay that currently has focus, or use the first one
        let overlay_info = self.diff_review_overlays.iter_mut().find_map(|overlay| {
            if overlay.prompt_editor.focus_handle(cx).is_focused(window) {
                overlay.comments_expanded = !overlay.comments_expanded;
                Some(overlay.hunk_key.clone())
            } else {
                None
            }
        });

        // If no focused overlay found, toggle the first one
        let hunk_key = overlay_info.or_else(|| {
            self.diff_review_overlays.first_mut().map(|overlay| {
                overlay.comments_expanded = !overlay.comments_expanded;
                overlay.hunk_key.clone()
            })
        });

        if let Some(hunk_key) = hunk_key {
            self.refresh_diff_review_overlay_height(&hunk_key, window, cx);
            cx.notify();
        }
    }

    /// Handles the EditReviewComment action - sets a comment into editing mode.
    pub fn edit_review_comment(
        &mut self,
        action: &EditReviewComment,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let comment_id = action.id;

        // Set the comment to editing mode
        self.set_comment_editing(comment_id, true, cx);

        // Find the overlay that contains this comment and create an inline editor if needed
        // First, find which hunk this comment belongs to
        let hunk_key = self
            .stored_review_comments
            .iter()
            .find_map(|(key, comments)| {
                if comments.iter().any(|c| c.id == comment_id) {
                    Some(key.clone())
                } else {
                    None
                }
            });

        let snapshot = self.buffer.read(cx).snapshot(cx);
        if let Some(hunk_key) = hunk_key {
            if let Some(overlay) = self
                .diff_review_overlays
                .iter_mut()
                .find(|overlay| Self::hunk_keys_match(&overlay.hunk_key, &hunk_key, &snapshot))
            {
                if let std::collections::hash_map::Entry::Vacant(entry) =
                    overlay.inline_edit_editors.entry(comment_id)
                {
                    // Find the comment text
                    let comment_text = self
                        .stored_review_comments
                        .iter()
                        .flat_map(|(_, comments)| comments)
                        .find(|c| c.id == comment_id)
                        .map(|c| c.comment.clone())
                        .unwrap_or_default();

                    // Create inline editor
                    let parent_editor = cx.entity().downgrade();
                    let inline_editor = cx.new(|cx| {
                        let mut editor = Editor::single_line(window, cx);
                        editor.set_text(&*comment_text, window, cx);
                        // Select all text for easy replacement
                        editor.select_all(&crate::actions::SelectAll, window, cx);
                        editor
                    });

                    // Register the Newline action to confirm the edit
                    let subscription = inline_editor.update(cx, |inline_editor, _cx| {
                        inline_editor.register_action({
                            let parent_editor = parent_editor.clone();
                            move |_: &crate::actions::Newline, window, cx| {
                                if let Some(editor) = parent_editor.upgrade() {
                                    editor.update(cx, |editor, cx| {
                                        editor.confirm_edit_review_comment(comment_id, window, cx);
                                    });
                                }
                            }
                        })
                    });

                    // Store the subscription to keep the action handler alive
                    overlay
                        .inline_edit_subscriptions
                        .insert(comment_id, subscription);

                    // Focus the inline editor
                    let focus_handle = inline_editor.focus_handle(cx);
                    window.focus(&focus_handle, cx);

                    entry.insert(inline_editor);
                }
            }
        }

        cx.notify();
    }

    /// Confirms an inline edit of a review comment.
    pub fn confirm_edit_review_comment(
        &mut self,
        comment_id: usize,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Get the new text from the inline editor
        // Find the overlay containing this comment's inline editor
        let snapshot = self.buffer.read(cx).snapshot(cx);
        let hunk_key = self
            .stored_review_comments
            .iter()
            .find_map(|(key, comments)| {
                if comments.iter().any(|c| c.id == comment_id) {
                    Some(key.clone())
                } else {
                    None
                }
            });

        let new_text = hunk_key
            .as_ref()
            .and_then(|hunk_key| {
                self.diff_review_overlays
                    .iter()
                    .find(|overlay| Self::hunk_keys_match(&overlay.hunk_key, hunk_key, &snapshot))
            })
            .as_ref()
            .and_then(|overlay| overlay.inline_edit_editors.get(&comment_id))
            .map(|editor| editor.read(cx).text(cx).trim().to_string());

        if let Some(new_text) = new_text {
            if !new_text.is_empty() {
                self.update_review_comment(comment_id, new_text, cx);
            }
        }

        // Remove the inline editor and its subscription
        if let Some(hunk_key) = hunk_key {
            if let Some(overlay) = self
                .diff_review_overlays
                .iter_mut()
                .find(|overlay| Self::hunk_keys_match(&overlay.hunk_key, &hunk_key, &snapshot))
            {
                overlay.inline_edit_editors.remove(&comment_id);
                overlay.inline_edit_subscriptions.remove(&comment_id);
            }
        }

        // Clear editing state
        self.set_comment_editing(comment_id, false, cx);
    }

    /// Cancels an inline edit of a review comment.
    pub fn cancel_edit_review_comment(
        &mut self,
        comment_id: usize,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Find which hunk this comment belongs to
        let hunk_key = self
            .stored_review_comments
            .iter()
            .find_map(|(key, comments)| {
                if comments.iter().any(|c| c.id == comment_id) {
                    Some(key.clone())
                } else {
                    None
                }
            });

        // Remove the inline editor and its subscription
        if let Some(hunk_key) = hunk_key {
            let snapshot = self.buffer.read(cx).snapshot(cx);
            if let Some(overlay) = self
                .diff_review_overlays
                .iter_mut()
                .find(|overlay| Self::hunk_keys_match(&overlay.hunk_key, &hunk_key, &snapshot))
            {
                overlay.inline_edit_editors.remove(&comment_id);
                overlay.inline_edit_subscriptions.remove(&comment_id);
            }
        }

        // Clear editing state
        self.set_comment_editing(comment_id, false, cx);
    }

    /// Action handler for ConfirmEditReviewComment.
    pub fn confirm_edit_review_comment_action(
        &mut self,
        action: &ConfirmEditReviewComment,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.confirm_edit_review_comment(action.id, window, cx);
    }

    /// Action handler for CancelEditReviewComment.
    pub fn cancel_edit_review_comment_action(
        &mut self,
        action: &CancelEditReviewComment,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.cancel_edit_review_comment(action.id, window, cx);
    }

    /// Handles the DeleteReviewComment action - removes a comment.
    pub fn delete_review_comment(
        &mut self,
        action: &DeleteReviewComment,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Get the hunk key before removing the comment
        // Find the hunk key from the comment itself
        let comment_id = action.id;
        let hunk_key = self
            .stored_review_comments
            .iter()
            .find_map(|(key, comments)| {
                if comments.iter().any(|c| c.id == comment_id) {
                    Some(key.clone())
                } else {
                    None
                }
            });

        // Also get it from the overlay for refresh purposes
        let overlay_hunk_key = self
            .diff_review_overlays
            .first()
            .map(|o| o.hunk_key.clone());

        self.remove_review_comment(action.id, cx);

        // Refresh the overlay height after removing a comment
        if let Some(hunk_key) = hunk_key.or(overlay_hunk_key) {
            self.refresh_diff_review_overlay_height(&hunk_key, window, cx);
        }
    }

    pub(super) fn render_diff_review_overlay(
        prompt_editor: &Entity<Editor>,
        hunk_key: &DiffHunkKey,
        editor_handle: &WeakEntity<Editor>,
        cx: &mut BlockContext,
    ) -> AnyElement {
        fn format_line_ranges(ranges: &[(u32, u32)]) -> Option<String> {
            if ranges.is_empty() {
                return None;
            }
            let formatted: Vec<String> = ranges
                .iter()
                .map(|(start, end)| {
                    let start_line = start + 1;
                    let end_line = end + 1;
                    if start_line == end_line {
                        format!("Line {start_line}")
                    } else {
                        format!("Lines {start_line}-{end_line}")
                    }
                })
                .collect();
            // Don't show label for single line in single excerpt
            if ranges.len() == 1 && ranges[0].0 == ranges[0].1 {
                return None;
            }
            Some(formatted.join(" ⋯ "))
        }

        let theme = cx.theme();
        let colors = theme.colors();

        let (comments, comments_expanded, inline_editors, user_avatar_uri, line_ranges) =
            editor_handle
                .upgrade()
                .map(|editor| {
                    let editor = editor.read(cx);
                    let snapshot = editor.buffer().read(cx).snapshot(cx);
                    let comments = editor.comments_for_hunk(hunk_key, &snapshot).to_vec();
                    let (expanded, editors, avatar_uri, line_ranges) = editor
                        .diff_review_overlays
                        .iter()
                        .find(|overlay| {
                            Editor::hunk_keys_match(&overlay.hunk_key, hunk_key, &snapshot)
                        })
                        .map(|o| {
                            let start_point = o.anchor_range.start.to_point(&snapshot);
                            let end_point = o.anchor_range.end.to_point(&snapshot);
                            // Get line ranges per excerpt to detect discontinuities
                            let buffer_ranges =
                                snapshot.range_to_buffer_ranges(start_point..end_point);
                            let ranges: Vec<(u32, u32)> = buffer_ranges
                                .iter()
                                .map(|(buffer_snapshot, range, _)| {
                                    let start = buffer_snapshot.offset_to_point(range.start.0).row;
                                    let end = buffer_snapshot.offset_to_point(range.end.0).row;
                                    (start, end)
                                })
                                .collect();
                            (
                                o.comments_expanded,
                                o.inline_edit_editors.clone(),
                                o.user_avatar_uri.clone(),
                                if ranges.is_empty() {
                                    None
                                } else {
                                    Some(ranges)
                                },
                            )
                        })
                        .unwrap_or((true, HashMap::default(), None, None));
                    (comments, expanded, editors, avatar_uri, line_ranges)
                })
                .unwrap_or((Vec::new(), true, HashMap::default(), None, None));

        let comment_count = comments.len();
        let avatar_size = px(20.);
        let action_icon_size = IconSize::XSmall;

        v_flex()
            .w_full()
            .bg(colors.editor_background)
            .border_b_1()
            .border_color(colors.border)
            .px_2()
            .pb_2()
            .gap_2()
            // Line range indicator (only shown for multi-line selections or multiple excerpts)
            .when_some(line_ranges, |el, ranges| {
                let label = format_line_ranges(&ranges);
                if let Some(label) = label {
                    el.child(
                        h_flex()
                            .w_full()
                            .px_2()
                            .child(Label::new(label).size(LabelSize::Small).color(Color::Muted)),
                    )
                } else {
                    el
                }
            })
            // Top row: editable input with user's avatar
            .child(
                h_flex()
                    .w_full()
                    .items_center()
                    .gap_2()
                    .px_2()
                    .py_1p5()
                    .rounded_md()
                    .bg(colors.surface_background)
                    .child(
                        div()
                            .size(avatar_size)
                            .flex_shrink_0()
                            .rounded_full()
                            .overflow_hidden()
                            .child(if let Some(ref avatar_uri) = user_avatar_uri {
                                Avatar::new(avatar_uri.clone())
                                    .size(avatar_size)
                                    .into_any_element()
                            } else {
                                Icon::new(IconName::Person)
                                    .size(IconSize::Small)
                                    .color(ui::Color::Muted)
                                    .into_any_element()
                            }),
                    )
                    .child(
                        div()
                            .flex_1()
                            .border_1()
                            .border_color(colors.border)
                            .rounded_md()
                            .bg(colors.editor_background)
                            .px_2()
                            .py_1()
                            .child(prompt_editor.clone()),
                    )
                    .child(
                        h_flex()
                            .flex_shrink_0()
                            .gap_1()
                            .child(
                                IconButton::new("diff-review-close", IconName::Close)
                                    .icon_color(ui::Color::Muted)
                                    .icon_size(action_icon_size)
                                    .tooltip(Tooltip::text("Close"))
                                    .on_click(|_, window, cx| {
                                        window
                                            .dispatch_action(Box::new(crate::actions::Cancel), cx);
                                    }),
                            )
                            .child(
                                IconButton::new("diff-review-add", IconName::Return)
                                    .icon_color(ui::Color::Muted)
                                    .icon_size(action_icon_size)
                                    .tooltip(Tooltip::text("Add comment"))
                                    .on_click(|_, window, cx| {
                                        window.dispatch_action(
                                            Box::new(crate::actions::SubmitDiffReviewComment),
                                            cx,
                                        );
                                    }),
                            ),
                    ),
            )
            // Expandable comments section (only shown when there are comments)
            .when(comment_count > 0, |el| {
                el.child(Self::render_comments_section(
                    comments,
                    comments_expanded,
                    inline_editors,
                    user_avatar_uri,
                    avatar_size,
                    action_icon_size,
                    colors,
                ))
            })
            .into_any_element()
    }

    pub(super) fn render_comments_section(
        comments: Vec<StoredReviewComment>,
        expanded: bool,
        inline_editors: HashMap<usize, Entity<Editor>>,
        user_avatar_uri: Option<SharedUri>,
        avatar_size: Pixels,
        action_icon_size: IconSize,
        colors: &theme::ThemeColors,
    ) -> impl IntoElement {
        let comment_count = comments.len();

        v_flex()
            .w_full()
            .gap_1()
            // Header with expand/collapse toggle
            .child(
                h_flex()
                    .id("review-comments-header")
                    .w_full()
                    .items_center()
                    .gap_1()
                    .px_2()
                    .py_1()
                    .cursor_pointer()
                    .rounded_md()
                    .hover(|style| style.bg(colors.ghost_element_hover))
                    .on_click(|_, window: &mut Window, cx| {
                        window.dispatch_action(
                            Box::new(crate::actions::ToggleReviewCommentsExpanded),
                            cx,
                        );
                    })
                    .child(
                        Icon::new(if expanded {
                            IconName::ChevronDown
                        } else {
                            IconName::ChevronRight
                        })
                        .size(IconSize::Small)
                        .color(ui::Color::Muted),
                    )
                    .child(
                        Label::new(format!(
                            "{} Comment{}",
                            comment_count,
                            if comment_count == 1 { "" } else { "s" }
                        ))
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                    ),
            )
            // Comments list (when expanded)
            .when(expanded, |el| {
                el.children(comments.into_iter().map(|comment| {
                    let inline_editor = inline_editors.get(&comment.id).cloned();
                    Self::render_comment_row(
                        comment,
                        inline_editor,
                        user_avatar_uri.clone(),
                        avatar_size,
                        action_icon_size,
                        colors,
                    )
                }))
            })
    }

    pub(super) fn render_comment_row(
        comment: StoredReviewComment,
        inline_editor: Option<Entity<Editor>>,
        user_avatar_uri: Option<SharedUri>,
        avatar_size: Pixels,
        action_icon_size: IconSize,
        colors: &theme::ThemeColors,
    ) -> impl IntoElement {
        let comment_id = comment.id;
        let is_editing = inline_editor.is_some();

        h_flex()
            .w_full()
            .items_center()
            .gap_2()
            .px_2()
            .py_1p5()
            .rounded_md()
            .bg(colors.surface_background)
            .child(
                div()
                    .size(avatar_size)
                    .flex_shrink_0()
                    .rounded_full()
                    .overflow_hidden()
                    .child(if let Some(ref avatar_uri) = user_avatar_uri {
                        Avatar::new(avatar_uri.clone())
                            .size(avatar_size)
                            .into_any_element()
                    } else {
                        Icon::new(IconName::Person)
                            .size(IconSize::Small)
                            .color(ui::Color::Muted)
                            .into_any_element()
                    }),
            )
            .child(if let Some(editor) = inline_editor {
                // Inline edit mode: show an editable text field
                div()
                    .flex_1()
                    .border_1()
                    .border_color(colors.border)
                    .rounded_md()
                    .bg(colors.editor_background)
                    .px_2()
                    .py_1()
                    .child(editor)
                    .into_any_element()
            } else {
                // Display mode: show the comment text
                div()
                    .flex_1()
                    .text_sm()
                    .text_color(colors.text)
                    .child(comment.comment)
                    .into_any_element()
            })
            .child(if is_editing {
                // Editing mode: show close and confirm buttons
                h_flex()
                    .gap_1()
                    .child(
                        IconButton::new(
                            format!("diff-review-cancel-edit-{comment_id}"),
                            IconName::Close,
                        )
                        .icon_color(ui::Color::Muted)
                        .icon_size(action_icon_size)
                        .tooltip(Tooltip::text("Cancel"))
                        .on_click(move |_, window, cx| {
                            window.dispatch_action(
                                Box::new(crate::actions::CancelEditReviewComment {
                                    id: comment_id,
                                }),
                                cx,
                            );
                        }),
                    )
                    .child(
                        IconButton::new(
                            format!("diff-review-confirm-edit-{comment_id}"),
                            IconName::Return,
                        )
                        .icon_color(ui::Color::Muted)
                        .icon_size(action_icon_size)
                        .tooltip(Tooltip::text("Confirm"))
                        .on_click(move |_, window, cx| {
                            window.dispatch_action(
                                Box::new(crate::actions::ConfirmEditReviewComment {
                                    id: comment_id,
                                }),
                                cx,
                            );
                        }),
                    )
                    .into_any_element()
            } else {
                // Display mode: no action buttons for now (edit/delete not yet implemented)
                gpui::Empty.into_any_element()
            })
    }
}
