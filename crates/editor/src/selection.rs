use super::*;

#[derive(Clone, Debug, PartialEq)]
pub enum SelectPhase {
    Begin {
        position: DisplayPoint,
        add: bool,
        click_count: usize,
    },
    BeginColumnar {
        position: DisplayPoint,
        reset: bool,
        mode: ColumnarMode,
        goal_column: u32,
    },
    Extend {
        position: DisplayPoint,
        click_count: usize,
    },
    Update {
        position: DisplayPoint,
        goal_column: u32,
        scroll_delta: gpui::Point<f32>,
    },
    End,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ColumnarMode {
    FromMouse,
    FromSelection,
}

#[derive(Clone, Debug)]
pub enum SelectMode {
    Character,
    Word(Range<Anchor>),
    Line(Range<Anchor>),
    All,
}

pub(crate) enum SelectionDragState {
    /// State when no drag related activity is detected.
    None,
    /// State when the mouse is down on a selection that is about to be dragged.
    ReadyToDrag {
        selection: Selection<Anchor>,
        click_position: gpui::Point<Pixels>,
        mouse_down_time: Instant,
    },
    /// State when the mouse is dragging the selection in the editor.
    Dragging {
        selection: Selection<Anchor>,
        drop_cursor: Selection<Anchor>,
        hide_drop_cursor: bool,
    },
}

pub(crate) enum ColumnarSelectionState {
    FromMouse {
        selection_tail: Anchor,
        display_point: Option<DisplayPoint>,
    },
    FromSelection {
        selection_tail: Anchor,
    },
}

#[derive(Debug)]
pub struct RemoteSelection {
    pub replica_id: ReplicaId,
    pub selection: Selection<Anchor>,
    pub cursor_shape: CursorShape,
    pub collaborator_id: CollaboratorId,
    pub line_mode: bool,
    pub user_name: Option<SharedString>,
    pub color: PlayerColor,
}

#[derive(Clone, Debug)]
pub(crate) struct SelectionHistoryEntry {
    selections: Arc<[Selection<Anchor>]>,
    select_next_state: Option<SelectNextState>,
    select_prev_state: Option<SelectNextState>,
    add_selections_state: Option<AddSelectionsState>,
}

#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub(crate) enum SelectionHistoryMode {
    #[default]
    Normal,
    Undoing,
    Redoing,
    Skipping,
}

#[derive(Debug)]
/// SelectionEffects controls the side-effects of updating the selection.
///
/// The default behaviour does "what you mostly want":
/// - it pushes to the nav history if the cursor moved by >10 lines
/// - it re-triggers completion requests
/// - it scrolls to fit
///
/// You might want to modify these behaviours. For example when doing a "jump"
/// like go to definition, we always want to add to nav history; but when scrolling
/// in vim mode we never do.
///
/// Similarly, you might want to disable scrolling if you don't want the viewport to
/// move.
#[derive(Clone)]
pub struct SelectionEffects {
    nav_history: Option<bool>,
    completions: bool,
    scroll: Option<Autoscroll>,
    from_search: bool,
}

impl Default for SelectionEffects {
    fn default() -> Self {
        Self {
            nav_history: None,
            completions: true,
            scroll: Some(Autoscroll::fit()),
            from_search: false,
        }
    }
}
impl SelectionEffects {
    pub fn scroll(scroll: Autoscroll) -> Self {
        Self {
            scroll: Some(scroll),
            ..Default::default()
        }
    }

    pub fn no_scroll() -> Self {
        Self {
            scroll: None,
            ..Default::default()
        }
    }

    pub fn completions(self, completions: bool) -> Self {
        Self {
            completions,
            ..self
        }
    }

    pub fn nav_history(self, nav_history: bool) -> Self {
        Self {
            nav_history: Some(nav_history),
            ..self
        }
    }

    pub fn from_search(self, from_search: bool) -> Self {
        Self {
            from_search,
            ..self
        }
    }
}

pub(crate) struct DeferredSelectionEffectsState {
    changed: bool,
    effects: SelectionEffects,
    old_cursor_position: Anchor,
    history_entry: SelectionHistoryEntry,
}

#[derive(Default)]
pub(crate) struct SelectionHistory {
    #[allow(clippy::type_complexity)]
    selections_by_transaction:
        HashMap<TransactionId, (Arc<[Selection<Anchor>]>, Option<Arc<[Selection<Anchor>]>>)>,
    pub(crate) mode: SelectionHistoryMode,
    undo_stack: VecDeque<SelectionHistoryEntry>,
    redo_stack: VecDeque<SelectionHistoryEntry>,
}

impl SelectionHistory {
    #[track_caller]
    pub(crate) fn insert_transaction(
        &mut self,
        transaction_id: TransactionId,
        selections: Arc<[Selection<Anchor>]>,
    ) {
        if selections.is_empty() {
            log::error!(
                "SelectionHistory::insert_transaction called with empty selections. Caller: {}",
                std::panic::Location::caller()
            );
            return;
        }
        self.selections_by_transaction
            .insert(transaction_id, (selections, None));
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn transaction(
        &self,
        transaction_id: TransactionId,
    ) -> Option<&(Arc<[Selection<Anchor>]>, Option<Arc<[Selection<Anchor>]>>)> {
        self.selections_by_transaction.get(&transaction_id)
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn transaction_mut(
        &mut self,
        transaction_id: TransactionId,
    ) -> Option<&mut (Arc<[Selection<Anchor>]>, Option<Arc<[Selection<Anchor>]>>)> {
        self.selections_by_transaction.get_mut(&transaction_id)
    }

    fn push(&mut self, entry: SelectionHistoryEntry) {
        if !entry.selections.is_empty() {
            match self.mode {
                SelectionHistoryMode::Normal => {
                    self.push_undo(entry);
                    self.redo_stack.clear();
                }
                SelectionHistoryMode::Undoing => self.push_redo(entry),
                SelectionHistoryMode::Redoing => self.push_undo(entry),
                SelectionHistoryMode::Skipping => {}
            }
        }
    }

    fn push_undo(&mut self, entry: SelectionHistoryEntry) {
        if self
            .undo_stack
            .back()
            .is_none_or(|e| e.selections != entry.selections)
        {
            self.undo_stack.push_back(entry);
            if self.undo_stack.len() > MAX_SELECTION_HISTORY_LEN {
                self.undo_stack.pop_front();
            }
        }
    }

    fn push_redo(&mut self, entry: SelectionHistoryEntry) {
        if self
            .redo_stack
            .back()
            .is_none_or(|e| e.selections != entry.selections)
        {
            self.redo_stack.push_back(entry);
            if self.redo_stack.len() > MAX_SELECTION_HISTORY_LEN {
                self.redo_stack.pop_front();
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AddSelectionsState {
    groups: Vec<AddSelectionsGroup>,
}

#[derive(Clone, Debug)]
pub(crate) struct AddSelectionsGroup {
    above: bool,
    stack: Vec<usize>,
}

#[derive(Clone)]
pub(crate) struct SelectNextState {
    query: AhoCorasick,
    wordwise: bool,
    done: bool,
}

impl std::fmt::Debug for SelectNextState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(std::any::type_name::<Self>())
            .field("wordwise", &self.wordwise)
            .field("done", &self.done)
            .finish()
    }
}

#[derive(Debug)]
pub(crate) struct AutocloseRegion {
    pub(crate) selection_id: usize,
    pub(crate) range: Range<Anchor>,
    pub(crate) pair: BracketPair,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ClipboardSelection {
    /// The number of bytes in this selection.
    pub len: usize,
    /// Whether this was a full-line selection.
    pub is_entire_line: bool,
    /// The indentation of the first line when this content was originally copied.
    pub first_line_indent: u32,
    #[serde(default)]
    pub file_path: Option<PathBuf>,
    #[serde(default)]
    pub line_range: Option<RangeInclusive<u32>>,
}

impl ClipboardSelection {
    pub fn for_buffer(
        len: usize,
        is_entire_line: bool,
        range: Range<Point>,
        buffer: &MultiBufferSnapshot,
        project: Option<&Entity<Project>>,
        cx: &App,
    ) -> Self {
        let first_line_indent = buffer
            .indent_size_for_line(MultiBufferRow(range.start.row))
            .len;

        let file_path = util::maybe!({
            let project = project?.read(cx);
            let file = buffer.file_at(range.start)?;
            let project_path = ProjectPath {
                worktree_id: file.worktree_id(cx),
                path: file.path().clone(),
            };
            project.absolute_path(&project_path, cx)
        });

        let line_range = if file_path.is_some() {
            buffer
                .range_to_buffer_range(range)
                .map(|(_, buffer_range)| buffer_range.start.row..=buffer_range.end.row)
        } else {
            None
        };

        Self {
            len,
            is_entire_line,
            first_line_indent,
            file_path,
            line_range,
        }
    }
}

// selections, scroll behavior, was newest selection reversed
type SelectSyntaxNodeHistoryState = (
    Box<[Selection<Anchor>]>,
    SelectSyntaxNodeScrollBehavior,
    bool,
);

#[derive(Default)]
pub(crate) struct SelectSyntaxNodeHistory {
    stack: Vec<SelectSyntaxNodeHistoryState>,
    // disable temporarily to allow changing selections without losing the stack
    pub disable_clearing: bool,
}

impl SelectSyntaxNodeHistory {
    pub fn try_clear(&mut self) {
        if !self.disable_clearing {
            self.stack.clear();
        }
    }

    pub fn push(&mut self, selection: SelectSyntaxNodeHistoryState) {
        self.stack.push(selection);
    }

    pub fn pop(&mut self) -> Option<SelectSyntaxNodeHistoryState> {
        self.stack.pop()
    }
}

pub(crate) enum SelectSyntaxNodeScrollBehavior {
    CursorTop,
    FitSelection,
    CursorBottom,
}

pub enum MultibufferSelectionMode {
    First,
    All,
}

impl Editor {
    pub(super) fn selections_did_change(
        &mut self,
        local: bool,
        old_cursor_position: &Anchor,
        effects: SelectionEffects,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.last_selection_from_search = effects.from_search;
        window.invalidate_character_coordinates();

        // Copy selections to primary selection buffer
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        if local {
            let selections = self
                .selections
                .all::<MultiBufferOffset>(&self.display_snapshot(cx));
            let buffer_handle = self.buffer.read(cx).read(cx);

            let mut text = String::new();
            for (index, selection) in selections.iter().enumerate() {
                let text_for_selection = buffer_handle
                    .text_for_range(selection.start..selection.end)
                    .collect::<String>();

                text.push_str(&text_for_selection);
                if index != selections.len() - 1 {
                    text.push('\n');
                }
            }

            if !text.is_empty() {
                cx.write_to_primary(ClipboardItem::new_string(text));
            }
        }

        let selection_anchors = self.selections.disjoint_anchors_arc();

        if self.focus_handle.is_focused(window) && self.leader_id.is_none() {
            self.buffer.update(cx, |buffer, cx| {
                buffer.set_active_selections(
                    &selection_anchors,
                    self.selections.line_mode(),
                    self.cursor_shape,
                    cx,
                )
            });
        }
        let display_map = self
            .display_map
            .update(cx, |display_map, cx| display_map.snapshot(cx));
        let buffer = display_map.buffer_snapshot();
        if self.selections.count() == 1 {
            self.add_selections_state = None;
        }
        self.select_next_state = None;
        self.select_prev_state = None;
        self.select_syntax_node_history.try_clear();
        self.invalidate_autoclose_regions(&selection_anchors, buffer);
        self.snippet_stack.invalidate(&selection_anchors, buffer);
        self.take_rename(false, window, cx);

        let newest_selection = self.selections.newest_anchor();
        let new_cursor_position = newest_selection.head();
        let selection_start = newest_selection.start;

        if effects.nav_history.is_none() || effects.nav_history == Some(true) {
            self.push_to_nav_history(
                *old_cursor_position,
                Some(new_cursor_position.to_point(buffer)),
                false,
                effects.nav_history == Some(true),
                cx,
            );
        }

        if local {
            if let Some((anchor, _)) = buffer.anchor_to_buffer_anchor(new_cursor_position) {
                self.register_buffer(anchor.buffer_id, cx);
            }

            let mut context_menu = self.context_menu.borrow_mut();
            let completion_menu = match context_menu.as_ref() {
                Some(CodeContextMenu::Completions(menu)) => Some(menu),
                Some(CodeContextMenu::CodeActions(_)) => {
                    *context_menu = None;
                    None
                }
                None => None,
            };
            let completion_position = completion_menu.map(|menu| menu.initial_position);
            drop(context_menu);

            if effects.completions
                && let Some(completion_position) = completion_position
            {
                let start_offset = selection_start.to_offset(buffer);
                let position_matches = start_offset == completion_position.to_offset(buffer);
                let continue_showing = if let Some((snap, ..)) =
                    buffer.point_to_buffer_offset(completion_position)
                    && !snap.capability.editable()
                {
                    false
                } else if position_matches {
                    if self.snippet_stack.is_empty() {
                        buffer.char_kind_before(start_offset, Some(CharScopeContext::Completion))
                            == Some(CharKind::Word)
                    } else {
                        // Snippet choices can be shown even when the cursor is in whitespace.
                        // Dismissing the menu with actions like backspace is handled by
                        // invalidation regions.
                        true
                    }
                } else {
                    false
                };

                if continue_showing {
                    self.open_or_update_completions_menu(None, None, false, window, cx);
                } else {
                    self.hide_context_menu(window, cx);
                }
            }

            hide_hover(self, cx);

            self.refresh_code_actions_for_selection(window, cx);
            self.refresh_document_highlights(cx);
            refresh_linked_ranges(self, window, cx);

            self.refresh_selected_text_highlights(&display_map, false, window, cx);
            self.refresh_matching_bracket_highlights(&display_map, cx);
            self.refresh_outline_symbols_at_cursor(cx);
            self.update_visible_edit_prediction(window, cx);
            self.hide_blame_popover(true, cx);
            if self.git_blame_inline_enabled {
                self.start_inline_blame_timer(window, cx);
            }
        }

        self.blink_manager.update(cx, BlinkManager::pause_blinking);

        if local && !self.suppress_selection_callback {
            if let Some(callback) = self.on_local_selections_changed.as_ref() {
                let cursor_position = self.selections.newest::<Point>(&display_map).head();
                callback(cursor_position, window, cx);
            }
        }

        cx.emit(EditorEvent::SelectionsChanged { local });

        let selections = &self.selections.disjoint_anchors_arc();
        if local && let Some(buffer_snapshot) = buffer.as_singleton() {
            let inmemory_selections = selections
                .iter()
                .map(|s| {
                    let start = s.range().start.text_anchor_in(buffer_snapshot);
                    let end = s.range().end.text_anchor_in(buffer_snapshot);
                    (start..end).to_point(buffer_snapshot)
                })
                .collect();
            self.update_restoration_data(cx, |data| {
                data.selections = inmemory_selections;
            });

            if WorkspaceSettings::get(None, cx).restore_on_startup
                != RestoreOnStartupBehavior::EmptyTab
                && let Some(workspace_id) = self.workspace_serialization_id(cx)
            {
                let snapshot = self.buffer().read(cx).snapshot(cx);
                let selections = selections.clone();
                let background_executor = cx.background_executor().clone();
                let editor_id = cx.entity().entity_id().as_u64() as ItemId;
                let db = EditorDb::global(cx);
                self.serialize_selections = cx.background_spawn(async move {
                    background_executor.timer(SERIALIZATION_THROTTLE_TIME).await;
                    let db_selections = selections
                        .iter()
                        .map(|selection| {
                            (
                                selection.start.to_offset(&snapshot).0,
                                selection.end.to_offset(&snapshot).0,
                            )
                        })
                        .collect();

                    db.save_editor_selections(editor_id, workspace_id, db_selections)
                        .await
                        .with_context(|| {
                            format!(
                                "persisting editor selections for editor {editor_id}, \
                                workspace {workspace_id:?}"
                            )
                        })
                        .log_err();
                });
            }
        }

        cx.notify();
    }

    pub(super) fn folds_did_change(&mut self, cx: &mut Context<Self>) {
        use text::ToOffset as _;

        if self.mode.is_minimap()
            || WorkspaceSettings::get(None, cx).restore_on_startup
                == RestoreOnStartupBehavior::EmptyTab
        {
            return;
        }

        let display_snapshot = self
            .display_map
            .update(cx, |display_map, cx| display_map.snapshot(cx));
        let Some(buffer_snapshot) = display_snapshot.buffer_snapshot().as_singleton() else {
            return;
        };
        let inmemory_folds = display_snapshot
            .folds_in_range(MultiBufferOffset(0)..display_snapshot.buffer_snapshot().len())
            .map(|fold| {
                let start = fold.range.start.text_anchor_in(buffer_snapshot);
                let end = fold.range.end.text_anchor_in(buffer_snapshot);
                (start..end).to_point(buffer_snapshot)
            })
            .collect();
        self.update_restoration_data(cx, |data| {
            data.folds = inmemory_folds;
        });

        let Some(workspace_id) = self.workspace_serialization_id(cx) else {
            return;
        };

        // Get file path for path-based fold storage (survives tab close)
        let Some(file_path) = self.buffer().read(cx).as_singleton().and_then(|buffer| {
            project::File::from_dyn(buffer.read(cx).file())
                .map(|file| Arc::<Path>::from(file.abs_path(cx)))
        }) else {
            return;
        };

        let background_executor = cx.background_executor().clone();
        const FINGERPRINT_LEN: usize = 32;
        let db_folds = display_snapshot
            .folds_in_range(MultiBufferOffset(0)..display_snapshot.buffer_snapshot().len())
            .map(|fold| {
                let start = fold
                    .range
                    .start
                    .text_anchor_in(buffer_snapshot)
                    .to_offset(buffer_snapshot);
                let end = fold
                    .range
                    .end
                    .text_anchor_in(buffer_snapshot)
                    .to_offset(buffer_snapshot);

                // Extract fingerprints - content at fold boundaries for validation on restore
                // Both fingerprints must be INSIDE the fold to avoid capturing surrounding
                // content that might change independently.
                // start_fp: first min(32, fold_len) bytes of fold content
                // end_fp: last min(32, fold_len) bytes of fold content
                // Clip to character boundaries to handle multibyte UTF-8 characters.
                let fold_len = end - start;
                let start_fp_end = buffer_snapshot
                    .clip_offset(start + std::cmp::min(FINGERPRINT_LEN, fold_len), Bias::Left);
                let start_fp: String = buffer_snapshot
                    .text_for_range(start..start_fp_end)
                    .collect();
                let end_fp_start = buffer_snapshot
                    .clip_offset(end.saturating_sub(FINGERPRINT_LEN).max(start), Bias::Right);
                let end_fp: String = buffer_snapshot.text_for_range(end_fp_start..end).collect();

                (start, end, start_fp, end_fp)
            })
            .collect::<Vec<_>>();
        let db = EditorDb::global(cx);
        self.serialize_folds = cx.background_spawn(async move {
            background_executor.timer(SERIALIZATION_THROTTLE_TIME).await;
            if db_folds.is_empty() {
                // No folds - delete any persisted folds for this file
                db.delete_file_folds(workspace_id, file_path)
                    .await
                    .with_context(|| format!("deleting file folds for workspace {workspace_id:?}"))
                    .log_err();
            } else {
                db.save_file_folds(workspace_id, file_path, db_folds)
                    .await
                    .with_context(|| {
                        format!("persisting file folds for workspace {workspace_id:?}")
                    })
                    .log_err();
            }
        });
    }

    pub fn sync_selections(
        &mut self,
        other: Entity<Editor>,
        cx: &mut Context<Self>,
    ) -> gpui::Subscription {
        let other_selections = other.read(cx).selections.disjoint_anchors().to_vec();
        if !other_selections.is_empty() {
            self.selections
                .change_with(&self.display_snapshot(cx), |selections| {
                    selections.select_anchors(other_selections);
                });
        }

        let other_subscription = cx.subscribe(&other, |this, other, other_evt, cx| {
            if let EditorEvent::SelectionsChanged { local: true } = other_evt {
                let other_selections = other.read(cx).selections.disjoint_anchors().to_vec();
                if other_selections.is_empty() {
                    return;
                }
                let snapshot = this.display_snapshot(cx);
                this.selections.change_with(&snapshot, |selections| {
                    selections.select_anchors(other_selections);
                });
            }
        });

        let this_subscription = cx.subscribe_self::<EditorEvent>(move |this, this_evt, cx| {
            if let EditorEvent::SelectionsChanged { local: true } = this_evt {
                let these_selections = this.selections.disjoint_anchors().to_vec();
                if these_selections.is_empty() {
                    return;
                }
                other.update(cx, |other_editor, cx| {
                    let snapshot = other_editor.display_snapshot(cx);
                    other_editor
                        .selections
                        .change_with(&snapshot, |selections| {
                            selections.select_anchors(these_selections);
                        })
                });
            }
        });

        Subscription::join(other_subscription, this_subscription)
    }

    pub(super) fn unfold_buffers_with_selections(&mut self, cx: &mut Context<Self>) {
        if self.buffer().read(cx).is_singleton() {
            return;
        }
        let snapshot = self.buffer.read(cx).snapshot(cx);
        let buffer_ids: HashSet<BufferId> = self
            .selections
            .disjoint_anchor_ranges()
            .flat_map(|range| snapshot.buffer_ids_for_range(range))
            .collect();
        for buffer_id in buffer_ids {
            self.unfold_buffer(buffer_id, cx);
        }
    }

    /// Changes selections using the provided mutation function. Changes to `self.selections` occur
    /// immediately, but when run within `transact` or `with_selection_effects_deferred` other
    /// effects of selection change occur at the end of the transaction.
    pub fn change_selections<R>(
        &mut self,
        effects: SelectionEffects,
        window: &mut Window,
        cx: &mut Context<Self>,
        change: impl FnOnce(&mut MutableSelectionsCollection<'_, '_>) -> R,
    ) -> R {
        let snapshot = self.display_snapshot(cx);
        if let Some(state) = &mut self.deferred_selection_effects_state {
            state.effects.scroll = effects.scroll.or(state.effects.scroll);
            state.effects.completions = effects.completions;
            state.effects.nav_history = effects.nav_history.or(state.effects.nav_history);
            let (changed, result) = self.selections.change_with(&snapshot, change);
            state.changed |= changed;
            return result;
        }
        let mut state = DeferredSelectionEffectsState {
            changed: false,
            effects,
            old_cursor_position: self.selections.newest_anchor().head(),
            history_entry: SelectionHistoryEntry {
                selections: self.selections.disjoint_anchors_arc(),
                select_next_state: self.select_next_state.clone(),
                select_prev_state: self.select_prev_state.clone(),
                add_selections_state: self.add_selections_state.clone(),
            },
        };
        let (changed, result) = self.selections.change_with(&snapshot, change);
        state.changed = state.changed || changed;
        if self.defer_selection_effects {
            self.deferred_selection_effects_state = Some(state);
        } else {
            self.apply_selection_effects(state, window, cx);
        }
        result
    }

    /// Defers the effects of selection change, so that the effects of multiple calls to
    /// `change_selections` are applied at the end. This way these intermediate states aren't added
    /// to selection history and the state of popovers based on selection position aren't
    /// erroneously updated.
    pub fn with_selection_effects_deferred<R>(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
        update: impl FnOnce(&mut Self, &mut Window, &mut Context<Self>) -> R,
    ) -> R {
        let already_deferred = self.defer_selection_effects;
        self.defer_selection_effects = true;
        let result = update(self, window, cx);
        if !already_deferred {
            self.defer_selection_effects = false;
            if let Some(state) = self.deferred_selection_effects_state.take() {
                self.apply_selection_effects(state, window, cx);
            }
        }
        result
    }

    pub(super) fn apply_selection_effects(
        &mut self,
        state: DeferredSelectionEffectsState,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if state.changed {
            self.selection_history.push(state.history_entry);

            if let Some(autoscroll) = state.effects.scroll {
                self.request_autoscroll(autoscroll, cx);
            }

            let old_cursor_position = &state.old_cursor_position;

            self.selections_did_change(true, old_cursor_position, state.effects, window, cx);

            if self.should_open_signature_help_automatically(old_cursor_position, cx) {
                self.show_signature_help_auto(window, cx);
            }
        }
    }

    pub fn edit<I, S, T>(&mut self, edits: I, cx: &mut Context<Self>)
    where
        I: IntoIterator<Item = (Range<S>, T)>,
        S: ToOffset,
        T: Into<Arc<str>>,
    {
        if self.read_only(cx) {
            return;
        }

        self.buffer
            .update(cx, |buffer, cx| buffer.edit(edits, None, cx));
    }

    pub fn edit_with_autoindent<I, S, T>(&mut self, edits: I, cx: &mut Context<Self>)
    where
        I: IntoIterator<Item = (Range<S>, T)>,
        S: ToOffset,
        T: Into<Arc<str>>,
    {
        if self.read_only(cx) {
            return;
        }

        self.buffer.update(cx, |buffer, cx| {
            buffer.edit(edits, self.autoindent_mode.clone(), cx)
        });
    }

    pub fn edit_with_block_indent<I, S, T>(
        &mut self,
        edits: I,
        original_indent_columns: Vec<Option<u32>>,
        cx: &mut Context<Self>,
    ) where
        I: IntoIterator<Item = (Range<S>, T)>,
        S: ToOffset,
        T: Into<Arc<str>>,
    {
        if self.read_only(cx) {
            return;
        }

        self.buffer.update(cx, |buffer, cx| {
            buffer.edit(
                edits,
                Some(AutoindentMode::Block {
                    original_indent_columns,
                }),
                cx,
            )
        });
    }

    pub(super) fn select(
        &mut self,
        phase: SelectPhase,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.hide_context_menu(window, cx);

        match phase {
            SelectPhase::Begin {
                position,
                add,
                click_count,
            } => self.begin_selection(position, add, click_count, window, cx),
            SelectPhase::BeginColumnar {
                position,
                goal_column,
                reset,
                mode,
            } => self.begin_columnar_selection(position, goal_column, reset, mode, window, cx),
            SelectPhase::Extend {
                position,
                click_count,
            } => self.extend_selection(position, click_count, window, cx),
            SelectPhase::Update {
                position,
                goal_column,
                scroll_delta,
            } => self.update_selection(position, goal_column, scroll_delta, window, cx),
            SelectPhase::End => self.end_selection(window, cx),
        }
    }

    pub(super) fn extend_selection(
        &mut self,
        position: DisplayPoint,
        click_count: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        let tail = self
            .selections
            .newest::<MultiBufferOffset>(&display_map)
            .tail();
        let click_count = click_count.max(match self.selections.select_mode() {
            SelectMode::Character => 1,
            SelectMode::Word(_) => 2,
            SelectMode::Line(_) => 3,
            SelectMode::All => 4,
        });
        self.begin_selection(position, false, click_count, window, cx);

        let tail_anchor = display_map.buffer_snapshot().anchor_before(tail);

        let current_selection = match self.selections.select_mode() {
            SelectMode::Character | SelectMode::All => tail_anchor..tail_anchor,
            SelectMode::Word(range) | SelectMode::Line(range) => range.clone(),
        };

        let mut pending_selection = self
            .selections
            .pending_anchor()
            .cloned()
            .expect("extend_selection not called with pending selection");

        if pending_selection
            .start
            .cmp(&current_selection.start, display_map.buffer_snapshot())
            == Ordering::Greater
        {
            pending_selection.start = current_selection.start;
        }
        if pending_selection
            .end
            .cmp(&current_selection.end, display_map.buffer_snapshot())
            == Ordering::Less
        {
            pending_selection.end = current_selection.end;
            pending_selection.reversed = true;
        }

        let mut pending_mode = self.selections.pending_mode().unwrap();
        match &mut pending_mode {
            SelectMode::Word(range) | SelectMode::Line(range) => *range = current_selection,
            _ => {}
        }

        let effects = if EditorSettings::get_global(cx).autoscroll_on_clicks {
            SelectionEffects::scroll(Autoscroll::fit())
        } else {
            SelectionEffects::no_scroll()
        };

        self.change_selections(effects, window, cx, |s| {
            s.set_pending(pending_selection.clone(), pending_mode);
            s.set_is_extending(true);
        });
    }

    pub(super) fn begin_selection(
        &mut self,
        position: DisplayPoint,
        add: bool,
        click_count: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.focus_handle.is_focused(window) {
            self.last_focused_descendant = None;
            window.focus(&self.focus_handle, cx);
        }

        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        let buffer = display_map.buffer_snapshot();
        let position = display_map.clip_point(position, Bias::Left);

        let start;
        let end;
        let mode;
        let mut auto_scroll;
        match click_count {
            1 => {
                start = buffer.anchor_before(position.to_point(&display_map));
                end = start;
                mode = SelectMode::Character;
                auto_scroll = true;
            }
            2 => {
                let position = display_map
                    .clip_point(position, Bias::Left)
                    .to_offset(&display_map, Bias::Left);
                let (range, _) = buffer.surrounding_word(position, None);
                start = buffer.anchor_before(range.start);
                end = buffer.anchor_before(range.end);
                mode = SelectMode::Word(start..end);
                auto_scroll = true;
            }
            3 => {
                let position = display_map
                    .clip_point(position, Bias::Left)
                    .to_point(&display_map);
                let line_start = display_map.prev_line_boundary(position).0;
                let next_line_start = buffer.clip_point(
                    display_map.next_line_boundary(position).0 + Point::new(1, 0),
                    Bias::Left,
                );
                start = buffer.anchor_before(line_start);
                end = buffer.anchor_before(next_line_start);
                mode = SelectMode::Line(start..end);
                auto_scroll = true;
            }
            _ => {
                start = buffer.anchor_before(MultiBufferOffset(0));
                end = buffer.anchor_before(buffer.len());
                mode = SelectMode::All;
                auto_scroll = false;
            }
        }
        auto_scroll &= EditorSettings::get_global(cx).autoscroll_on_clicks;

        let point_to_delete: Option<usize> = {
            let selected_points: Vec<Selection<Point>> =
                self.selections.disjoint_in_range(start..end, &display_map);

            if !add || click_count > 1 {
                None
            } else if !selected_points.is_empty() {
                Some(selected_points[0].id)
            } else {
                let clicked_point_already_selected =
                    self.selections.disjoint_anchors().iter().find(|selection| {
                        selection.start.to_point(buffer) == start.to_point(buffer)
                            || selection.end.to_point(buffer) == end.to_point(buffer)
                    });

                clicked_point_already_selected.map(|selection| selection.id)
            }
        };

        let selections_count = self.selections.count();
        let effects = if auto_scroll {
            SelectionEffects::default()
        } else {
            SelectionEffects::no_scroll()
        };

        self.change_selections(effects, window, cx, |s| {
            if let Some(point_to_delete) = point_to_delete {
                s.delete(point_to_delete);

                if selections_count == 1 {
                    s.set_pending_anchor_range(start..end, mode);
                }
            } else {
                if !add {
                    s.clear_disjoint();
                }

                s.set_pending_anchor_range(start..end, mode);
            }
        });
    }

    pub(super) fn begin_columnar_selection(
        &mut self,
        position: DisplayPoint,
        goal_column: u32,
        reset: bool,
        mode: ColumnarMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.focus_handle.is_focused(window) {
            self.last_focused_descendant = None;
            window.focus(&self.focus_handle, cx);
        }

        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));

        if reset {
            let pointer_position = display_map
                .buffer_snapshot()
                .anchor_before(position.to_point(&display_map));

            self.change_selections(
                SelectionEffects::scroll(Autoscroll::newest()),
                window,
                cx,
                |s| {
                    s.clear_disjoint();
                    s.set_pending_anchor_range(
                        pointer_position..pointer_position,
                        SelectMode::Character,
                    );
                },
            );
        };

        let tail = self.selections.newest::<Point>(&display_map).tail();
        let selection_anchor = display_map.buffer_snapshot().anchor_before(tail);
        self.columnar_selection_state = match mode {
            ColumnarMode::FromMouse => Some(ColumnarSelectionState::FromMouse {
                selection_tail: selection_anchor,
                display_point: if reset {
                    if position.column() != goal_column {
                        Some(DisplayPoint::new(position.row(), goal_column))
                    } else {
                        None
                    }
                } else {
                    None
                },
            }),
            ColumnarMode::FromSelection => Some(ColumnarSelectionState::FromSelection {
                selection_tail: selection_anchor,
            }),
        };

        if !reset {
            self.select_columns(position, goal_column, &display_map, window, cx);
        }
    }

    pub(super) fn update_selection(
        &mut self,
        position: DisplayPoint,
        goal_column: u32,
        scroll_delta: gpui::Point<f32>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));

        if self.columnar_selection_state.is_some() {
            self.select_columns(position, goal_column, &display_map, window, cx);
        } else if let Some(mut pending) = self.selections.pending_anchor().cloned() {
            let buffer = display_map.buffer_snapshot();
            let head;
            let tail;
            let mode = self.selections.pending_mode().unwrap();
            match &mode {
                SelectMode::Character => {
                    head = position.to_point(&display_map);
                    tail = pending.tail().to_point(buffer);
                }
                SelectMode::Word(original_range) => {
                    let offset = display_map
                        .clip_point(position, Bias::Left)
                        .to_offset(&display_map, Bias::Left);
                    let original_range = original_range.to_offset(buffer);

                    let head_offset = if buffer.is_inside_word(offset, None)
                        || original_range.contains(&offset)
                    {
                        let (word_range, _) = buffer.surrounding_word(offset, None);
                        if word_range.start < original_range.start {
                            word_range.start
                        } else {
                            word_range.end
                        }
                    } else {
                        offset
                    };

                    head = head_offset.to_point(buffer);
                    if head_offset <= original_range.start {
                        tail = original_range.end.to_point(buffer);
                    } else {
                        tail = original_range.start.to_point(buffer);
                    }
                }
                SelectMode::Line(original_range) => {
                    let original_range = original_range.to_point(display_map.buffer_snapshot());

                    let position = display_map
                        .clip_point(position, Bias::Left)
                        .to_point(&display_map);
                    let line_start = display_map.prev_line_boundary(position).0;
                    let next_line_start = buffer.clip_point(
                        display_map.next_line_boundary(position).0 + Point::new(1, 0),
                        Bias::Left,
                    );

                    if line_start < original_range.start {
                        head = line_start
                    } else {
                        head = next_line_start
                    }

                    if head <= original_range.start {
                        tail = original_range.end;
                    } else {
                        tail = original_range.start;
                    }
                }
                SelectMode::All => {
                    return;
                }
            };

            if head < tail {
                pending.start = buffer.anchor_before(head);
                pending.end = buffer.anchor_before(tail);
                pending.reversed = true;
            } else {
                pending.start = buffer.anchor_before(tail);
                pending.end = buffer.anchor_before(head);
                pending.reversed = false;
            }

            self.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
                s.set_pending(pending.clone(), mode);
            });
        } else {
            log::error!("update_selection dispatched with no pending selection");
            return;
        }

        self.apply_scroll_delta(scroll_delta, window, cx);
        cx.notify();
    }

    pub(super) fn end_selection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.columnar_selection_state.take();
        if let Some(pending_mode) = self.selections.pending_mode() {
            let selections = self
                .selections
                .all::<MultiBufferOffset>(&self.display_snapshot(cx));
            self.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
                s.select(selections);
                s.clear_pending();
                if s.is_extending() {
                    s.set_is_extending(false);
                } else {
                    s.set_select_mode(pending_mode);
                }
            });
        }
    }

    pub(super) fn select_columns(
        &mut self,
        head: DisplayPoint,
        goal_column: u32,
        display_map: &DisplaySnapshot,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(columnar_state) = self.columnar_selection_state.as_ref() else {
            return;
        };

        let tail = match columnar_state {
            ColumnarSelectionState::FromMouse {
                selection_tail,
                display_point,
            } => display_point.unwrap_or_else(|| selection_tail.to_display_point(display_map)),
            ColumnarSelectionState::FromSelection { selection_tail } => {
                selection_tail.to_display_point(display_map)
            }
        };

        let start_row = cmp::min(tail.row(), head.row());
        let end_row = cmp::max(tail.row(), head.row());
        let start_column = cmp::min(tail.column(), goal_column);
        let end_column = cmp::max(tail.column(), goal_column);
        let reversed = start_column < tail.column();

        let selection_ranges = (start_row.0..=end_row.0)
            .map(DisplayRow)
            .filter_map(|row| {
                if (matches!(columnar_state, ColumnarSelectionState::FromMouse { .. })
                    || start_column <= display_map.line_len(row))
                    && !display_map.is_block_line(row)
                {
                    let start = display_map
                        .clip_point(DisplayPoint::new(row, start_column), Bias::Left)
                        .to_point(display_map);
                    let end = display_map
                        .clip_point(DisplayPoint::new(row, end_column), Bias::Right)
                        .to_point(display_map);
                    if reversed {
                        Some(end..start)
                    } else {
                        Some(start..end)
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if selection_ranges.is_empty() {
            return;
        }

        let ranges = match columnar_state {
            ColumnarSelectionState::FromMouse { .. } => {
                let mut non_empty_ranges = selection_ranges
                    .iter()
                    .filter(|selection_range| selection_range.start != selection_range.end)
                    .peekable();
                if non_empty_ranges.peek().is_some() {
                    non_empty_ranges.cloned().collect()
                } else {
                    selection_ranges
                }
            }
            _ => selection_ranges,
        };

        self.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
            s.select_ranges(ranges);
        });
        cx.notify();
    }

    pub fn has_non_empty_selection(&self, snapshot: &DisplaySnapshot) -> bool {
        self.selections
            .all_adjusted(snapshot)
            .iter()
            .any(|selection| !selection.is_empty())
    }

    pub fn has_pending_nonempty_selection(&self) -> bool {
        let pending_nonempty_selection = match self.selections.pending_anchor() {
            Some(Selection { start, end, .. }) => start != end,
            None => false,
        };

        pending_nonempty_selection
            || (self.columnar_selection_state.is_some()
                && self.selections.disjoint_anchors().len() > 1)
    }

    pub fn has_pending_selection(&self) -> bool {
        self.selections.pending_anchor().is_some() || self.columnar_selection_state.is_some()
    }

    pub fn cancel(&mut self, _: &Cancel, window: &mut Window, cx: &mut Context<Self>) {
        self.selection_mark_mode = false;
        self.selection_drag_state = SelectionDragState::None;

        if self.dismiss_menus_and_popups(true, window, cx) {
            cx.notify();
            return;
        }
        if self.clear_expanded_diff_hunks(cx) {
            cx.notify();
            return;
        }
        if self.show_git_blame_gutter {
            self.show_git_blame_gutter = false;
            cx.notify();
            return;
        }

        if self.mode.is_full()
            && self.change_selections(Default::default(), window, cx, |s| s.try_cancel())
        {
            cx.notify();
            return;
        }

        cx.propagate();
    }

    pub fn dismiss_menus_and_popups(
        &mut self,
        is_user_requested: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let mut dismissed = false;

        dismissed |= self.take_rename(false, window, cx).is_some();
        dismissed |= self.hide_blame_popover(true, cx);
        dismissed |= hide_hover(self, cx);
        dismissed |= self.hide_signature_help(cx, SignatureHelpHiddenBy::Escape);
        dismissed |= self.hide_context_menu(window, cx).is_some();
        dismissed |= self.mouse_context_menu.take().is_some();
        dismissed |= is_user_requested
            && self.discard_edit_prediction(EditPredictionDiscardReason::Rejected, cx);
        dismissed |= self.snippet_stack.pop().is_some();
        if self.diff_review_drag_state.is_some() {
            self.cancel_diff_review_drag(cx);
            dismissed = true;
        }
        if !self.diff_review_overlays.is_empty() {
            self.dismiss_all_diff_review_overlays(cx);
            dismissed = true;
        }

        if self.mode.is_full() && self.has_active_diagnostic_group() {
            self.dismiss_diagnostics(cx);
            dismissed = true;
        }

        dismissed
    }

    pub(super) fn linked_editing_ranges_for(
        &self,
        query_range: Range<text::Anchor>,
        cx: &App,
    ) -> Option<HashMap<Entity<Buffer>, Vec<Range<text::Anchor>>>> {
        use text::ToOffset as TO;

        if self.linked_edit_ranges.is_empty() {
            return None;
        }
        if query_range.start.buffer_id != query_range.end.buffer_id {
            return None;
        };
        let multibuffer_snapshot = self.buffer.read(cx).snapshot(cx);
        let buffer = self.buffer.read(cx).buffer(query_range.end.buffer_id)?;
        let buffer_snapshot = buffer.read(cx).snapshot();
        let (base_range, linked_ranges) = self.linked_edit_ranges.get(
            buffer_snapshot.remote_id(),
            query_range.clone(),
            &buffer_snapshot,
        )?;
        // find offset from the start of current range to current cursor position
        let start_byte_offset = TO::to_offset(&base_range.start, &buffer_snapshot);

        let start_offset = TO::to_offset(&query_range.start, &buffer_snapshot);
        let start_difference = start_offset - start_byte_offset;
        let end_offset = TO::to_offset(&query_range.end, &buffer_snapshot);
        let end_difference = end_offset - start_byte_offset;

        // Current range has associated linked ranges.
        let mut linked_edits = HashMap::<_, Vec<_>>::default();
        for range in linked_ranges.iter() {
            let start_offset = TO::to_offset(&range.start, &buffer_snapshot);
            let end_offset = start_offset + end_difference;
            let start_offset = start_offset + start_difference;
            if start_offset > buffer_snapshot.len() || end_offset > buffer_snapshot.len() {
                continue;
            }
            if self.selections.disjoint_anchor_ranges().any(|s| {
                let Some((selection_start, _)) =
                    multibuffer_snapshot.anchor_to_buffer_anchor(s.start)
                else {
                    return false;
                };
                let Some((selection_end, _)) = multibuffer_snapshot.anchor_to_buffer_anchor(s.end)
                else {
                    return false;
                };
                if selection_start.buffer_id != query_range.start.buffer_id
                    || selection_end.buffer_id != query_range.end.buffer_id
                {
                    return false;
                }
                TO::to_offset(&selection_start, &buffer_snapshot) <= end_offset
                    && TO::to_offset(&selection_end, &buffer_snapshot) >= start_offset
            }) {
                continue;
            }
            let start = buffer_snapshot.anchor_after(start_offset);
            let end = buffer_snapshot.anchor_after(end_offset);
            linked_edits
                .entry(buffer.clone())
                .or_default()
                .push(start..end);
        }
        Some(linked_edits)
    }
    pub fn set_mark(&mut self, _: &actions::SetMark, window: &mut Window, cx: &mut Context<Self>) {
        if self.selection_mark_mode {
            self.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
                s.move_with(&mut |_, sel| {
                    sel.collapse_to(sel.head(), SelectionGoal::None);
                });
            })
        }
        self.selection_mark_mode = true;
        cx.notify();
    }
    pub fn swap_selection_ends(
        &mut self,
        _: &actions::SwapSelectionEnds,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
            s.move_with(&mut |_, sel| {
                if sel.start != sel.end {
                    sel.reversed = !sel.reversed
                }
            });
        });
        self.request_autoscroll(Autoscroll::newest(), cx);
        cx.notify();
    }
    /// Replaces the editor's selections with the provided `text`, applying the
    /// given `autoindent_mode` (`None` will skip autoindentation).
    ///
    /// Early returns if the editor is in read-only mode, without applying any
    /// edits.
    pub(super) fn replace_selections(
        &mut self,
        text: &str,
        autoindent_mode: Option<AutoindentMode>,
        window: &mut Window,
        cx: &mut Context<Self>,
        apply_linked_edits: bool,
    ) {
        if self.read_only(cx) {
            return;
        }

        let text: Arc<str> = text.into();
        self.transact(window, cx, |this, window, cx| {
            let old_selections = this.selections.all_adjusted(&this.display_snapshot(cx));
            let linked_edits = if apply_linked_edits {
                this.linked_edits_for_selections(text.clone(), cx)
            } else {
                LinkedEdits::new()
            };

            let selection_anchors = this.buffer.update(cx, |buffer, cx| {
                let anchors = {
                    let snapshot = buffer.read(cx);
                    old_selections
                        .iter()
                        .map(|s| {
                            let anchor = snapshot.anchor_after(s.head());
                            s.map(|_| anchor)
                        })
                        .collect::<Vec<_>>()
                };
                buffer.edit(
                    old_selections
                        .iter()
                        .map(|s| (s.start..s.end, text.clone())),
                    autoindent_mode,
                    cx,
                );
                anchors
            });

            linked_edits.apply(cx);

            this.change_selections(Default::default(), window, cx, |s| {
                s.select_anchors(selection_anchors);
            });

            if apply_linked_edits {
                refresh_linked_ranges(this, window, cx);
            }

            cx.notify();
        });
    }
    /// Collects linked edits for the current selections, pairing each linked
    /// range with `text`.
    pub fn linked_edits_for_selections(&self, text: Arc<str>, cx: &App) -> LinkedEdits {
        let multibuffer_snapshot = self.buffer().read(cx).snapshot(cx);
        let mut linked_edits = LinkedEdits::new();
        if !self.linked_edit_ranges.is_empty() {
            for selection in self.selections.disjoint_anchors() {
                let Some((_, range)) =
                    multibuffer_snapshot.anchor_range_to_buffer_anchor_range(selection.range())
                else {
                    continue;
                };
                linked_edits.push(self, range, text.clone(), cx);
            }
        }
        linked_edits
    }
    /// Deletes the content covered by the current selections and applies
    /// linked edits.
    pub fn delete_selections_with_linked_edits(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.replace_selections("", None, window, cx, true);
    }
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_linked_edit_ranges_for_testing(
        &mut self,
        ranges: Vec<(Range<Point>, Vec<Range<Point>>)>,
        cx: &mut Context<Self>,
    ) -> Option<()> {
        let Some((buffer, _)) = self
            .buffer
            .read(cx)
            .text_anchor_for_position(self.selections.newest_anchor().start, cx)
        else {
            return None;
        };
        let buffer = buffer.read(cx);
        let buffer_id = buffer.remote_id();
        let mut linked_ranges = Vec::with_capacity(ranges.len());
        for (base_range, linked_ranges_points) in ranges {
            let base_anchor =
                buffer.anchor_before(base_range.start)..buffer.anchor_after(base_range.end);
            let linked_anchors = linked_ranges_points
                .into_iter()
                .map(|range| buffer.anchor_before(range.start)..buffer.anchor_after(range.end))
                .collect();
            linked_ranges.push((base_anchor, linked_anchors));
        }
        let mut map = HashMap::default();
        map.insert(buffer_id, linked_ranges);
        self.linked_edit_ranges = linked_editing_ranges::LinkedEditingRanges(map);
        Some(())
    }
    /// If any empty selections is touching the start of its innermost containing autoclose
    /// region, expand it to select the brackets.
    pub(super) fn select_autoclose_pair(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let selections = self
            .selections
            .all::<MultiBufferOffset>(&self.display_snapshot(cx));
        let buffer = self.buffer.read(cx).read(cx);
        let new_selections = self
            .selections_with_autoclose_regions(selections, &buffer)
            .map(|(mut selection, region)| {
                if !selection.is_empty() {
                    return selection;
                }

                if let Some(region) = region {
                    let mut range = region.range.to_offset(&buffer);
                    if selection.start == range.start && range.start.0 >= region.pair.start.len() {
                        range.start -= region.pair.start.len();
                        if buffer.contains_str_at(range.start, &region.pair.start)
                            && buffer.contains_str_at(range.end, &region.pair.end)
                        {
                            range.end += region.pair.end.len();
                            selection.start = range.start;
                            selection.end = range.end;

                            return selection;
                        }
                    }
                }

                let always_treat_brackets_as_autoclosed = buffer
                    .language_settings_at(selection.start, cx)
                    .always_treat_brackets_as_autoclosed;

                if !always_treat_brackets_as_autoclosed {
                    return selection;
                }

                if let Some(scope) = buffer.language_scope_at(selection.start) {
                    for (pair, enabled) in scope.brackets() {
                        if !enabled || !pair.close {
                            continue;
                        }

                        if buffer.contains_str_at(selection.start, &pair.end) {
                            let pair_start_len = pair.start.len();
                            if buffer.contains_str_at(
                                selection.start.saturating_sub_usize(pair_start_len),
                                &pair.start,
                            ) {
                                selection.start -= pair_start_len;
                                selection.end += pair.end.len();

                                return selection;
                            }
                        }
                    }
                }

                selection
            })
            .collect();

        drop(buffer);
        self.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
            selections.select(new_selections)
        });
    }
    /// Iterate the given selections, and for each one, find the smallest surrounding
    /// autoclose region. This uses the ordering of the selections and the autoclose
    /// regions to avoid repeated comparisons.
    pub(super) fn selections_with_autoclose_regions<'a, D: ToOffset + Clone>(
        &'a self,
        selections: impl IntoIterator<Item = Selection<D>>,
        buffer: &'a MultiBufferSnapshot,
    ) -> impl Iterator<Item = (Selection<D>, Option<&'a AutocloseRegion>)> {
        let mut i = 0;
        let mut regions = self.autoclose_regions.as_slice();
        selections.into_iter().map(move |selection| {
            let range = selection.start.to_offset(buffer)..selection.end.to_offset(buffer);

            let mut enclosing = None;
            while let Some(pair_state) = regions.get(i) {
                if pair_state.range.end.to_offset(buffer) < range.start {
                    regions = &regions[i + 1..];
                    i = 0;
                } else if pair_state.range.start.to_offset(buffer) > range.end {
                    break;
                } else {
                    if pair_state.selection_id == selection.id {
                        enclosing = Some(pair_state);
                    }
                    i += 1;
                }
            }

            (selection, enclosing)
        })
    }
    /// Remove any autoclose regions that no longer contain their selection or have invalid anchors in ranges.
    pub(super) fn invalidate_autoclose_regions(
        &mut self,
        mut selections: &[Selection<Anchor>],
        buffer: &MultiBufferSnapshot,
    ) {
        self.autoclose_regions.retain(|state| {
            if !state.range.start.is_valid(buffer) || !state.range.end.is_valid(buffer) {
                return false;
            }

            let mut i = 0;
            while let Some(selection) = selections.get(i) {
                if selection.end.cmp(&state.range.start, buffer).is_lt() {
                    selections = &selections[1..];
                    continue;
                }
                if selection.start.cmp(&state.range.end, buffer).is_gt() {
                    break;
                }
                if selection.id == state.selection_id {
                    return true;
                } else {
                    i += 1;
                }
            }
            false
        });
    }
    pub fn set_selections_from_remote(
        &mut self,
        selections: Vec<Selection<Anchor>>,
        pending_selection: Option<Selection<Anchor>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let old_cursor_position = self.selections.newest_anchor().head();
        self.selections
            .change_with(&self.display_snapshot(cx), |s| {
                s.select_anchors(selections);
                if let Some(pending_selection) = pending_selection {
                    s.set_pending(pending_selection, SelectMode::Character);
                } else {
                    s.clear_pending();
                }
            });
        self.selections_did_change(
            false,
            &old_cursor_position,
            SelectionEffects::default(),
            window,
            cx,
        );
    }
    pub fn select_left(&mut self, _: &SelectLeft, window: &mut Window, cx: &mut Context<Self>) {
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, _| (movement::left(map, head), SelectionGoal::None));
        })
    }
    pub fn select_right(&mut self, _: &SelectRight, window: &mut Window, cx: &mut Context<Self>) {
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, _| {
                (movement::right(map, head), SelectionGoal::None)
            });
        });
    }
    pub fn select_down_by_lines(
        &mut self,
        action: &SelectDownByLines,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let text_layout_details = &self.text_layout_details(window, cx);
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, goal| {
                movement::down_by_rows(map, head, action.lines, goal, false, text_layout_details)
            })
        })
    }
    pub fn select_up_by_lines(
        &mut self,
        action: &SelectUpByLines,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let text_layout_details = &self.text_layout_details(window, cx);
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, goal| {
                movement::up_by_rows(map, head, action.lines, goal, false, text_layout_details)
            })
        })
    }
    pub fn select_page_up(
        &mut self,
        _: &SelectPageUp,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(row_count) = self.visible_row_count() else {
            return;
        };

        let text_layout_details = &self.text_layout_details(window, cx);

        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, goal| {
                movement::up_by_rows(map, head, row_count, goal, false, text_layout_details)
            })
        })
    }
    pub fn select_up(&mut self, _: &SelectUp, window: &mut Window, cx: &mut Context<Self>) {
        let text_layout_details = &self.text_layout_details(window, cx);
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, goal| {
                movement::up(map, head, goal, false, text_layout_details)
            })
        })
    }
    pub fn select_page_down(
        &mut self,
        _: &SelectPageDown,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(row_count) = self.visible_row_count() else {
            return;
        };

        let text_layout_details = &self.text_layout_details(window, cx);

        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, goal| {
                movement::down_by_rows(map, head, row_count, goal, false, text_layout_details)
            })
        })
    }
    pub fn select_down(&mut self, _: &SelectDown, window: &mut Window, cx: &mut Context<Self>) {
        let text_layout_details = &self.text_layout_details(window, cx);
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, goal| {
                movement::down(map, head, goal, false, text_layout_details)
            })
        });
    }
    pub fn select_to_previous_word_start(
        &mut self,
        _: &SelectToPreviousWordStart,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, _| {
                (
                    movement::previous_word_start(map, head),
                    SelectionGoal::None,
                )
            });
        })
    }
    pub fn select_to_previous_subword_start(
        &mut self,
        _: &SelectToPreviousSubwordStart,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, _| {
                (
                    movement::previous_subword_start(map, head),
                    SelectionGoal::None,
                )
            });
        })
    }
    pub fn select_to_next_word_end(
        &mut self,
        _: &SelectToNextWordEnd,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, _| {
                (movement::next_word_end(map, head), SelectionGoal::None)
            });
        })
    }
    pub fn select_to_next_subword_end(
        &mut self,
        _: &SelectToNextSubwordEnd,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, _| {
                (movement::next_subword_end(map, head), SelectionGoal::None)
            });
        })
    }
    pub fn select_to_beginning_of_line(
        &mut self,
        action: &SelectToBeginningOfLine,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let stop_at_indent = action.stop_at_indent && !self.mode.is_single_line();
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, _| {
                (
                    movement::indented_line_beginning(
                        map,
                        head,
                        action.stop_at_soft_wraps,
                        stop_at_indent,
                    ),
                    SelectionGoal::None,
                )
            });
        });
    }
    pub fn select_to_end_of_line(
        &mut self,
        action: &SelectToEndOfLine,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, _| {
                (
                    movement::line_end(map, head, action.stop_at_soft_wraps),
                    SelectionGoal::None,
                )
            });
        })
    }
    pub fn select_to_start_of_paragraph(
        &mut self,
        _: &SelectToStartOfParagraph,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if matches!(self.mode, EditorMode::SingleLine) {
            cx.propagate();
            return;
        }
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, _| {
                (
                    movement::start_of_paragraph(map, head, 1),
                    SelectionGoal::None,
                )
            });
        })
    }
    pub fn select_to_end_of_paragraph(
        &mut self,
        _: &SelectToEndOfParagraph,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if matches!(self.mode, EditorMode::SingleLine) {
            cx.propagate();
            return;
        }
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, _| {
                (
                    movement::end_of_paragraph(map, head, 1),
                    SelectionGoal::None,
                )
            });
        })
    }
    pub fn select_to_start_of_excerpt(
        &mut self,
        _: &SelectToStartOfExcerpt,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if matches!(self.mode, EditorMode::SingleLine) {
            cx.propagate();
            return;
        }
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, _| {
                (
                    movement::start_of_excerpt(map, head, workspace::searchable::Direction::Prev),
                    SelectionGoal::None,
                )
            });
        })
    }
    pub fn select_to_start_of_next_excerpt(
        &mut self,
        _: &SelectToStartOfNextExcerpt,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if matches!(self.mode, EditorMode::SingleLine) {
            cx.propagate();
            return;
        }
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, _| {
                (
                    movement::start_of_excerpt(map, head, workspace::searchable::Direction::Next),
                    SelectionGoal::None,
                )
            });
        })
    }
    pub fn select_to_end_of_excerpt(
        &mut self,
        _: &SelectToEndOfExcerpt,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if matches!(self.mode, EditorMode::SingleLine) {
            cx.propagate();
            return;
        }
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, _| {
                (
                    movement::end_of_excerpt(map, head, workspace::searchable::Direction::Next),
                    SelectionGoal::None,
                )
            });
        })
    }
    pub fn select_to_end_of_previous_excerpt(
        &mut self,
        _: &SelectToEndOfPreviousExcerpt,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if matches!(self.mode, EditorMode::SingleLine) {
            cx.propagate();
            return;
        }
        self.change_selections(Default::default(), window, cx, |s| {
            s.move_heads_with(&mut |map, head, _| {
                (
                    movement::end_of_excerpt(map, head, workspace::searchable::Direction::Prev),
                    SelectionGoal::None,
                )
            });
        })
    }
    pub fn select_to_beginning(
        &mut self,
        _: &SelectToBeginning,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let mut selection = self.selections.last::<Point>(&self.display_snapshot(cx));
        selection.set_head(Point::zero(), SelectionGoal::None);
        self.change_selections(Default::default(), window, cx, |s| {
            s.select(vec![selection]);
        });
    }
    pub fn select_to_end(&mut self, _: &SelectToEnd, window: &mut Window, cx: &mut Context<Self>) {
        let buffer = self.buffer.read(cx).snapshot(cx);
        let mut selection = self
            .selections
            .first::<MultiBufferOffset>(&self.display_snapshot(cx));
        selection.set_head(buffer.len(), SelectionGoal::None);
        self.change_selections(Default::default(), window, cx, |s| {
            s.select(vec![selection]);
        });
    }
    pub fn select_all(&mut self, _: &SelectAll, window: &mut Window, cx: &mut Context<Self>) {
        self.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
            s.select_ranges(vec![Anchor::Min..Anchor::Max]);
        });
    }
    pub fn select_line(&mut self, _: &SelectLine, window: &mut Window, cx: &mut Context<Self>) {
        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        let mut selections = self.selections.all::<Point>(&display_map);
        let max_point = display_map.buffer_snapshot().max_point();
        for selection in &mut selections {
            let rows = selection.spanned_rows(true, &display_map);
            selection.start = Point::new(rows.start.0, 0);
            selection.end = cmp::min(max_point, Point::new(rows.end.0, 0));
            selection.reversed = false;
        }
        self.change_selections(Default::default(), window, cx, |s| {
            s.select(selections);
        });
    }
    pub fn split_selection_into_lines(
        &mut self,
        action: &SplitSelectionIntoLines,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let selections = self
            .selections
            .all::<Point>(&self.display_snapshot(cx))
            .into_iter()
            .map(|selection| selection.start..selection.end)
            .collect::<Vec<_>>();
        self.unfold_ranges(&selections, true, false, cx);

        let mut new_selection_ranges = Vec::new();
        {
            let buffer = self.buffer.read(cx).read(cx);
            for selection in selections {
                for row in selection.start.row..selection.end.row {
                    let line_start = Point::new(row, 0);
                    let line_end = Point::new(row, buffer.line_len(MultiBufferRow(row)));

                    if action.keep_selections {
                        // Keep the selection range for each line
                        let selection_start = if row == selection.start.row {
                            selection.start
                        } else {
                            line_start
                        };
                        new_selection_ranges.push(selection_start..line_end);
                    } else {
                        // Collapse to cursor at end of line
                        new_selection_ranges.push(line_end..line_end);
                    }
                }

                let is_multiline_selection = selection.start.row != selection.end.row;
                // Don't insert last one if it's a multi-line selection ending at the start of a line,
                // so this action feels more ergonomic when paired with other selection operations
                let should_skip_last = is_multiline_selection && selection.end.column == 0;
                if !should_skip_last {
                    if action.keep_selections {
                        if is_multiline_selection {
                            let line_start = Point::new(selection.end.row, 0);
                            new_selection_ranges.push(line_start..selection.end);
                        } else {
                            new_selection_ranges.push(selection.start..selection.end);
                        }
                    } else {
                        new_selection_ranges.push(selection.end..selection.end);
                    }
                }
            }
        }
        self.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
            s.select_ranges(new_selection_ranges);
        });
    }
    pub fn add_selection_above(
        &mut self,
        action: &AddSelectionAbove,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.add_selection(true, action.skip_soft_wrap, window, cx);
    }
    pub fn add_selection_below(
        &mut self,
        action: &AddSelectionBelow,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.add_selection(false, action.skip_soft_wrap, window, cx);
    }
    fn add_selection(
        &mut self,
        above: bool,
        skip_soft_wrap: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        let all_selections = self.selections.all::<Point>(&display_map);
        let text_layout_details = self.text_layout_details(window, cx);

        let (mut columnar_selections, new_selections_to_columnarize) = {
            if let Some(state) = self.add_selections_state.as_ref() {
                let columnar_selection_ids: HashSet<_> = state
                    .groups
                    .iter()
                    .flat_map(|group| group.stack.iter())
                    .copied()
                    .collect();

                all_selections
                    .into_iter()
                    .partition(|s| columnar_selection_ids.contains(&s.id))
            } else {
                (Vec::new(), all_selections)
            }
        };

        let mut state = self
            .add_selections_state
            .take()
            .unwrap_or_else(|| AddSelectionsState { groups: Vec::new() });

        for selection in new_selections_to_columnarize {
            let range = selection.display_range(&display_map).sorted();
            let start_x = display_map.x_for_display_point(range.start, &text_layout_details);
            let end_x = display_map.x_for_display_point(range.end, &text_layout_details);
            let positions = start_x.min(end_x)..start_x.max(end_x);
            let mut stack = Vec::new();
            for row in range.start.row().0..=range.end.row().0 {
                if let Some(selection) = self.selections.build_columnar_selection(
                    &display_map,
                    DisplayRow(row),
                    &positions,
                    selection.reversed,
                    &text_layout_details,
                ) {
                    stack.push(selection.id);
                    columnar_selections.push(selection);
                }
            }
            if !stack.is_empty() {
                if above {
                    stack.reverse();
                }
                state.groups.push(AddSelectionsGroup { above, stack });
            }
        }

        let mut final_selections = Vec::new();
        let end_row = if above {
            DisplayRow(0)
        } else {
            display_map.max_point().row()
        };

        // When `skip_soft_wrap` is true, we use UTF-16 columns instead of pixel
        // positions to place new selections, so we need to keep track of the
        // column range of the oldest selection in each group, because
        // intermediate selections may have been clamped to shorter lines.
        let mut goal_columns_by_selection_id = if skip_soft_wrap {
            let mut map = HashMap::default();
            for group in state.groups.iter() {
                if let Some(oldest_id) = group.stack.first() {
                    if let Some(oldest_selection) =
                        columnar_selections.iter().find(|s| s.id == *oldest_id)
                    {
                        let snapshot = display_map.buffer_snapshot();
                        let start_col =
                            snapshot.point_to_point_utf16(oldest_selection.start).column;
                        let end_col = snapshot.point_to_point_utf16(oldest_selection.end).column;
                        let goal_columns = start_col.min(end_col)..start_col.max(end_col);
                        for id in &group.stack {
                            map.insert(*id, goal_columns.clone());
                        }
                    }
                }
            }
            map
        } else {
            HashMap::default()
        };

        let mut last_added_item_per_group = HashMap::default();
        for group in state.groups.iter_mut() {
            if let Some(last_id) = group.stack.last() {
                last_added_item_per_group.insert(*last_id, group);
            }
        }

        for selection in columnar_selections {
            if let Some(group) = last_added_item_per_group.get_mut(&selection.id) {
                if above == group.above {
                    let range = selection.display_range(&display_map).sorted();
                    debug_assert_eq!(range.start.row(), range.end.row());
                    let row = range.start.row();
                    let positions =
                        if let SelectionGoal::HorizontalRange { start, end } = selection.goal {
                            Pixels::from(start)..Pixels::from(end)
                        } else {
                            let start_x =
                                display_map.x_for_display_point(range.start, &text_layout_details);
                            let end_x =
                                display_map.x_for_display_point(range.end, &text_layout_details);
                            start_x.min(end_x)..start_x.max(end_x)
                        };

                    let maybe_new_selection = if skip_soft_wrap {
                        let goal_columns = goal_columns_by_selection_id
                            .remove(&selection.id)
                            .unwrap_or_else(|| {
                                let snapshot = display_map.buffer_snapshot();
                                let start_col =
                                    snapshot.point_to_point_utf16(selection.start).column;
                                let end_col = snapshot.point_to_point_utf16(selection.end).column;
                                start_col.min(end_col)..start_col.max(end_col)
                            });
                        self.selections.find_next_columnar_selection_by_buffer_row(
                            &display_map,
                            row,
                            end_row,
                            above,
                            &goal_columns,
                            selection.reversed,
                            &text_layout_details,
                        )
                    } else {
                        self.selections.find_next_columnar_selection_by_display_row(
                            &display_map,
                            row,
                            end_row,
                            above,
                            &positions,
                            selection.reversed,
                            &text_layout_details,
                        )
                    };

                    if let Some(new_selection) = maybe_new_selection {
                        group.stack.push(new_selection.id);
                        if above {
                            final_selections.push(new_selection);
                            final_selections.push(selection);
                        } else {
                            final_selections.push(selection);
                            final_selections.push(new_selection);
                        }
                    } else {
                        final_selections.push(selection);
                    }
                } else {
                    group.stack.pop();
                }
            } else {
                final_selections.push(selection);
            }
        }

        self.change_selections(Default::default(), window, cx, |s| {
            s.select(final_selections);
        });

        let final_selection_ids: HashSet<_> = self
            .selections
            .all::<Point>(&display_map)
            .iter()
            .map(|s| s.id)
            .collect();
        state.groups.retain_mut(|group| {
            // selections might get merged above so we remove invalid items from stacks
            group.stack.retain(|id| final_selection_ids.contains(id));

            // single selection in stack can be treated as initial state
            group.stack.len() > 1
        });

        if !state.groups.is_empty() {
            self.add_selections_state = Some(state);
        }
    }
    pub fn insert_snippet_at_selections(
        &mut self,
        action: &InsertSnippet,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.try_insert_snippet_at_selections(action, window, cx)
            .log_err();
    }
    fn try_insert_snippet_at_selections(
        &mut self,
        action: &InsertSnippet,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let insertion_ranges = self
            .selections
            .all::<MultiBufferOffset>(&self.display_snapshot(cx))
            .into_iter()
            .map(|selection| selection.range())
            .collect_vec();

        let snippet = if let Some(snippet_body) = &action.snippet {
            if action.language.is_none() && action.name.is_none() {
                Snippet::parse(snippet_body)?
            } else {
                bail!("`snippet` is mutually exclusive with `language` and `name`")
            }
        } else if let Some(name) = &action.name {
            let project = self.project().context("no project")?;
            let snippet_store = project.read(cx).snippets().read(cx);
            let snippet = snippet_store
                .snippets_for(action.language.clone(), cx)
                .into_iter()
                .find(|snippet| snippet.name == *name)
                .context("snippet not found")?;
            Snippet::parse(&snippet.body)?
        } else {
            // todo(andrew): open modal to select snippet
            bail!("`name` or `snippet` is required")
        };

        self.insert_snippet(&insertion_ranges, snippet, window, cx)
    }
    fn select_match_ranges(
        &mut self,
        range: Range<MultiBufferOffset>,
        reversed: bool,
        replace_newest: bool,
        auto_scroll: Option<Autoscroll>,
        window: &mut Window,
        cx: &mut Context<Editor>,
    ) {
        self.unfold_ranges(
            std::slice::from_ref(&range),
            false,
            auto_scroll.is_some(),
            cx,
        );
        let effects = if let Some(scroll) = auto_scroll {
            SelectionEffects::scroll(scroll)
        } else {
            SelectionEffects::no_scroll()
        };
        self.change_selections(effects, window, cx, |s| {
            if replace_newest {
                s.delete(s.newest_anchor().id);
            }
            if reversed {
                s.insert_range(range.end..range.start);
            } else {
                s.insert_range(range);
            }
        });
    }
    pub fn select_next_match_internal(
        &mut self,
        display_map: &DisplaySnapshot,
        replace_newest: bool,
        autoscroll: Option<Autoscroll>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let buffer = display_map.buffer_snapshot();
        let mut selections = self.selections.all::<MultiBufferOffset>(&display_map);
        if let Some(mut select_next_state) = self.select_next_state.take() {
            let query = &select_next_state.query;
            if !select_next_state.done {
                let first_selection = selections.iter().min_by_key(|s| s.id).unwrap();
                let last_selection = selections.iter().max_by_key(|s| s.id).unwrap();
                let mut next_selected_range = None;

                let bytes_after_last_selection =
                    buffer.bytes_in_range(last_selection.end..buffer.len());
                let bytes_before_first_selection =
                    buffer.bytes_in_range(MultiBufferOffset(0)..first_selection.start);
                let query_matches = query
                    .stream_find_iter(bytes_after_last_selection)
                    .map(|result| (last_selection.end, result))
                    .chain(
                        query
                            .stream_find_iter(bytes_before_first_selection)
                            .map(|result| (MultiBufferOffset(0), result)),
                    );

                for (start_offset, query_match) in query_matches {
                    let query_match = query_match.unwrap(); // can only fail due to I/O
                    let offset_range =
                        start_offset + query_match.start()..start_offset + query_match.end();

                    if !select_next_state.wordwise
                        || (!buffer.is_inside_word(offset_range.start, None)
                            && !buffer.is_inside_word(offset_range.end, None))
                    {
                        let idx = selections
                            .partition_point(|selection| selection.end <= offset_range.start);
                        let overlaps = selections
                            .get(idx)
                            .map_or(false, |selection| selection.start < offset_range.end);

                        if !overlaps {
                            next_selected_range = Some(offset_range);
                            break;
                        }
                    }
                }

                if let Some(next_selected_range) = next_selected_range {
                    self.select_match_ranges(
                        next_selected_range,
                        last_selection.reversed,
                        replace_newest,
                        autoscroll,
                        window,
                        cx,
                    );
                } else {
                    select_next_state.done = true;
                }
            }

            self.select_next_state = Some(select_next_state);
        } else {
            let mut only_carets = true;
            let mut same_text_selected = true;
            let mut selected_text = None;

            let mut selections_iter = selections.iter().peekable();
            while let Some(selection) = selections_iter.next() {
                if selection.start != selection.end {
                    only_carets = false;
                }

                if same_text_selected {
                    if selected_text.is_none() {
                        selected_text =
                            Some(buffer.text_for_range(selection.range()).collect::<String>());
                    }

                    if let Some(next_selection) = selections_iter.peek() {
                        if next_selection.len() == selection.len() {
                            let next_selected_text = buffer
                                .text_for_range(next_selection.range())
                                .collect::<String>();
                            if Some(next_selected_text) != selected_text {
                                same_text_selected = false;
                                selected_text = None;
                            }
                        } else {
                            same_text_selected = false;
                            selected_text = None;
                        }
                    }
                }
            }

            if only_carets {
                for selection in &mut selections {
                    let (word_range, _) = buffer.surrounding_word(selection.start, None);
                    selection.start = word_range.start;
                    selection.end = word_range.end;
                    selection.goal = SelectionGoal::None;
                    selection.reversed = false;
                    self.select_match_ranges(
                        selection.start..selection.end,
                        selection.reversed,
                        replace_newest,
                        autoscroll,
                        window,
                        cx,
                    );
                }

                if selections.len() == 1 {
                    let selection = selections
                        .last()
                        .expect("ensured that there's only one selection");
                    let query = buffer
                        .text_for_range(selection.start..selection.end)
                        .collect::<String>();
                    let is_empty = query.is_empty();
                    let select_state = SelectNextState {
                        query: self.build_query(&[query], cx)?,
                        wordwise: true,
                        done: is_empty,
                    };
                    self.select_next_state = Some(select_state);
                } else {
                    self.select_next_state = None;
                }
            } else if let Some(selected_text) = selected_text {
                self.select_next_state = Some(SelectNextState {
                    query: self.build_query(&[selected_text], cx)?,
                    wordwise: false,
                    done: false,
                });
                self.select_next_match_internal(
                    display_map,
                    replace_newest,
                    autoscroll,
                    window,
                    cx,
                )?;
            }
        }
        Ok(())
    }
    pub fn select_all_matches(
        &mut self,
        _action: &SelectAllMatches,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));

        self.select_next_match_internal(&display_map, false, None, window, cx)?;
        let Some(select_next_state) = self.select_next_state.as_mut().filter(|state| !state.done)
        else {
            return Ok(());
        };

        let mut new_selections = Vec::new();
        let initial_selection = self.selections.oldest::<MultiBufferOffset>(&display_map);
        let reversed = initial_selection.reversed;
        let buffer = display_map.buffer_snapshot();
        let query_matches = select_next_state
            .query
            .stream_find_iter(buffer.bytes_in_range(MultiBufferOffset(0)..buffer.len()));

        for query_match in query_matches.into_iter() {
            let query_match = query_match.context("query match for select all action")?; // can only fail due to I/O
            let offset_range = if reversed {
                MultiBufferOffset(query_match.end())..MultiBufferOffset(query_match.start())
            } else {
                MultiBufferOffset(query_match.start())..MultiBufferOffset(query_match.end())
            };

            let is_partial_word_match = select_next_state.wordwise
                && (buffer.is_inside_word(offset_range.start, None)
                    || buffer.is_inside_word(offset_range.end, None));

            let is_initial_selection = MultiBufferOffset(query_match.start())
                == initial_selection.start
                && MultiBufferOffset(query_match.end()) == initial_selection.end;

            if !is_partial_word_match && !is_initial_selection {
                new_selections.push(offset_range);
            }
        }

        // Ensure that the initial range is the last selection, as
        // `MutableSelectionsCollection::select_ranges` makes the last selection
        // the newest selection, which the editor then relies on as the primary
        // cursor for scroll targeting. Without this, the last match would then
        // be automatically focused when the user started editing the selected
        // matches.
        let initial_directed_range = if reversed {
            initial_selection.end..initial_selection.start
        } else {
            initial_selection.start..initial_selection.end
        };
        new_selections.push(initial_directed_range);

        select_next_state.done = true;
        self.unfold_ranges(&new_selections, false, false, cx);
        self.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
            selections.select_ranges(new_selections)
        });

        Ok(())
    }
    pub fn select_next(
        &mut self,
        action: &SelectNext,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        self.select_next_match_internal(
            &display_map,
            action.replace_newest,
            Some(Autoscroll::newest()),
            window,
            cx,
        )
    }
    pub fn select_previous(
        &mut self,
        action: &SelectPrevious,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        let buffer = display_map.buffer_snapshot();
        let mut selections = self.selections.all::<MultiBufferOffset>(&display_map);
        if let Some(mut select_prev_state) = self.select_prev_state.take() {
            let query = &select_prev_state.query;
            if !select_prev_state.done {
                let first_selection = selections.iter().min_by_key(|s| s.id).unwrap();
                let last_selection = selections.iter().max_by_key(|s| s.id).unwrap();
                let mut next_selected_range = None;
                // When we're iterating matches backwards, the oldest match will actually be the furthest one in the buffer.
                let bytes_before_last_selection =
                    buffer.reversed_bytes_in_range(MultiBufferOffset(0)..last_selection.start);
                let bytes_after_first_selection =
                    buffer.reversed_bytes_in_range(first_selection.end..buffer.len());
                let query_matches = query
                    .stream_find_iter(bytes_before_last_selection)
                    .map(|result| (last_selection.start, result))
                    .chain(
                        query
                            .stream_find_iter(bytes_after_first_selection)
                            .map(|result| (buffer.len(), result)),
                    );
                for (end_offset, query_match) in query_matches {
                    let query_match = query_match.unwrap(); // can only fail due to I/O
                    let offset_range =
                        end_offset - query_match.end()..end_offset - query_match.start();

                    if !select_prev_state.wordwise
                        || (!buffer.is_inside_word(offset_range.start, None)
                            && !buffer.is_inside_word(offset_range.end, None))
                    {
                        next_selected_range = Some(offset_range);
                        break;
                    }
                }

                if let Some(next_selected_range) = next_selected_range {
                    self.select_match_ranges(
                        next_selected_range,
                        last_selection.reversed,
                        action.replace_newest,
                        Some(Autoscroll::newest()),
                        window,
                        cx,
                    );
                } else {
                    select_prev_state.done = true;
                }
            }

            self.select_prev_state = Some(select_prev_state);
        } else {
            let mut only_carets = true;
            let mut same_text_selected = true;
            let mut selected_text = None;

            let mut selections_iter = selections.iter().peekable();
            while let Some(selection) = selections_iter.next() {
                if selection.start != selection.end {
                    only_carets = false;
                }

                if same_text_selected {
                    if selected_text.is_none() {
                        selected_text =
                            Some(buffer.text_for_range(selection.range()).collect::<String>());
                    }

                    if let Some(next_selection) = selections_iter.peek() {
                        if next_selection.len() == selection.len() {
                            let next_selected_text = buffer
                                .text_for_range(next_selection.range())
                                .collect::<String>();
                            if Some(next_selected_text) != selected_text {
                                same_text_selected = false;
                                selected_text = None;
                            }
                        } else {
                            same_text_selected = false;
                            selected_text = None;
                        }
                    }
                }
            }

            if only_carets {
                for selection in &mut selections {
                    let (word_range, _) = buffer.surrounding_word(selection.start, None);
                    selection.start = word_range.start;
                    selection.end = word_range.end;
                    selection.goal = SelectionGoal::None;
                    selection.reversed = false;
                    self.select_match_ranges(
                        selection.start..selection.end,
                        selection.reversed,
                        action.replace_newest,
                        Some(Autoscroll::newest()),
                        window,
                        cx,
                    );
                }
                if selections.len() == 1 {
                    let selection = selections
                        .last()
                        .expect("ensured that there's only one selection");
                    let query = buffer
                        .text_for_range(selection.start..selection.end)
                        .collect::<String>();
                    let is_empty = query.is_empty();
                    let select_state = SelectNextState {
                        query: self.build_query(&[query.chars().rev().collect::<String>()], cx)?,
                        wordwise: true,
                        done: is_empty,
                    };
                    self.select_prev_state = Some(select_state);
                } else {
                    self.select_prev_state = None;
                }
            } else if let Some(selected_text) = selected_text {
                self.select_prev_state = Some(SelectNextState {
                    query: self
                        .build_query(&[selected_text.chars().rev().collect::<String>()], cx)?,
                    wordwise: false,
                    done: false,
                });
                self.select_previous(action, window, cx)?;
            }
        }
        Ok(())
    }
    pub fn select_enclosing_symbol(
        &mut self,
        _: &SelectEnclosingSymbol,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let buffer = self.buffer.read(cx).snapshot(cx);
        let old_selections = self
            .selections
            .all::<MultiBufferOffset>(&self.display_snapshot(cx))
            .into_boxed_slice();

        fn update_selection(
            selection: &Selection<MultiBufferOffset>,
            buffer_snap: &MultiBufferSnapshot,
        ) -> Option<Selection<MultiBufferOffset>> {
            let cursor = selection.head();
            let (_buffer_id, symbols) = buffer_snap.symbols_containing(cursor, None)?;
            for symbol in symbols.iter().rev() {
                let start = symbol.range.start.to_offset(buffer_snap);
                let end = symbol.range.end.to_offset(buffer_snap);
                let new_range = start..end;
                if start < selection.start || end > selection.end {
                    return Some(Selection {
                        id: selection.id,
                        start: new_range.start,
                        end: new_range.end,
                        goal: SelectionGoal::None,
                        reversed: selection.reversed,
                    });
                }
            }
            None
        }

        let mut selected_larger_symbol = false;
        let new_selections = old_selections
            .iter()
            .map(|selection| match update_selection(selection, &buffer) {
                Some(new_selection) => {
                    if new_selection.range() != selection.range() {
                        selected_larger_symbol = true;
                    }
                    new_selection
                }
                None => selection.clone(),
            })
            .collect::<Vec<_>>();

        if selected_larger_symbol {
            self.change_selections(Default::default(), window, cx, |s| {
                s.select(new_selections);
            });
        }
    }
    pub fn select_larger_syntax_node(
        &mut self,
        _: &SelectLargerSyntaxNode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(visible_row_count) = self.visible_row_count() else {
            return;
        };
        let old_selections: Box<[_]> = self
            .selections
            .all::<MultiBufferOffset>(&self.display_snapshot(cx))
            .into();
        if old_selections.is_empty() {
            return;
        }

        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        let buffer = self.buffer.read(cx).snapshot(cx);

        let mut selected_larger_node = false;
        let mut new_selections = old_selections
            .iter()
            .map(|selection| {
                let old_range = selection.start..selection.end;

                if let Some((node, _)) = buffer.syntax_ancestor(old_range.clone()) {
                    // manually select word at selection
                    if ["string_content", "inline"].contains(&node.kind()) {
                        let (word_range, _) = buffer.surrounding_word(old_range.start, None);
                        // ignore if word is already selected
                        if !word_range.is_empty() && old_range != word_range {
                            let (last_word_range, _) = buffer.surrounding_word(old_range.end, None);
                            // only select word if start and end point belongs to same word
                            if word_range == last_word_range {
                                selected_larger_node = true;
                                return Selection {
                                    id: selection.id,
                                    start: word_range.start,
                                    end: word_range.end,
                                    goal: SelectionGoal::None,
                                    reversed: selection.reversed,
                                };
                            }
                        }
                    }
                }

                let mut new_range = old_range.clone();
                while let Some((node, range)) = buffer.syntax_ancestor(new_range.clone()) {
                    new_range = range;
                    if !node.is_named() {
                        continue;
                    }
                    if !display_map.intersects_fold(new_range.start)
                        && !display_map.intersects_fold(new_range.end)
                    {
                        break;
                    }
                }

                selected_larger_node |= new_range != old_range;
                Selection {
                    id: selection.id,
                    start: new_range.start,
                    end: new_range.end,
                    goal: SelectionGoal::None,
                    reversed: selection.reversed,
                }
            })
            .collect::<Vec<_>>();

        if !selected_larger_node {
            return; // don't put this call in the history
        }

        // scroll based on transformation done to the last selection created by the user
        let (last_old, last_new) = old_selections
            .last()
            .zip(new_selections.last().cloned())
            .expect("old_selections isn't empty");

        let is_selection_reversed = if new_selections.len() == 1 {
            let should_be_reversed = last_old.start != last_new.start;
            new_selections.last_mut().expect("checked above").reversed = should_be_reversed;
            should_be_reversed
        } else {
            last_new.reversed
        };

        if selected_larger_node {
            self.select_syntax_node_history.disable_clearing = true;
            self.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
                s.select(new_selections.clone());
            });
            self.select_syntax_node_history.disable_clearing = false;
        }

        let start_row = last_new.start.to_display_point(&display_map).row().0;
        let end_row = last_new.end.to_display_point(&display_map).row().0;
        let selection_height = end_row - start_row + 1;
        let scroll_margin_rows = self.vertical_scroll_margin() as u32;

        let fits_on_the_screen = visible_row_count >= selection_height + scroll_margin_rows * 2;
        let scroll_behavior = if fits_on_the_screen {
            self.request_autoscroll(Autoscroll::fit(), cx);
            SelectSyntaxNodeScrollBehavior::FitSelection
        } else if is_selection_reversed {
            self.scroll_cursor_top(&ScrollCursorTop, window, cx);
            SelectSyntaxNodeScrollBehavior::CursorTop
        } else {
            self.scroll_cursor_bottom(&ScrollCursorBottom, window, cx);
            SelectSyntaxNodeScrollBehavior::CursorBottom
        };

        let old_selections: Box<[Selection<Anchor>]> = old_selections
            .iter()
            .map(|s| s.map(|offset| buffer.anchor_before(offset)))
            .collect();
        self.select_syntax_node_history.push((
            old_selections,
            scroll_behavior,
            is_selection_reversed,
        ));
    }
    pub fn select_smaller_syntax_node(
        &mut self,
        _: &SelectSmallerSyntaxNode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some((mut selections, scroll_behavior, is_selection_reversed)) =
            self.select_syntax_node_history.pop()
        {
            if let Some(selection) = selections.last_mut() {
                selection.reversed = is_selection_reversed;
            }

            let snapshot = self.buffer.read(cx).snapshot(cx);
            let selections: Vec<Selection<MultiBufferOffset>> = selections
                .iter()
                .map(|s| s.map(|anchor| anchor.to_offset(&snapshot)))
                .collect();

            self.select_syntax_node_history.disable_clearing = true;
            self.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
                s.select(selections);
            });
            self.select_syntax_node_history.disable_clearing = false;

            match scroll_behavior {
                SelectSyntaxNodeScrollBehavior::CursorTop => {
                    self.scroll_cursor_top(&ScrollCursorTop, window, cx);
                }
                SelectSyntaxNodeScrollBehavior::FitSelection => {
                    self.request_autoscroll(Autoscroll::fit(), cx);
                }
                SelectSyntaxNodeScrollBehavior::CursorBottom => {
                    self.scroll_cursor_bottom(&ScrollCursorBottom, window, cx);
                }
            }
        }
    }
    pub fn select_next_syntax_node(
        &mut self,
        _: &SelectNextSyntaxNode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let old_selections = self.selections.all_anchors(&self.display_snapshot(cx));
        if old_selections.is_empty() {
            return;
        }

        let buffer = self.buffer.read(cx).snapshot(cx);
        let mut selected_sibling = false;

        let new_selections = old_selections
            .iter()
            .map(|selection| {
                let old_range =
                    selection.start.to_offset(&buffer)..selection.end.to_offset(&buffer);
                if let Some(results) = buffer.map_excerpt_ranges(
                    old_range,
                    |buf, _excerpt_range, input_buffer_range| {
                        let Some(node) = buf.syntax_next_sibling(input_buffer_range) else {
                            return Vec::new();
                        };
                        vec![(
                            BufferOffset(node.byte_range().start)
                                ..BufferOffset(node.byte_range().end),
                            (),
                        )]
                    },
                ) && let [(new_range, _)] = results.as_slice()
                {
                    selected_sibling = true;
                    let new_range =
                        buffer.anchor_after(new_range.start)..buffer.anchor_before(new_range.end);
                    Selection {
                        id: selection.id,
                        start: new_range.start,
                        end: new_range.end,
                        goal: SelectionGoal::None,
                        reversed: selection.reversed,
                    }
                } else {
                    selection.clone()
                }
            })
            .collect::<Vec<_>>();

        if selected_sibling {
            self.change_selections(
                SelectionEffects::scroll(Autoscroll::fit()),
                window,
                cx,
                |s| {
                    s.select(new_selections);
                },
            );
        }
    }
    pub fn select_prev_syntax_node(
        &mut self,
        _: &SelectPreviousSyntaxNode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let old_selections: Arc<[_]> = self.selections.all_anchors(&self.display_snapshot(cx));

        let multibuffer_snapshot = self.buffer.read(cx).snapshot(cx);
        let mut selected_sibling = false;

        let new_selections = old_selections
            .iter()
            .map(|selection| {
                let old_range = selection.start.to_offset(&multibuffer_snapshot)
                    ..selection.end.to_offset(&multibuffer_snapshot);
                if let Some(results) = multibuffer_snapshot.map_excerpt_ranges(
                    old_range,
                    |buf, _excerpt_range, input_buffer_range| {
                        let Some(node) = buf.syntax_prev_sibling(input_buffer_range) else {
                            return Vec::new();
                        };
                        vec![(
                            BufferOffset(node.byte_range().start)
                                ..BufferOffset(node.byte_range().end),
                            (),
                        )]
                    },
                ) && let [(new_range, _)] = results.as_slice()
                {
                    selected_sibling = true;
                    let new_range = multibuffer_snapshot.anchor_after(new_range.start)
                        ..multibuffer_snapshot.anchor_before(new_range.end);
                    Selection {
                        id: selection.id,
                        start: new_range.start,
                        end: new_range.end,
                        goal: SelectionGoal::None,
                        reversed: selection.reversed,
                    }
                } else {
                    selection.clone()
                }
            })
            .collect::<Vec<_>>();

        if selected_sibling {
            self.change_selections(
                SelectionEffects::scroll(Autoscroll::fit()),
                window,
                cx,
                |s| {
                    s.select(new_selections);
                },
            );
        }
    }
    pub fn select_to_start_of_larger_syntax_node(
        &mut self,
        _: &SelectToStartOfLargerSyntaxNode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.select_to_syntax_nodes(window, cx, false);
    }
    pub fn select_to_end_of_larger_syntax_node(
        &mut self,
        _: &SelectToEndOfLargerSyntaxNode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.select_to_syntax_nodes(window, cx, true);
    }
    fn select_to_syntax_nodes(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
        move_to_end: bool,
    ) {
        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        let buffer = self.buffer.read(cx).snapshot(cx);
        let old_selections = self.selections.all::<MultiBufferOffset>(&display_map);

        let new_selections = old_selections
            .iter()
            .map(|selection| {
                let new_pos = self.find_syntax_node_boundary(
                    selection.head(),
                    move_to_end,
                    &display_map,
                    &buffer,
                );

                let mut new_selection = selection.clone();
                new_selection.set_head(new_pos, SelectionGoal::None);
                new_selection
            })
            .collect::<Vec<_>>();

        self.change_selections(Default::default(), window, cx, |s| {
            s.select(new_selections);
        });
    }
    pub fn undo_selection(
        &mut self,
        _: &UndoSelection,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(entry) = self.selection_history.undo_stack.pop_back() {
            self.selection_history.mode = SelectionHistoryMode::Undoing;
            self.with_selection_effects_deferred(window, cx, |this, window, cx| {
                this.end_selection(window, cx);
                this.change_selections(
                    SelectionEffects::scroll(Autoscroll::newest()),
                    window,
                    cx,
                    |s| s.select_anchors(entry.selections.to_vec()),
                );
            });
            self.selection_history.mode = SelectionHistoryMode::Normal;

            self.select_next_state = entry.select_next_state;
            self.select_prev_state = entry.select_prev_state;
            self.add_selections_state = entry.add_selections_state;
        }
    }
    pub fn redo_selection(
        &mut self,
        _: &RedoSelection,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(entry) = self.selection_history.redo_stack.pop_back() {
            self.selection_history.mode = SelectionHistoryMode::Redoing;
            self.with_selection_effects_deferred(window, cx, |this, window, cx| {
                this.end_selection(window, cx);
                this.change_selections(
                    SelectionEffects::scroll(Autoscroll::newest()),
                    window,
                    cx,
                    |s| s.select_anchors(entry.selections.to_vec()),
                );
            });
            self.selection_history.mode = SelectionHistoryMode::Normal;

            self.select_next_state = entry.select_next_state;
            self.select_prev_state = entry.select_prev_state;
            self.add_selections_state = entry.add_selections_state;
        }
    }
}

impl EditorSnapshot {
    pub fn remote_selections_in_range<'a>(
        &'a self,
        range: &'a Range<Anchor>,
        collaboration_hub: &dyn CollaborationHub,
        cx: &'a App,
    ) -> impl 'a + Iterator<Item = RemoteSelection> {
        let participant_names = collaboration_hub.user_names(cx);
        let participant_indices = collaboration_hub.user_participant_indices(cx);
        let collaborators_by_peer_id = collaboration_hub.collaborators(cx);
        let collaborators_by_replica_id = collaborators_by_peer_id
            .values()
            .map(|collaborator| (collaborator.replica_id, collaborator))
            .collect::<HashMap<_, _>>();
        self.buffer_snapshot()
            .selections_in_range(range, false)
            .filter_map(move |(replica_id, line_mode, cursor_shape, selection)| {
                if replica_id == ReplicaId::AGENT {
                    Some(RemoteSelection {
                        replica_id,
                        selection,
                        cursor_shape,
                        line_mode,
                        collaborator_id: CollaboratorId::Agent,
                        user_name: Some("Agent".into()),
                        color: cx.theme().players().agent(),
                    })
                } else {
                    let collaborator = collaborators_by_replica_id.get(&replica_id)?;
                    let participant_index = participant_indices.get(&collaborator.user_id).copied();
                    let user_name = participant_names.get(&collaborator.user_id).cloned();
                    Some(RemoteSelection {
                        replica_id,
                        selection,
                        cursor_shape,
                        line_mode,
                        collaborator_id: CollaboratorId::PeerId(collaborator.peer_id),
                        user_name,
                        color: if let Some(index) = participant_index {
                            cx.theme().players().color_for_participant(index.0)
                        } else {
                            cx.theme().players().absent()
                        },
                    })
                }
            })
    }
}

impl EntityInputHandler for Editor {
    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        adjusted_range: &mut Option<Range<usize>>,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<String> {
        let snapshot = self.buffer.read(cx).read(cx);
        let start = snapshot.clip_offset_utf16(
            MultiBufferOffsetUtf16(OffsetUtf16(range_utf16.start)),
            Bias::Left,
        );
        let end = snapshot.clip_offset_utf16(
            MultiBufferOffsetUtf16(OffsetUtf16(range_utf16.end)),
            Bias::Right,
        );
        if (start.0.0..end.0.0) != range_utf16 {
            adjusted_range.replace(start.0.0..end.0.0);
        }
        Some(snapshot.text_for_range(start..end).collect())
    }

    fn selected_text_range(
        &mut self,
        ignore_disabled_input: bool,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<UTF16Selection> {
        // Prevent the IME menu from appearing when holding down an alphabetic key
        // while input is disabled.
        if !ignore_disabled_input && !self.input_enabled {
            return None;
        }

        let selection = self
            .selections
            .newest::<MultiBufferOffsetUtf16>(&self.display_snapshot(cx));
        let range = selection.range();

        Some(UTF16Selection {
            range: range.start.0.0..range.end.0.0,
            reversed: selection.reversed,
        })
    }

    fn marked_text_range(&self, _: &mut Window, cx: &mut Context<Self>) -> Option<Range<usize>> {
        let snapshot = self.buffer.read(cx).read(cx);
        let range = self
            .text_highlights(HighlightKey::InputComposition, cx)?
            .1
            .first()?;
        Some(range.start.to_offset_utf16(&snapshot).0.0..range.end.to_offset_utf16(&snapshot).0.0)
    }

    fn unmark_text(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        self.clear_highlights(HighlightKey::InputComposition, cx);
        self.ime_transaction.take();
    }

    fn replace_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        text: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.input_enabled {
            cx.emit(EditorEvent::InputIgnored { text: text.into() });
            return;
        }

        self.transact(window, cx, |this, window, cx| {
            let new_selected_ranges = if let Some(range_utf16) = range_utf16 {
                if let Some(marked_ranges) = this.marked_text_ranges(cx) {
                    // During IME composition, macOS reports the replacement range
                    // relative to the first marked region (the only one visible via
                    // marked_text_range). The correct targets for replacement are the
                    // marked ranges themselves — one per cursor — so use them directly.
                    Some(marked_ranges)
                } else if range_utf16.start == range_utf16.end {
                    // An empty replacement range means "insert at cursor" with no text
                    // to replace. macOS reports the cursor position from its own
                    // (single-cursor) view of the buffer, which diverges from our actual
                    // cursor positions after multi-cursor edits have shifted offsets.
                    // Treating this as range_utf16=None lets each cursor insert in place.
                    None
                } else {
                    // Outside of IME composition (e.g. Accessibility Keyboard word
                    // completion), the range is an absolute document offset for the
                    // newest cursor. Fan it out to all cursors via
                    // selection_replacement_ranges, which applies the delta relative
                    // to the newest selection to every cursor.
                    let range_utf16 = MultiBufferOffsetUtf16(OffsetUtf16(range_utf16.start))
                        ..MultiBufferOffsetUtf16(OffsetUtf16(range_utf16.end));
                    Some(this.selection_replacement_ranges(range_utf16, cx))
                }
            } else {
                this.marked_text_ranges(cx)
            };

            let range_to_replace = new_selected_ranges.as_ref().and_then(|ranges_to_replace| {
                let newest_selection_id = this.selections.newest_anchor().id;
                this.selections
                    .all::<MultiBufferOffsetUtf16>(&this.display_snapshot(cx))
                    .iter()
                    .zip(ranges_to_replace.iter())
                    .find_map(|(selection, range)| {
                        if selection.id == newest_selection_id {
                            Some(
                                (range.start.0.0 as isize - selection.head().0.0 as isize)
                                    ..(range.end.0.0 as isize - selection.head().0.0 as isize),
                            )
                        } else {
                            None
                        }
                    })
            });

            cx.emit(EditorEvent::InputHandled {
                utf16_range_to_replace: range_to_replace,
                text: text.into(),
            });

            if let Some(new_selected_ranges) = new_selected_ranges {
                // Only backspace if at least one range covers actual text. When all
                // ranges are empty (e.g. a trailing-space insertion from Accessibility
                // Keyboard sends replacementRange=cursor..cursor), backspace would
                // incorrectly delete the character just before the cursor.
                let should_backspace = new_selected_ranges.iter().any(|r| r.start != r.end);
                this.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                    selections.select_ranges(new_selected_ranges)
                });
                if should_backspace {
                    this.backspace(&Default::default(), window, cx);
                }
            }

            this.handle_input(text, window, cx);
        });

        if let Some(transaction) = self.ime_transaction {
            self.buffer.update(cx, |buffer, cx| {
                buffer.group_until_transaction(transaction, cx);
            });
        }

        self.unmark_text(window, cx);
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        text: &str,
        new_selected_range_utf16: Option<Range<usize>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.input_enabled {
            return;
        }

        let transaction = self.transact(window, cx, |this, window, cx| {
            let ranges_to_replace = if let Some(mut marked_ranges) = this.marked_text_ranges(cx) {
                let snapshot = this.buffer.read(cx).read(cx);
                if let Some(relative_range_utf16) = range_utf16.as_ref() {
                    for marked_range in &mut marked_ranges {
                        marked_range.end = marked_range.start + relative_range_utf16.end;
                        marked_range.start += relative_range_utf16.start;
                        marked_range.start =
                            snapshot.clip_offset_utf16(marked_range.start, Bias::Left);
                        marked_range.end =
                            snapshot.clip_offset_utf16(marked_range.end, Bias::Right);
                    }
                }
                Some(marked_ranges)
            } else if let Some(range_utf16) = range_utf16 {
                let range_utf16 = MultiBufferOffsetUtf16(OffsetUtf16(range_utf16.start))
                    ..MultiBufferOffsetUtf16(OffsetUtf16(range_utf16.end));
                Some(this.selection_replacement_ranges(range_utf16, cx))
            } else {
                None
            };

            let range_to_replace = ranges_to_replace.as_ref().and_then(|ranges_to_replace| {
                let newest_selection_id = this.selections.newest_anchor().id;
                this.selections
                    .all::<MultiBufferOffsetUtf16>(&this.display_snapshot(cx))
                    .iter()
                    .zip(ranges_to_replace.iter())
                    .find_map(|(selection, range)| {
                        if selection.id == newest_selection_id {
                            Some(
                                (range.start.0.0 as isize - selection.head().0.0 as isize)
                                    ..(range.end.0.0 as isize - selection.head().0.0 as isize),
                            )
                        } else {
                            None
                        }
                    })
            });

            cx.emit(EditorEvent::InputHandled {
                utf16_range_to_replace: range_to_replace,
                text: text.into(),
            });

            if let Some(ranges) = ranges_to_replace {
                this.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
                    s.select_ranges(ranges)
                });
            }

            let marked_ranges = {
                let snapshot = this.buffer.read(cx).read(cx);
                this.selections
                    .disjoint_anchors_arc()
                    .iter()
                    .map(|selection| {
                        selection.start.bias_left(&snapshot)..selection.end.bias_right(&snapshot)
                    })
                    .collect::<Vec<_>>()
            };

            if text.is_empty() {
                this.unmark_text(window, cx);
            } else {
                this.highlight_text(
                    HighlightKey::InputComposition,
                    marked_ranges.clone(),
                    HighlightStyle {
                        underline: Some(UnderlineStyle {
                            thickness: px(1.),
                            color: None,
                            wavy: false,
                        }),
                        ..Default::default()
                    },
                    cx,
                );
            }

            // Disable auto-closing when composing text (i.e. typing a `"` on a Brazilian keyboard)
            let use_autoclose = this.use_autoclose;
            let use_auto_surround = this.use_auto_surround;
            this.set_use_autoclose(false);
            this.set_use_auto_surround(false);
            this.handle_input(text, window, cx);
            this.set_use_autoclose(use_autoclose);
            this.set_use_auto_surround(use_auto_surround);

            if let Some(new_selected_range) = new_selected_range_utf16 {
                let snapshot = this.buffer.read(cx).read(cx);
                let new_selected_ranges = marked_ranges
                    .into_iter()
                    .map(|marked_range| {
                        let insertion_start = marked_range.start.to_offset_utf16(&snapshot).0;
                        let new_start = MultiBufferOffsetUtf16(OffsetUtf16(
                            insertion_start.0 + new_selected_range.start,
                        ));
                        let new_end = MultiBufferOffsetUtf16(OffsetUtf16(
                            insertion_start.0 + new_selected_range.end,
                        ));
                        snapshot.clip_offset_utf16(new_start, Bias::Left)
                            ..snapshot.clip_offset_utf16(new_end, Bias::Right)
                    })
                    .collect::<Vec<_>>();

                drop(snapshot);
                this.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                    selections.select_ranges(new_selected_ranges)
                });
            }
        });

        self.ime_transaction = self.ime_transaction.or(transaction);
        if let Some(transaction) = self.ime_transaction {
            self.buffer.update(cx, |buffer, cx| {
                buffer.group_until_transaction(transaction, cx);
            });
        }

        if self
            .text_highlights(HighlightKey::InputComposition, cx)
            .is_none()
        {
            self.ime_transaction.take();
        }
    }

    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        element_bounds: gpui::Bounds<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<gpui::Bounds<Pixels>> {
        let text_layout_details = self.text_layout_details(window, cx);
        let CharacterDimensions {
            em_width,
            em_advance,
            line_height,
        } = self.character_dimensions(window, cx);

        let snapshot = self.snapshot(window, cx);
        let scroll_position = snapshot.scroll_position();
        let scroll_left = scroll_position.x * ScrollOffset::from(em_advance);

        let start =
            MultiBufferOffsetUtf16(OffsetUtf16(range_utf16.start)).to_display_point(&snapshot);
        let x = Pixels::from(
            ScrollOffset::from(
                snapshot.x_for_display_point(start, &text_layout_details)
                    + self.gutter_dimensions.full_width(),
            ) - scroll_left,
        );
        let y = line_height * (start.row().as_f64() - scroll_position.y) as f32;

        Some(Bounds {
            origin: element_bounds.origin + point(x, y),
            size: size(em_width, line_height),
        })
    }

    fn character_index_for_point(
        &mut self,
        point: gpui::Point<Pixels>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<usize> {
        let position_map = self.last_position_map.as_ref()?;
        if !position_map.text_hitbox.contains(&point) {
            return None;
        }
        let display_point = position_map.point_for_position(point).previous_valid;
        let anchor = position_map
            .snapshot
            .display_point_to_anchor(display_point, Bias::Left);
        let utf16_offset = anchor.to_offset_utf16(&position_map.snapshot.buffer_snapshot());
        Some(utf16_offset.0.0)
    }

    fn accepts_text_input(&self, _window: &mut Window, _cx: &mut Context<Self>) -> bool {
        self.expects_character_input
    }
}
