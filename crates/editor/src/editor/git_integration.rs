use super::*;

impl Editor {
    pub fn working_directory(&self, cx: &App) -> Option<PathBuf> {
        if let Some(buffer) = self.buffer().read(cx).as_singleton() {
            if let Some(file) = buffer.read(cx).file().and_then(|f| f.as_local())
                && let Some(dir) = file.abs_path(cx).parent()
            {
                return Some(dir.to_owned());
            }
        }

        None
    }

    pub(super) fn target_file<'a>(&self, cx: &'a App) -> Option<&'a dyn language::LocalFile> {
        self.active_buffer(cx)?
            .read(cx)
            .file()
            .and_then(|f| f.as_local())
    }

    pub fn target_file_abs_path(&self, cx: &mut Context<Self>) -> Option<PathBuf> {
        self.active_buffer(cx).and_then(|buffer| {
            let buffer = buffer.read(cx);
            if let Some(project_path) = buffer.project_path(cx) {
                let project = self.project()?.read(cx);
                project.absolute_path(&project_path, cx)
            } else {
                buffer
                    .file()
                    .and_then(|file| file.as_local().map(|file| file.abs_path(cx)))
            }
        })
    }

    pub fn reveal_in_finder(
        &mut self,
        _: &RevealInFileManager,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(path) = self.target_file_abs_path(cx) {
            if let Some(project) = self.project() {
                project.update(cx, |project, cx| project.reveal_path(&path, cx));
            } else {
                cx.reveal_path(&path);
            }
        }
    }

    pub fn copy_path(
        &mut self,
        _: &zed_actions::workspace::CopyPath,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(path) = self.target_file_abs_path(cx)
            && let Some(path) = path.to_str()
        {
            cx.write_to_clipboard(ClipboardItem::new_string(path.to_string()));
        } else {
            cx.propagate();
        }
    }

    pub fn copy_relative_path(
        &mut self,
        _: &zed_actions::workspace::CopyRelativePath,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(path) = self.active_buffer(cx).and_then(|buffer| {
            let project = self.project()?.read(cx);
            let path = buffer.read(cx).file()?.path();
            let path = path.display(project.path_style(cx));
            Some(path)
        }) {
            cx.write_to_clipboard(ClipboardItem::new_string(path.to_string()));
        } else {
            cx.propagate();
        }
    }

    /// Returns the project path for the editor's buffer, if any buffer is
    /// opened in the editor.
    pub fn project_path(&self, cx: &App) -> Option<ProjectPath> {
        if let Some(buffer) = self.buffer.read(cx).as_singleton() {
            buffer.read(cx).project_path(cx)
        } else {
            None
        }
    }

    // Returns true if the editor handled a go-to-line request
    pub fn go_to_active_debug_line(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        maybe!({
            let breakpoint_store = self.breakpoint_store.as_ref()?;

            let (active_stack_frame, debug_line_pane_id) = {
                let store = breakpoint_store.read(cx);
                let active_stack_frame = store.active_position().cloned();
                let debug_line_pane_id = store.active_debug_line_pane_id();
                (active_stack_frame, debug_line_pane_id)
            };

            let Some(active_stack_frame) = active_stack_frame else {
                self.clear_row_highlights::<ActiveDebugLine>();
                return None;
            };

            if let Some(debug_line_pane_id) = debug_line_pane_id {
                if let Some(workspace) = self
                    .workspace
                    .as_ref()
                    .and_then(|(workspace, _)| workspace.upgrade())
                {
                    let editor_pane_id = workspace
                        .read(cx)
                        .pane_for_item_id(cx.entity_id())
                        .map(|pane| pane.entity_id());

                    if editor_pane_id.is_some_and(|id| id != debug_line_pane_id) {
                        self.clear_row_highlights::<ActiveDebugLine>();
                        return None;
                    }
                }
            }

            let position = active_stack_frame.position;

            let snapshot = self.buffer.read(cx).snapshot(cx);
            let multibuffer_anchor = snapshot.anchor_in_excerpt(position)?;

            self.clear_row_highlights::<ActiveDebugLine>();

            self.go_to_line::<ActiveDebugLine>(
                multibuffer_anchor,
                Some(cx.theme().colors().editor_debugger_active_line_background),
                window,
                cx,
            );

            cx.notify();

            Some(())
        })
        .is_some()
    }

    pub fn copy_file_name_without_extension(
        &mut self,
        _: &CopyFileNameWithoutExtension,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(file_stem) = self.active_buffer(cx).and_then(|buffer| {
            let file = buffer.read(cx).file()?;
            file.path().file_stem()
        }) {
            cx.write_to_clipboard(ClipboardItem::new_string(file_stem.to_string()));
        }
    }

    pub fn copy_file_name(&mut self, _: &CopyFileName, _: &mut Window, cx: &mut Context<Self>) {
        if let Some(file_name) = self.active_buffer(cx).and_then(|buffer| {
            let file = buffer.read(cx).file()?;
            Some(file.file_name(cx))
        }) {
            cx.write_to_clipboard(ClipboardItem::new_string(file_name.to_string()));
        }
    }

    pub fn toggle_git_blame(
        &mut self,
        _: &::git::Blame,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.show_git_blame_gutter = !self.show_git_blame_gutter;

        if self.show_git_blame_gutter && !self.has_blame_entries(cx) {
            self.start_git_blame(true, window, cx);
        }

        cx.notify();
    }

    pub fn toggle_git_blame_inline(
        &mut self,
        _: &ToggleGitBlameInline,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.toggle_git_blame_inline_internal(true, window, cx);
        cx.notify();
    }

    pub fn open_git_blame_commit(
        &mut self,
        _: &OpenGitBlameCommit,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.open_git_blame_commit_internal(window, cx);
    }

    pub(super) fn open_git_blame_commit_internal(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<()> {
        let blame = self.blame.as_ref()?;
        let snapshot = self.snapshot(window, cx);
        let cursor = self
            .selections
            .newest::<Point>(&snapshot.display_snapshot)
            .head();
        let (buffer, point) = snapshot.buffer_snapshot().point_to_buffer_point(cursor)?;
        let (_, blame_entry) = blame
            .update(cx, |blame, cx| {
                blame
                    .blame_for_rows(
                        &[RowInfo {
                            buffer_id: Some(buffer.remote_id()),
                            buffer_row: Some(point.row),
                            ..Default::default()
                        }],
                        cx,
                    )
                    .next()
            })
            .flatten()?;
        let renderer = cx.global::<GlobalBlameRenderer>().0.clone();
        let repo = blame.read(cx).repository(cx, buffer.remote_id())?;
        let workspace = self.workspace()?.downgrade();
        renderer.open_blame_commit(blame_entry, repo, workspace, window, cx);
        None
    }

    pub fn git_blame_inline_enabled(&self) -> bool {
        self.git_blame_inline_enabled
    }

    pub fn toggle_selection_menu(
        &mut self,
        _: &ToggleSelectionMenu,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.show_selection_menu = self
            .show_selection_menu
            .map(|show_selections_menu| !show_selections_menu)
            .or_else(|| Some(!EditorSettings::get_global(cx).toolbar.selections_menu));

        cx.notify();
    }

    pub fn selection_menu_enabled(&self, cx: &App) -> bool {
        self.show_selection_menu
            .unwrap_or_else(|| EditorSettings::get_global(cx).toolbar.selections_menu)
    }

    pub(super) fn start_git_blame(
        &mut self,
        user_triggered: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(project) = self.project() {
            if let Some(buffer) = self.buffer().read(cx).as_singleton()
                && buffer.read(cx).file().is_none()
            {
                return;
            }

            let focused = self.focus_handle(cx).contains_focused(window, cx);

            let project = project.clone();
            let blame = cx
                .new(|cx| GitBlame::new(self.buffer.clone(), project, user_triggered, focused, cx));
            self.blame_subscription =
                Some(cx.observe_in(&blame, window, |_, _, _, cx| cx.notify()));
            self.blame = Some(blame);
        }
    }

    pub(super) fn toggle_git_blame_inline_internal(
        &mut self,
        user_triggered: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.git_blame_inline_enabled {
            self.git_blame_inline_enabled = false;
            self.show_git_blame_inline = false;
            self.show_git_blame_inline_delay_task.take();
        } else {
            self.git_blame_inline_enabled = true;
            self.start_git_blame_inline(user_triggered, window, cx);
        }

        cx.notify();
    }

    pub(super) fn start_git_blame_inline(
        &mut self,
        user_triggered: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.start_git_blame(user_triggered, window, cx);

        if ProjectSettings::get_global(cx)
            .git
            .inline_blame_delay()
            .is_some()
        {
            self.start_inline_blame_timer(window, cx);
        } else {
            self.show_git_blame_inline = true
        }
    }

    pub fn blame(&self) -> Option<&Entity<GitBlame>> {
        self.blame.as_ref()
    }

    pub fn show_git_blame_gutter(&self) -> bool {
        self.show_git_blame_gutter
    }

    pub fn render_git_blame_gutter(&self, cx: &App) -> bool {
        !self.mode().is_minimap() && self.show_git_blame_gutter && self.has_blame_entries(cx)
    }

    pub fn render_git_blame_inline(&self, window: &Window, cx: &App) -> bool {
        self.show_git_blame_inline
            && (self.focus_handle.is_focused(window) || self.inline_blame_popover.is_some())
            && !self.newest_selection_head_on_empty_line(cx)
            && self.has_blame_entries(cx)
    }

    pub(super) fn has_blame_entries(&self, cx: &App) -> bool {
        self.blame()
            .is_some_and(|blame| blame.read(cx).has_generated_entries())
    }

    pub(super) fn newest_selection_head_on_empty_line(&self, cx: &App) -> bool {
        let cursor_anchor = self.selections.newest_anchor().head();

        let snapshot = self.buffer.read(cx).snapshot(cx);
        let buffer_row = MultiBufferRow(cursor_anchor.to_point(&snapshot).row);

        snapshot.line_len(buffer_row) == 0
    }

    pub(super) fn get_permalink_to_line(&self, cx: &mut Context<Self>) -> Task<Result<url::Url>> {
        let buffer_and_selection = maybe!({
            let selection = self.selections.newest::<Point>(&self.display_snapshot(cx));
            let selection_range = selection.range();

            let multi_buffer = self.buffer().read(cx);
            let multi_buffer_snapshot = multi_buffer.snapshot(cx);
            let buffer_ranges = multi_buffer_snapshot
                .range_to_buffer_ranges(selection_range.start..selection_range.end);

            let (buffer_snapshot, range, _) = if selection.reversed {
                buffer_ranges.first()
            } else {
                buffer_ranges.last()
            }?;

            let buffer_range = range.to_point(buffer_snapshot);
            let buffer = multi_buffer.buffer(buffer_snapshot.remote_id()).unwrap();

            let Some(buffer_diff) = multi_buffer.diff_for(buffer_snapshot.remote_id()) else {
                return Some((buffer, buffer_range.start.row..buffer_range.end.row));
            };

            let buffer_diff_snapshot = buffer_diff.read(cx).snapshot(cx);
            let start = buffer_diff_snapshot
                .buffer_point_to_base_text_point(buffer_range.start, &buffer_snapshot);
            let end = buffer_diff_snapshot
                .buffer_point_to_base_text_point(buffer_range.end, &buffer_snapshot);

            Some((buffer, start.row..end.row))
        });

        let Some((buffer, selection)) = buffer_and_selection else {
            return Task::ready(Err(anyhow!("failed to determine buffer and selection")));
        };

        let Some(project) = self.project() else {
            return Task::ready(Err(anyhow!("editor does not have project")));
        };

        project.update(cx, |project, cx| {
            project.get_permalink_to_line(&buffer, selection, cx)
        })
    }

    pub fn copy_permalink_to_line(
        &mut self,
        _: &CopyPermalinkToLine,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let permalink_task = self.get_permalink_to_line(cx);
        let workspace = self.workspace();

        cx.spawn_in(window, async move |_, cx| match permalink_task.await {
            Ok(permalink) => {
                cx.update(|_, cx| {
                    cx.write_to_clipboard(ClipboardItem::new_string(permalink.to_string()));
                })
                .ok();
            }
            Err(err) => {
                let message = format!("Failed to copy permalink: {err}");

                anyhow::Result::<()>::Err(err).log_err();

                if let Some(workspace) = workspace {
                    workspace
                        .update_in(cx, |workspace, _, cx| {
                            struct CopyPermalinkToLine;

                            workspace.show_toast(
                                Toast::new(
                                    NotificationId::unique::<CopyPermalinkToLine>(),
                                    message,
                                ),
                                cx,
                            )
                        })
                        .ok();
                }
            }
        })
        .detach();
    }

    pub fn copy_file_location(
        &mut self,
        _: &CopyFileLocation,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let selection = self.selections.newest::<Point>(&self.display_snapshot(cx));

        let start_line = selection.start.row + 1;
        let end_line = selection.end.row + 1;

        let end_line = if selection.end.column == 0 && end_line > start_line {
            end_line - 1
        } else {
            end_line
        };

        if let Some(file_location) = self.active_buffer(cx).and_then(|buffer| {
            let project = self.project()?.read(cx);
            let file = buffer.read(cx).file()?;
            let path = file.path().display(project.path_style(cx));

            let location = if start_line == end_line {
                format!("{path}:{start_line}")
            } else {
                format!("{path}:{start_line}-{end_line}")
            };
            Some(location)
        }) {
            cx.write_to_clipboard(ClipboardItem::new_string(file_location));
        }
    }

    pub fn open_permalink_to_line(
        &mut self,
        _: &OpenPermalinkToLine,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let permalink_task = self.get_permalink_to_line(cx);
        let workspace = self.workspace();

        cx.spawn_in(window, async move |_, cx| match permalink_task.await {
            Ok(permalink) => {
                cx.update(|_, cx| {
                    cx.open_url(permalink.as_ref());
                })
                .ok();
            }
            Err(err) => {
                let message = format!("Failed to open permalink: {err}");

                anyhow::Result::<()>::Err(err).log_err();

                if let Some(workspace) = workspace {
                    workspace.update(cx, |workspace, cx| {
                        struct OpenPermalinkToLine;

                        workspace.show_toast(
                            Toast::new(NotificationId::unique::<OpenPermalinkToLine>(), message),
                            cx,
                        )
                    });
                }
            }
        })
        .detach();
    }

    pub fn insert_uuid_v4(
        &mut self,
        _: &InsertUuidV4,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.insert_uuid(UuidVersion::V4, window, cx);
    }

    pub fn insert_uuid_v7(
        &mut self,
        _: &InsertUuidV7,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.insert_uuid(UuidVersion::V7, window, cx);
    }

    pub(super) fn insert_uuid(
        &mut self,
        version: UuidVersion,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.read_only(cx) {
            return;
        }
        self.hide_mouse_cursor(HideMouseCursorOrigin::TypingAction, cx);
        self.transact(window, cx, |this, window, cx| {
            let edits = this
                .selections
                .all::<Point>(&this.display_snapshot(cx))
                .into_iter()
                .map(|selection| {
                    let uuid = match version {
                        UuidVersion::V4 => uuid::Uuid::new_v4(),
                        UuidVersion::V7 => uuid::Uuid::now_v7(),
                    };

                    (selection.range(), uuid.to_string())
                });
            this.edit(edits, cx);
            this.refresh_edit_prediction(true, false, window, cx);
        });
    }

    pub fn open_selections_in_multibuffer(
        &mut self,
        _: &OpenSelectionsInMultibuffer,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let multibuffer = self.buffer.read(cx);

        let Some(buffer) = multibuffer.as_singleton() else {
            return;
        };
        let buffer_snapshot = buffer.read(cx).snapshot();

        let Some(workspace) = self.workspace() else {
            return;
        };

        let title = multibuffer.title(cx).to_string();

        let locations = self
            .selections
            .all_anchors(&self.display_snapshot(cx))
            .iter()
            .map(|selection| {
                (
                    buffer.clone(),
                    (selection.start.text_anchor_in(&buffer_snapshot)
                        ..selection.end.text_anchor_in(&buffer_snapshot))
                        .to_point(buffer.read(cx)),
                )
            })
            .into_group_map();

        cx.spawn_in(window, async move |_, cx| {
            workspace.update_in(cx, |workspace, window, cx| {
                Self::open_locations_in_multibuffer(
                    workspace,
                    locations,
                    format!("Selections for '{title}'"),
                    false,
                    false,
                    MultibufferSelectionMode::All,
                    window,
                    cx,
                );
            })
        })
        .detach();
    }
}
