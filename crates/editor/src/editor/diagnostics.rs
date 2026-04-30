use super::*;

impl Editor {
    pub(super) fn refresh_active_diagnostics(&mut self, cx: &mut Context<Editor>) {
        if !self.diagnostics_enabled() {
            return;
        }

        if let ActiveDiagnostic::Group(active_diagnostics) = &mut self.active_diagnostics {
            let buffer = self.buffer.read(cx).snapshot(cx);
            let primary_range_start = active_diagnostics.active_range.start.to_offset(&buffer);
            let primary_range_end = active_diagnostics.active_range.end.to_offset(&buffer);
            let is_valid = buffer
                .diagnostics_in_range::<MultiBufferOffset>(primary_range_start..primary_range_end)
                .any(|entry| {
                    entry.diagnostic.is_primary
                        && !entry.range.is_empty()
                        && entry.range.start == primary_range_start
                        && entry.diagnostic.message == active_diagnostics.active_message
                });

            if !is_valid {
                self.dismiss_diagnostics(cx);
            }
        }
    }

    pub fn active_diagnostic_group(&self) -> Option<&ActiveDiagnosticGroup> {
        match &self.active_diagnostics {
            ActiveDiagnostic::Group(group) => Some(group),
            _ => None,
        }
    }

    pub fn set_all_diagnostics_active(&mut self, cx: &mut Context<Self>) {
        if !self.diagnostics_enabled() {
            return;
        }
        self.dismiss_diagnostics(cx);
        self.active_diagnostics = ActiveDiagnostic::All;
    }

    pub(super) fn activate_diagnostics(
        &mut self,
        buffer_id: BufferId,
        diagnostic: DiagnosticEntryRef<'_, MultiBufferOffset>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.diagnostics_enabled() || matches!(self.active_diagnostics, ActiveDiagnostic::All) {
            return;
        }
        self.dismiss_diagnostics(cx);
        let snapshot = self.snapshot(window, cx);
        let buffer = self.buffer.read(cx).snapshot(cx);
        let Some(renderer) = GlobalDiagnosticRenderer::global(cx) else {
            return;
        };

        let diagnostic_group = buffer
            .diagnostic_group(buffer_id, diagnostic.diagnostic.group_id)
            .collect::<Vec<_>>();

        let language_registry = self
            .project()
            .map(|project| project.read(cx).languages().clone());

        let blocks = renderer.render_group(
            diagnostic_group,
            buffer_id,
            snapshot,
            cx.weak_entity(),
            language_registry,
            cx,
        );

        let blocks = self.display_map.update(cx, |display_map, cx| {
            display_map.insert_blocks(blocks, cx).into_iter().collect()
        });
        self.active_diagnostics = ActiveDiagnostic::Group(ActiveDiagnosticGroup {
            active_range: buffer.anchor_before(diagnostic.range.start)
                ..buffer.anchor_after(diagnostic.range.end),
            active_message: diagnostic.diagnostic.message.clone(),
            group_id: diagnostic.diagnostic.group_id,
            blocks,
        });
        cx.notify();
    }

    pub(super) fn dismiss_diagnostics(&mut self, cx: &mut Context<Self>) {
        if matches!(self.active_diagnostics, ActiveDiagnostic::All) {
            return;
        };

        let prev = mem::replace(&mut self.active_diagnostics, ActiveDiagnostic::None);
        if let ActiveDiagnostic::Group(group) = prev {
            self.display_map.update(cx, |display_map, cx| {
                display_map.remove_blocks(group.blocks, cx);
            });
            cx.notify();
        }
    }

    /// Disable inline diagnostics rendering for this editor.
    pub fn disable_inline_diagnostics(&mut self) {
        self.inline_diagnostics_enabled = false;
        self.inline_diagnostics_update = Task::ready(());
        self.inline_diagnostics.clear();
    }

    pub fn disable_diagnostics(&mut self, cx: &mut Context<Self>) {
        self.diagnostics_enabled = false;
        self.dismiss_diagnostics(cx);
        self.inline_diagnostics_update = Task::ready(());
        self.inline_diagnostics.clear();
    }

    pub fn disable_word_completions(&mut self) {
        self.word_completions_enabled = false;
    }

    pub fn diagnostics_enabled(&self) -> bool {
        self.diagnostics_enabled && self.lsp_data_enabled()
    }

    pub fn inline_diagnostics_enabled(&self) -> bool {
        self.inline_diagnostics_enabled && self.diagnostics_enabled()
    }

    pub fn show_inline_diagnostics(&self) -> bool {
        self.show_inline_diagnostics
    }

    pub fn toggle_inline_diagnostics(
        &mut self,
        _: &ToggleInlineDiagnostics,
        window: &mut Window,
        cx: &mut Context<Editor>,
    ) {
        self.show_inline_diagnostics = !self.show_inline_diagnostics;
        self.refresh_inline_diagnostics(false, window, cx);
    }

    pub fn set_max_diagnostics_severity(&mut self, severity: DiagnosticSeverity, cx: &mut App) {
        self.diagnostics_max_severity = severity;
        self.display_map.update(cx, |display_map, _| {
            display_map.diagnostics_max_severity = self.diagnostics_max_severity;
        });
    }

    pub fn toggle_diagnostics(
        &mut self,
        _: &ToggleDiagnostics,
        window: &mut Window,
        cx: &mut Context<Editor>,
    ) {
        if !self.diagnostics_enabled() {
            return;
        }

        let new_severity = if self.diagnostics_max_severity == DiagnosticSeverity::Off {
            EditorSettings::get_global(cx)
                .diagnostics_max_severity
                .filter(|severity| severity != &DiagnosticSeverity::Off)
                .unwrap_or(DiagnosticSeverity::Hint)
        } else {
            DiagnosticSeverity::Off
        };
        self.set_max_diagnostics_severity(new_severity, cx);
        if self.diagnostics_max_severity == DiagnosticSeverity::Off {
            self.active_diagnostics = ActiveDiagnostic::None;
            self.inline_diagnostics_update = Task::ready(());
            self.inline_diagnostics.clear();
        } else {
            self.refresh_inline_diagnostics(false, window, cx);
        }

        cx.notify();
    }

    pub fn toggle_minimap(
        &mut self,
        _: &ToggleMinimap,
        window: &mut Window,
        cx: &mut Context<Editor>,
    ) {
        if self.supports_minimap(cx) {
            self.set_minimap_visibility(self.minimap_visibility.toggle_visibility(), window, cx);
        }
    }

    pub(super) fn refresh_inline_diagnostics(
        &mut self,
        debounce: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let max_severity = ProjectSettings::get_global(cx)
            .diagnostics
            .inline
            .max_severity
            .unwrap_or(self.diagnostics_max_severity);

        if !self.inline_diagnostics_enabled()
            || !self.diagnostics_enabled()
            || !self.show_inline_diagnostics
            || max_severity == DiagnosticSeverity::Off
        {
            self.inline_diagnostics_update = Task::ready(());
            self.inline_diagnostics.clear();
            return;
        }

        let debounce_ms = ProjectSettings::get_global(cx)
            .diagnostics
            .inline
            .update_debounce_ms;
        let debounce = if debounce && debounce_ms > 0 {
            Some(Duration::from_millis(debounce_ms))
        } else {
            None
        };
        self.inline_diagnostics_update = cx.spawn_in(window, async move |editor, cx| {
            if let Some(debounce) = debounce {
                cx.background_executor().timer(debounce).await;
            }
            let Some(snapshot) = editor.upgrade().map(|editor| {
                editor.update(cx, |editor, cx| editor.buffer().read(cx).snapshot(cx))
            }) else {
                return;
            };

            let new_inline_diagnostics = cx
                .background_spawn(async move {
                    let mut inline_diagnostics = Vec::<(Anchor, InlineDiagnostic)>::new();
                    for diagnostic_entry in
                        snapshot.diagnostics_in_range(MultiBufferOffset(0)..snapshot.len())
                    {
                        let message = diagnostic_entry
                            .diagnostic
                            .message
                            .split_once('\n')
                            .map(|(line, _)| line)
                            .map(SharedString::new)
                            .unwrap_or_else(|| {
                                SharedString::new(&*diagnostic_entry.diagnostic.message)
                            });
                        let start_anchor = snapshot.anchor_before(diagnostic_entry.range.start);
                        let (Ok(i) | Err(i)) = inline_diagnostics
                            .binary_search_by(|(probe, _)| probe.cmp(&start_anchor, &snapshot));
                        inline_diagnostics.insert(
                            i,
                            (
                                start_anchor,
                                InlineDiagnostic {
                                    message,
                                    group_id: diagnostic_entry.diagnostic.group_id,
                                    start: diagnostic_entry.range.start.to_point(&snapshot),
                                    is_primary: diagnostic_entry.diagnostic.is_primary,
                                    severity: diagnostic_entry.diagnostic.severity,
                                },
                            ),
                        );
                    }
                    inline_diagnostics
                })
                .await;

            editor
                .update(cx, |editor, cx| {
                    editor.inline_diagnostics = new_inline_diagnostics;
                    cx.notify();
                })
                .ok();
        });
    }

    pub(super) fn pull_diagnostics(
        &mut self,
        buffer_id: BufferId,
        _window: &Window,
        cx: &mut Context<Self>,
    ) -> Option<()> {
        // `ActiveDiagnostic::All` is a special mode where editor's diagnostics are managed by the external view,
        // skip any LSP updates for it.

        if self.active_diagnostics == ActiveDiagnostic::All || !self.diagnostics_enabled() {
            return None;
        }
        let pull_diagnostics_settings = ProjectSettings::get_global(cx)
            .diagnostics
            .lsp_pull_diagnostics;
        if !pull_diagnostics_settings.enabled {
            return None;
        }
        let debounce = Duration::from_millis(pull_diagnostics_settings.debounce_ms);
        let project = self.project()?.downgrade();
        let buffer = self.buffer().read(cx).buffer(buffer_id)?;

        self.pull_diagnostics_task = cx.spawn(async move |_, cx| {
            cx.background_executor().timer(debounce).await;
            if let Ok(task) = project.update(cx, |project, cx| {
                project.lsp_store().update(cx, |lsp_store, cx| {
                    lsp_store.pull_diagnostics_for_buffer(buffer, cx)
                })
            }) {
                task.await.log_err();
            }
            project
                .update(cx, |project, cx| {
                    project.lsp_store().update(cx, |lsp_store, cx| {
                        lsp_store.pull_document_diagnostics_for_buffer_edit(buffer_id, cx);
                    })
                })
                .log_err();
        });

        Some(())
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

    pub fn transact(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
        update: impl FnOnce(&mut Self, &mut Window, &mut Context<Self>),
    ) -> Option<TransactionId> {
        self.with_selection_effects_deferred(window, cx, |this, window, cx| {
            this.start_transaction_at(Instant::now(), window, cx);
            update(this, window, cx);
            this.end_transaction_at(Instant::now(), cx)
        })
    }

    pub fn start_transaction_at(
        &mut self,
        now: Instant,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<TransactionId> {
        self.end_selection(window, cx);
        if let Some(tx_id) = self
            .buffer
            .update(cx, |buffer, cx| buffer.start_transaction_at(now, cx))
        {
            self.selection_history
                .insert_transaction(tx_id, self.selections.disjoint_anchors_arc());
            cx.emit(EditorEvent::TransactionBegun {
                transaction_id: tx_id,
            });
            Some(tx_id)
        } else {
            None
        }
    }

    pub fn end_transaction_at(
        &mut self,
        now: Instant,
        cx: &mut Context<Self>,
    ) -> Option<TransactionId> {
        if let Some(transaction_id) = self
            .buffer
            .update(cx, |buffer, cx| buffer.end_transaction_at(now, cx))
        {
            if let Some((_, end_selections)) =
                self.selection_history.transaction_mut(transaction_id)
            {
                *end_selections = Some(self.selections.disjoint_anchors_arc());
            } else {
                log::error!("unexpectedly ended a transaction that wasn't started by this editor");
            }

            cx.emit(EditorEvent::Edited { transaction_id });
            Some(transaction_id)
        } else {
            None
        }
    }

    pub fn modify_transaction_selection_history(
        &mut self,
        transaction_id: TransactionId,
        modify: impl FnOnce(&mut (Arc<[Selection<Anchor>]>, Option<Arc<[Selection<Anchor>]>>)),
    ) -> bool {
        self.selection_history
            .transaction_mut(transaction_id)
            .map(modify)
            .is_some()
    }
}
