use crate::{
    ModelUsageContext,
    language_model_selector::{
        LanguageModelSelector, LanguageModelSelectorAutomaticOption, LanguageModelSelectorFilter,
        language_model_selector,
    },
    ui::ModelSelectorTooltip,
};
use fs::Fs;
use gpui::{Entity, FocusHandle, SharedString, Subscription};
use language_model::{IconOrSvg, LanguageModel};
use picker::popover_menu::PickerPopoverMenu;
use settings::{Settings as _, update_settings_file};
use std::sync::Arc;
use ui::{PopoverMenuHandle, Tooltip, prelude::*};

pub struct AgentModelSelector {
    selector: Entity<LanguageModelSelector>,
    menu_handle: PopoverMenuHandle<LanguageModelSelector>,
    button_id: &'static str,
    label_prefix: Option<&'static str>,
}

impl AgentModelSelector {
    pub(crate) fn new(
        fs: Arc<dyn Fs>,
        menu_handle: PopoverMenuHandle<LanguageModelSelector>,
        focus_handle: FocusHandle,
        model_usage_context: ModelUsageContext,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let button_id = model_usage_context.button_id();
        let label_prefix = model_usage_context.label_prefix();
        Self {
            selector: cx.new(move |cx| {
                language_model_selector(
                    {
                        let model_context = model_usage_context.clone();
                        move |cx| model_context.configured_model(cx)
                    },
                    {
                        let fs = fs.clone();
                        move |model, cx| {
                            let provider = model.provider_id().0.to_string();
                            let model_id = model.id().0.to_string();
                            match &model_usage_context {
                                ModelUsageContext::InlineAssistant => {
                                    update_settings_file(fs.clone(), cx, move |settings, _cx| {
                                        settings
                                            .agent
                                            .get_or_insert_default()
                                            .set_inline_assistant_model(provider.clone(), model_id);
                                    });
                                }
                            }
                        }
                    },
                    {
                        let fs = fs.clone();
                        move |model, should_be_favorite, cx| {
                            crate::favorite_models::toggle_in_settings(
                                model,
                                should_be_favorite,
                                fs.clone(),
                                cx,
                            );
                        }
                    },
                    model_usage_context.model_filter(),
                    None,
                    true, // Use popover styles for picker
                    focus_handle.clone(),
                    window,
                    cx,
                )
            }),
            menu_handle,
            button_id,
            label_prefix,
        }
    }

    pub fn toggle(&self, window: &mut Window, cx: &mut Context<Self>) {
        self.menu_handle.toggle(window, cx);
    }

    pub fn active_model(&self, cx: &App) -> Option<language_model::ConfiguredModel> {
        self.selector.read(cx).delegate.active_model(cx)
    }

    pub fn cycle_favorite_models(&self, window: &mut Window, cx: &mut Context<Self>) {
        self.selector.update(cx, |selector, cx| {
            selector.delegate.cycle_favorite_models(window, cx);
        });
    }
}

pub struct SubagentModelSelector {
    thread: Entity<agent::Thread>,
    selector: Entity<LanguageModelSelector>,
    menu_handle: PopoverMenuHandle<LanguageModelSelector>,
    _subscription: Subscription,
}

impl SubagentModelSelector {
    pub(crate) fn new(
        fs: Arc<dyn Fs>,
        thread: Entity<agent::Thread>,
        menu_handle: PopoverMenuHandle<LanguageModelSelector>,
        focus_handle: FocusHandle,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let subscription = cx.observe(&thread, |_, _, cx| cx.notify());

        Self {
            selector: cx.new({
                let thread = thread.clone();
                move |cx| {
                    language_model_selector(
                        {
                            let thread = thread.clone();
                            move |cx| thread.read(cx).configured_subagent_model(cx)
                        },
                        {
                            let fs = fs.clone();
                            let thread = thread.clone();
                            move |model, cx| {
                                let selection = language_model_selection(&model, cx);
                                thread.update(cx, |thread, cx| {
                                    thread
                                        .set_subagent_model_selection(Some(selection.clone()), cx);
                                });
                                update_settings_file(fs.clone(), cx, move |settings, _cx| {
                                    settings.agent.get_or_insert_default().subagent_model =
                                        Some(selection.clone());
                                });
                            }
                        },
                        {
                            let fs = fs.clone();
                            move |model, should_be_favorite, cx| {
                                crate::favorite_models::toggle_in_settings(
                                    model,
                                    should_be_favorite,
                                    fs.clone(),
                                    cx,
                                );
                            }
                        },
                        LanguageModelSelectorFilter::ToolCapable,
                        Some(LanguageModelSelectorAutomaticOption::new("Automatic", {
                            let fs = fs.clone();
                            let thread = thread.clone();
                            move |cx| {
                                thread.update(cx, |thread, cx| {
                                    thread.set_subagent_model_selection(None, cx);
                                });
                                update_settings_file(fs.clone(), cx, move |settings, _cx| {
                                    settings.agent.get_or_insert_default().subagent_model = None;
                                });
                            }
                        })),
                        true,
                        focus_handle.clone(),
                        window,
                        cx,
                    )
                }
            }),
            thread,
            menu_handle,
            _subscription: subscription,
        }
    }
}

impl Render for SubagentModelSelector {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let has_custom_selection = self.thread.read(cx).subagent_model_selection().is_some();
        let model = self.selector.read(cx).delegate.active_model(cx);
        let model_name = match (has_custom_selection, model.as_ref()) {
            (false, _) => SharedString::from("Automatic"),
            (true, Some(model)) => model.model.name().0,
            (true, None) => SharedString::from("Select a Model"),
        };
        let label = SharedString::from(format!("Subagent model: {}", model_name.as_ref()));

        let provider_icon = model.as_ref().map(|model| model.provider.icon());
        let color = if self.menu_handle.is_deployed() {
            Color::Accent
        } else {
            Color::Muted
        };
        PickerPopoverMenu::new(
            self.selector.clone(),
            Button::new("active-subagent-model", label)
                .label_size(LabelSize::Small)
                .color(color)
                .when(has_custom_selection, |this| {
                    this.when_some(provider_icon, |this, icon| {
                        this.start_icon(
                            match icon {
                                IconOrSvg::Svg(path) => Icon::from_external_svg(path),
                                IconOrSvg::Icon(name) => Icon::new(name),
                            }
                            .color(color)
                            .size(IconSize::XSmall),
                        )
                    })
                })
                .end_icon(
                    Icon::new(IconName::ChevronDown)
                        .color(color)
                        .size(IconSize::XSmall),
                ),
            Tooltip::text("Configure Subagent Model"),
            gpui::Anchor::TopRight,
            cx,
        )
        .with_handle(self.menu_handle.clone())
        .offset(gpui::Point {
            x: px(0.0),
            y: px(2.0),
        })
        .render(window, cx)
    }
}

fn language_model_selection(
    model: &Arc<dyn LanguageModel>,
    cx: &App,
) -> settings::LanguageModelSelection {
    let provider = model.provider_id().0.to_string();
    let model_id = model.id().0.to_string();
    let favorite = agent_settings::AgentSettings::get_global(cx)
        .favorite_models
        .iter()
        .find(|favorite| {
            favorite.provider.0.as_str() == provider.as_str()
                && favorite.model.as_str() == model_id.as_str()
        })
        .cloned();

    agent_settings::language_model_to_selection(model, favorite.as_ref())
}

impl Render for AgentModelSelector {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let model = self.selector.read(cx).delegate.active_model(cx);
        let model_name = model
            .as_ref()
            .map(|model| model.model.name().0)
            .unwrap_or_else(|| SharedString::from("Select a Model"));
        let label = self
            .label_prefix
            .map(|prefix| SharedString::from(format!("{prefix}: {}", model_name.as_ref())))
            .unwrap_or(model_name);

        let provider_icon = model.as_ref().map(|model| model.provider.icon());
        let color = if self.menu_handle.is_deployed() {
            Color::Accent
        } else {
            Color::Muted
        };

        let show_cycle_row = self.selector.read(cx).delegate.favorites_count() > 1;

        let tooltip = Tooltip::element({
            move |_, _cx| {
                ModelSelectorTooltip::new()
                    .show_cycle_row(show_cycle_row)
                    .into_any_element()
            }
        });

        PickerPopoverMenu::new(
            self.selector.clone(),
            Button::new(self.button_id, label)
                .label_size(LabelSize::Small)
                .color(color)
                .when_some(provider_icon, |this, icon| {
                    this.start_icon(
                        match icon {
                            IconOrSvg::Svg(path) => Icon::from_external_svg(path),
                            IconOrSvg::Icon(name) => Icon::new(name),
                        }
                        .color(color)
                        .size(IconSize::XSmall),
                    )
                })
                .end_icon(
                    Icon::new(IconName::ChevronDown)
                        .color(color)
                        .size(IconSize::XSmall),
                ),
            tooltip,
            gpui::Anchor::TopRight,
            cx,
        )
        .with_handle(self.menu_handle.clone())
        .offset(gpui::Point {
            x: px(0.0),
            y: px(2.0),
        })
        .render(window, cx)
    }
}
