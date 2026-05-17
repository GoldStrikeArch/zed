use collections::HashMap;
use git::{Oid, repository::InitialGraphCommitData};
use smallvec::{SmallVec, smallvec};
use std::{ops::Range, rc::Rc, sync::Arc};
use theme::AccentColors;

pub(super) fn accent_colors_count(accents: &AccentColors) -> usize {
    accents.0.len()
}

#[derive(Copy, Clone, Debug)]
struct BranchColor(u8);

#[derive(Debug)]
enum LaneState {
    Empty,
    Active {
        child: Oid,
        parent: Oid,
        color: Option<BranchColor>,
        starting_row: usize,
        starting_col: usize,
        destination_column: Option<usize>,
        segments: SmallVec<[CommitLineSegment; 1]>,
    },
}

impl LaneState {
    fn to_commit_lines(
        &mut self,
        ending_row: usize,
        lane_column: usize,
        parent_column: usize,
        parent_color: BranchColor,
    ) -> Option<CommitLine> {
        let state = std::mem::replace(self, LaneState::Empty);

        match state {
            LaneState::Active {
                #[cfg_attr(not(test), allow(unused_variables))]
                parent,
                #[cfg_attr(not(test), allow(unused_variables))]
                child,
                color,
                starting_row,
                starting_col,
                destination_column,
                mut segments,
            } => {
                let final_destination = destination_column.unwrap_or(parent_column);
                let final_color = color.unwrap_or(parent_color);

                Some(CommitLine {
                    #[cfg(test)]
                    child,
                    #[cfg(test)]
                    parent,
                    child_column: starting_col,
                    full_interval: starting_row..ending_row,
                    color_idx: final_color.0 as usize,
                    segments: {
                        match segments.last_mut() {
                            Some(CommitLineSegment::Straight { to_row })
                                if *to_row == usize::MAX =>
                            {
                                if final_destination != lane_column {
                                    *to_row = ending_row - 1;

                                    let curved_line = CommitLineSegment::Curve {
                                        to_column: final_destination,
                                        on_row: ending_row,
                                        curve_kind: CurveKind::Checkout,
                                    };

                                    if *to_row == starting_row {
                                        let last_index = segments.len() - 1;
                                        segments[last_index] = curved_line;
                                    } else {
                                        segments.push(curved_line);
                                    }
                                } else {
                                    *to_row = ending_row;
                                }
                            }
                            Some(CommitLineSegment::Curve {
                                on_row,
                                to_column,
                                curve_kind,
                            }) if *on_row == usize::MAX => {
                                if *to_column == usize::MAX {
                                    *to_column = final_destination;
                                }
                                if matches!(curve_kind, CurveKind::Merge) {
                                    *on_row = starting_row + 1;
                                    if *on_row < ending_row {
                                        if *to_column != final_destination {
                                            segments.push(CommitLineSegment::Straight {
                                                to_row: ending_row - 1,
                                            });
                                            segments.push(CommitLineSegment::Curve {
                                                to_column: final_destination,
                                                on_row: ending_row,
                                                curve_kind: CurveKind::Checkout,
                                            });
                                        } else {
                                            segments.push(CommitLineSegment::Straight {
                                                to_row: ending_row,
                                            });
                                        }
                                    } else if *to_column != final_destination {
                                        segments.push(CommitLineSegment::Curve {
                                            to_column: final_destination,
                                            on_row: ending_row,
                                            curve_kind: CurveKind::Checkout,
                                        });
                                    }
                                } else {
                                    *on_row = ending_row;
                                    if *to_column != final_destination {
                                        segments.push(CommitLineSegment::Straight {
                                            to_row: ending_row,
                                        });
                                        segments.push(CommitLineSegment::Curve {
                                            to_column: final_destination,
                                            on_row: ending_row,
                                            curve_kind: CurveKind::Checkout,
                                        });
                                    }
                                }
                            }
                            Some(CommitLineSegment::Curve {
                                on_row, to_column, ..
                            }) => {
                                if *on_row < ending_row {
                                    if *to_column != final_destination {
                                        segments.push(CommitLineSegment::Straight {
                                            to_row: ending_row - 1,
                                        });
                                        segments.push(CommitLineSegment::Curve {
                                            to_column: final_destination,
                                            on_row: ending_row,
                                            curve_kind: CurveKind::Checkout,
                                        });
                                    } else {
                                        segments.push(CommitLineSegment::Straight {
                                            to_row: ending_row,
                                        });
                                    }
                                } else if *to_column != final_destination {
                                    segments.push(CommitLineSegment::Curve {
                                        to_column: final_destination,
                                        on_row: ending_row,
                                        curve_kind: CurveKind::Checkout,
                                    });
                                }
                            }
                            _ => {}
                        }

                        segments
                    },
                })
            }
            LaneState::Empty => None,
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            LaneState::Empty => true,
            LaneState::Active { .. } => false,
        }
    }
}

pub(super) struct CommitEntry {
    pub(super) data: Arc<InitialGraphCommitData>,
    pub(super) lane: usize,
    pub(super) color_idx: usize,
}

type ActiveLaneIdx = usize;

pub(super) enum AllCommitCount {
    NotLoaded,
    Loaded(usize),
}

#[derive(Debug)]
pub(super) enum CurveKind {
    Merge,
    Checkout,
}

#[derive(Debug)]
pub(super) enum CommitLineSegment {
    Straight {
        to_row: usize,
    },
    Curve {
        to_column: usize,
        on_row: usize,
        curve_kind: CurveKind,
    },
}

#[derive(Debug)]
pub(super) struct CommitLine {
    #[cfg(test)]
    pub(super) child: Oid,
    #[cfg(test)]
    pub(super) parent: Oid,
    pub(super) child_column: usize,
    pub(super) full_interval: Range<usize>,
    pub(super) color_idx: usize,
    pub(super) segments: SmallVec<[CommitLineSegment; 1]>,
}

impl CommitLine {
    pub(super) fn get_first_visible_segment_idx(
        &self,
        first_visible_row: usize,
    ) -> Option<(usize, usize)> {
        if first_visible_row > self.full_interval.end {
            return None;
        } else if first_visible_row <= self.full_interval.start {
            return Some((0, self.child_column));
        }

        let mut current_column = self.child_column;

        for (idx, segment) in self.segments.iter().enumerate() {
            match segment {
                CommitLineSegment::Straight { to_row } => {
                    if *to_row >= first_visible_row {
                        return Some((idx, current_column));
                    }
                }
                CommitLineSegment::Curve {
                    to_column, on_row, ..
                } => {
                    if *on_row >= first_visible_row {
                        return Some((idx, current_column));
                    }
                    current_column = *to_column;
                }
            }
        }

        None
    }
}

pub(super) struct GraphData {
    lane_states: SmallVec<[LaneState; 8]>,
    lane_colors: HashMap<ActiveLaneIdx, BranchColor>,
    parent_to_lanes: HashMap<Oid, SmallVec<[usize; 1]>>,
    next_color: BranchColor,
    accent_colors_count: usize,
    pub(super) commits: Vec<Rc<CommitEntry>>,
    pub(super) max_commit_count: AllCommitCount,
    pub(super) max_lanes: usize,
    pub(super) lines: Vec<Rc<CommitLine>>,
}

impl GraphData {
    pub(super) fn new(accent_colors_count: usize) -> Self {
        GraphData {
            lane_states: SmallVec::default(),
            lane_colors: HashMap::default(),
            parent_to_lanes: HashMap::default(),
            next_color: BranchColor(0),
            accent_colors_count,
            commits: Vec::default(),
            max_commit_count: AllCommitCount::NotLoaded,
            max_lanes: 0,
            lines: Vec::default(),
        }
    }

    pub(super) fn clear(&mut self) {
        self.lane_states.clear();
        self.lane_colors.clear();
        self.parent_to_lanes.clear();
        self.commits.clear();
        self.lines.clear();
        self.next_color = BranchColor(0);
        self.max_commit_count = AllCommitCount::NotLoaded;
        self.max_lanes = 0;
    }

    pub(super) fn add_commits(&mut self, commits: &[Arc<InitialGraphCommitData>]) {
        self.commits.reserve(commits.len());
        self.lines.reserve(commits.len() / 2);

        for commit in commits.iter() {
            let commit_row = self.commits.len();
            let commit_lane = self.next_commit_lane(commit.sha);
            let commit_color = self.get_lane_color(commit_lane);

            self.close_pending_lanes(commit.sha, commit_row, commit_lane, commit_color);
            self.open_parent_lanes(commit, commit_row, commit_lane, commit_color);

            self.max_lanes = self.max_lanes.max(self.lane_states.len());
            self.push_commit(commit.clone(), commit_lane, commit_color);
        }

        self.max_commit_count = AllCommitCount::Loaded(self.commits.len());
    }

    fn next_commit_lane(&mut self, commit: Oid) -> ActiveLaneIdx {
        self.parent_to_lanes
            .get(&commit)
            .and_then(|lanes| lanes.iter().min().copied())
            .unwrap_or_else(|| self.first_empty_lane_idx())
    }

    fn close_pending_lanes(
        &mut self,
        commit: Oid,
        commit_row: usize,
        commit_lane: ActiveLaneIdx,
        commit_color: BranchColor,
    ) {
        let Some(lanes) = self.parent_to_lanes.remove(&commit) else {
            return;
        };

        for lane_column in lanes {
            let commits = &self.commits;
            let state = &mut self.lane_states[lane_column];

            Self::avoid_overlapping_merge_curve(
                commits,
                state,
                lane_column,
                commit_row,
                commit_lane,
            );

            if let Some(commit_line) =
                state.to_commit_lines(commit_row, lane_column, commit_lane, commit_color)
            {
                self.lines.push(Rc::new(commit_line));
            }
        }
    }

    fn open_parent_lanes(
        &mut self,
        commit: &Arc<InitialGraphCommitData>,
        commit_row: usize,
        commit_lane: ActiveLaneIdx,
        commit_color: BranchColor,
    ) {
        for (parent_index, parent) in commit.parents.iter().enumerate() {
            let (parent_lane, color, segments) = if parent_index == 0 {
                (
                    commit_lane,
                    Some(commit_color),
                    smallvec![CommitLineSegment::Straight { to_row: usize::MAX }],
                )
            } else {
                (
                    self.first_empty_lane_idx(),
                    None,
                    smallvec![CommitLineSegment::Curve {
                        to_column: usize::MAX,
                        on_row: usize::MAX,
                        curve_kind: CurveKind::Merge,
                    }],
                )
            };

            self.lane_states[parent_lane] = LaneState::Active {
                parent: *parent,
                child: commit.sha,
                color,
                starting_col: commit_lane,
                starting_row: commit_row,
                destination_column: None,
                segments,
            };

            self.parent_to_lanes
                .entry(*parent)
                .or_default()
                .push(parent_lane);
        }
    }

    fn push_commit(
        &mut self,
        commit: Arc<InitialGraphCommitData>,
        commit_lane: ActiveLaneIdx,
        commit_color: BranchColor,
    ) {
        self.commits.push(Rc::new(CommitEntry {
            data: commit,
            lane: commit_lane,
            color_idx: commit_color.0 as usize,
        }));
    }

    fn avoid_overlapping_merge_curve(
        commits: &[Rc<CommitEntry>],
        state: &mut LaneState,
        lane_column: ActiveLaneIdx,
        commit_row: usize,
        commit_lane: ActiveLaneIdx,
    ) {
        let LaneState::Active {
            starting_row,
            segments,
            ..
        } = state
        else {
            return;
        };

        let Some(CommitLineSegment::Curve {
            to_column,
            curve_kind: CurveKind::Merge,
            ..
        }) = segments.first_mut()
        else {
            return;
        };

        let curve_row = *starting_row + 1;
        let would_overlap = lane_column != commit_lane
            && curve_row < commit_row
            && commits[curve_row..commit_row]
                .iter()
                .any(|commit| commit.lane == commit_lane);

        if would_overlap {
            *to_column = lane_column;
        }
    }

    fn first_empty_lane_idx(&mut self) -> ActiveLaneIdx {
        self.lane_states
            .iter()
            .position(LaneState::is_empty)
            .unwrap_or_else(|| {
                self.lane_states.push(LaneState::Empty);
                self.lane_states.len() - 1
            })
    }

    fn get_lane_color(&mut self, lane_idx: ActiveLaneIdx) -> BranchColor {
        let accent_colors_count = self.accent_colors_count;
        *self.lane_colors.entry(lane_idx).or_insert_with(|| {
            let color_idx = self.next_color;
            self.next_color = BranchColor((self.next_color.0 + 1) % accent_colors_count as u8);
            color_idx
        })
    }
}

#[cfg(test)]
pub(super) fn verify_all_invariants_for_test(
    graph: &GraphData,
    commits: &[Arc<InitialGraphCommitData>],
) -> anyhow::Result<()> {
    test_helpers::verify_all_invariants_for_test(graph, commits)
}

#[cfg(test)]
mod test_helpers {
    use super::*;
    use anyhow::{Context, Result, bail};
    use collections::{HashMap, HashSet};
    use std::sync::Arc;

    fn build_oid_to_row_map(graph: &GraphData) -> HashMap<Oid, usize> {
        graph
            .commits
            .iter()
            .enumerate()
            .map(|(idx, entry)| (entry.data.sha, idx))
            .collect()
    }

    fn verify_commit_order(
        graph: &GraphData,
        commits: &[Arc<InitialGraphCommitData>],
    ) -> Result<()> {
        if graph.commits.len() != commits.len() {
            bail!(
                "Commit count mismatch: graph has {} commits, expected {}",
                graph.commits.len(),
                commits.len()
            );
        }

        for (idx, (graph_commit, expected_commit)) in
            graph.commits.iter().zip(commits.iter()).enumerate()
        {
            if graph_commit.data.sha != expected_commit.sha {
                bail!(
                    "Commit order mismatch at index {}: graph has {:?}, expected {:?}",
                    idx,
                    graph_commit.data.sha,
                    expected_commit.sha
                );
            }
        }

        Ok(())
    }

    fn verify_line_endpoints(graph: &GraphData, oid_to_row: &HashMap<Oid, usize>) -> Result<()> {
        for line in &graph.lines {
            let child_row = *oid_to_row
                .get(&line.child)
                .context("Line references non-existent child commit")?;

            let parent_row = *oid_to_row
                .get(&line.parent)
                .context("Line references non-existent parent commit")?;

            if child_row >= parent_row {
                bail!(
                    "child_row ({}) must be < parent_row ({})",
                    child_row,
                    parent_row
                );
            }

            if line.full_interval.start != child_row {
                bail!(
                    "full_interval.start ({}) != child_row ({})",
                    line.full_interval.start,
                    child_row
                );
            }

            if line.full_interval.end != parent_row {
                bail!(
                    "full_interval.end ({}) != parent_row ({})",
                    line.full_interval.end,
                    parent_row
                );
            }

            if let Some(last_segment) = line.segments.last() {
                let segment_end_row = match last_segment {
                    CommitLineSegment::Straight { to_row } => *to_row,
                    CommitLineSegment::Curve { on_row, .. } => *on_row,
                };

                if segment_end_row != line.full_interval.end {
                    bail!(
                        "last segment ends at row {} but full_interval.end is {}",
                        segment_end_row,
                        line.full_interval.end
                    );
                }
            }
        }

        Ok(())
    }

    fn verify_column_correctness(
        graph: &GraphData,
        oid_to_row: &HashMap<Oid, usize>,
    ) -> Result<()> {
        for line in &graph.lines {
            let child_row = *oid_to_row
                .get(&line.child)
                .context("Line references non-existent child commit")?;

            let parent_row = *oid_to_row
                .get(&line.parent)
                .context("Line references non-existent parent commit")?;

            let child_lane = graph.commits[child_row].lane;
            if line.child_column != child_lane {
                bail!(
                    "child_column ({}) != child's lane ({})",
                    line.child_column,
                    child_lane
                );
            }

            let mut current_column = line.child_column;
            for segment in &line.segments {
                if let CommitLineSegment::Curve { to_column, .. } = segment {
                    current_column = *to_column;
                }
            }

            let parent_lane = graph.commits[parent_row].lane;
            if current_column != parent_lane {
                bail!(
                    "ending column ({}) != parent's lane ({})",
                    current_column,
                    parent_lane
                );
            }
        }

        Ok(())
    }

    fn verify_segment_continuity(graph: &GraphData) -> Result<()> {
        for line in &graph.lines {
            if line.segments.is_empty() {
                bail!("Line has no segments");
            }

            let mut current_row = line.full_interval.start;

            for (idx, segment) in line.segments.iter().enumerate() {
                let segment_end_row = match segment {
                    CommitLineSegment::Straight { to_row } => *to_row,
                    CommitLineSegment::Curve { on_row, .. } => *on_row,
                };

                if segment_end_row < current_row {
                    bail!(
                        "segment {} ends at row {} which is before current row {}",
                        idx,
                        segment_end_row,
                        current_row
                    );
                }

                current_row = segment_end_row;
            }
        }

        Ok(())
    }

    fn verify_line_overlaps(graph: &GraphData) -> Result<()> {
        for line in &graph.lines {
            let child_row = line.full_interval.start;

            let mut current_column = line.child_column;
            let mut current_row = child_row;

            for segment in &line.segments {
                match segment {
                    CommitLineSegment::Straight { to_row } => {
                        for row in (current_row + 1)..*to_row {
                            if row < graph.commits.len() {
                                let commit_at_row = &graph.commits[row];
                                if commit_at_row.lane == current_column {
                                    bail!(
                                        "straight segment from row {} to {} in column {} passes through commit {:?} at row {}",
                                        current_row,
                                        to_row,
                                        current_column,
                                        commit_at_row.data.sha,
                                        row
                                    );
                                }
                            }
                        }
                        current_row = *to_row;
                    }
                    CommitLineSegment::Curve {
                        to_column, on_row, ..
                    } => {
                        current_column = *to_column;
                        current_row = *on_row;
                    }
                }
            }
        }

        Ok(())
    }

    fn verify_keep_shared_parents_on_leftmost_lane(graph: &GraphData) -> Result<()> {
        let mut active_lane_parents: Vec<Option<Oid>> = Vec::new();
        let mut parent_to_lanes: HashMap<Oid, SmallVec<[usize; 1]>> = HashMap::default();

        for (row, entry) in graph.commits.iter().enumerate() {
            let pending_lanes = parent_to_lanes.remove(&entry.data.sha).unwrap_or_default();

            if pending_lanes.len() > 1
                && let Some(expected_lane) = pending_lanes.iter().copied().min()
                && entry.lane != expected_lane
            {
                bail!(
                    "commit {:?} at row {} uses lane {}, but shared parent should use leftmost pending lane {} from {:?}",
                    entry.data.sha,
                    row,
                    entry.lane,
                    expected_lane,
                    pending_lanes
                );
            }

            for lane in pending_lanes {
                let Some(active_lane_parent) = active_lane_parents.get_mut(lane) else {
                    bail!(
                        "commit {:?} at row {} was pending on missing lane {}",
                        entry.data.sha,
                        row,
                        lane
                    );
                };

                if *active_lane_parent != Some(entry.data.sha) {
                    bail!(
                        "commit {:?} at row {} was pending on lane {}, but that lane points to {:?}",
                        entry.data.sha,
                        row,
                        lane,
                        active_lane_parent
                    );
                }

                *active_lane_parent = None;
            }

            for (parent_index, parent) in entry.data.parents.iter().enumerate() {
                let lane = if parent_index == 0 {
                    entry.lane
                } else if let Some(empty_lane) =
                    active_lane_parents.iter().position(Option::is_none)
                {
                    empty_lane
                } else {
                    active_lane_parents.push(None);
                    active_lane_parents.len() - 1
                };

                if lane >= active_lane_parents.len() {
                    active_lane_parents.resize(lane + 1, None);
                }

                active_lane_parents[lane] = Some(*parent);
                parent_to_lanes.entry(*parent).or_default().push(lane);
            }
        }

        Ok(())
    }

    fn verify_coverage(graph: &GraphData) -> Result<()> {
        let mut expected_edges: HashSet<(Oid, Oid)> = HashSet::default();
        for entry in &graph.commits {
            for parent in &entry.data.parents {
                expected_edges.insert((entry.data.sha, *parent));
            }
        }

        let mut found_edges: HashSet<(Oid, Oid)> = HashSet::default();
        for line in &graph.lines {
            let edge = (line.child, line.parent);

            if !found_edges.insert(edge) {
                bail!(
                    "Duplicate line found for edge {:?} -> {:?}",
                    line.child,
                    line.parent
                );
            }

            if !expected_edges.contains(&edge) {
                bail!(
                    "Orphan line found: {:?} -> {:?} is not in the commit graph",
                    line.child,
                    line.parent
                );
            }
        }

        for (child, parent) in &expected_edges {
            if !found_edges.contains(&(*child, *parent)) {
                bail!("Missing line for edge {:?} -> {:?}", child, parent);
            }
        }

        assert_eq!(
            expected_edges.symmetric_difference(&found_edges).count(),
            0,
            "The symmetric difference should be zero"
        );

        Ok(())
    }

    fn verify_merge_line_optimality(
        graph: &GraphData,
        oid_to_row: &HashMap<Oid, usize>,
    ) -> Result<()> {
        for line in &graph.lines {
            let first_segment = line.segments.first();
            let is_merge_line = matches!(
                first_segment,
                Some(CommitLineSegment::Curve {
                    curve_kind: CurveKind::Merge,
                    ..
                })
            );

            if !is_merge_line {
                continue;
            }

            let child_row = *oid_to_row
                .get(&line.child)
                .context("Line references non-existent child commit")?;

            let parent_row = *oid_to_row
                .get(&line.parent)
                .context("Line references non-existent parent commit")?;

            let parent_lane = graph.commits[parent_row].lane;

            let Some(CommitLineSegment::Curve { to_column, .. }) = first_segment else {
                continue;
            };

            let curves_directly_to_parent = *to_column == parent_lane;

            if !curves_directly_to_parent {
                continue;
            }

            let curve_row = child_row + 1;
            let has_commits_in_path = graph.commits[curve_row..parent_row]
                .iter()
                .any(|c| c.lane == parent_lane);

            if has_commits_in_path {
                bail!(
                    "Merge line from {:?} to {:?} curves directly to parent lane {} but there are commits in that lane between rows {} and {}",
                    line.child,
                    line.parent,
                    parent_lane,
                    curve_row,
                    parent_row
                );
            }

            let curve_ends_at_parent = curve_row == parent_row;

            if curve_ends_at_parent {
                if line.segments.len() != 1 {
                    bail!(
                        "Merge line from {:?} to {:?} curves directly to parent (curve_row == parent_row), but has {} segments instead of 1 [MergeCurve]",
                        line.child,
                        line.parent,
                        line.segments.len()
                    );
                }
            } else {
                if line.segments.len() != 2 {
                    bail!(
                        "Merge line from {:?} to {:?} curves directly to parent lane without overlap, but has {} segments instead of 2 [MergeCurve, Straight]",
                        line.child,
                        line.parent,
                        line.segments.len()
                    );
                }

                let is_straight_segment = matches!(
                    line.segments.get(1),
                    Some(CommitLineSegment::Straight { .. })
                );

                if !is_straight_segment {
                    bail!(
                        "Merge line from {:?} to {:?} curves directly to parent lane without overlap, but second segment is not a Straight segment",
                        line.child,
                        line.parent
                    );
                }
            }
        }

        Ok(())
    }

    pub(super) fn verify_all_invariants_for_test(
        graph: &GraphData,
        commits: &[Arc<InitialGraphCommitData>],
    ) -> Result<()> {
        let oid_to_row = build_oid_to_row_map(graph);

        verify_commit_order(graph, commits).context("commit order")?;
        verify_line_endpoints(graph, &oid_to_row).context("line endpoints")?;
        verify_column_correctness(graph, &oid_to_row).context("column correctness")?;
        verify_segment_continuity(graph).context("segment continuity")?;
        verify_merge_line_optimality(graph, &oid_to_row).context("merge line optimality")?;
        verify_keep_shared_parents_on_leftmost_lane(graph)
            .context("keep shared parents on leftmost lane")?;
        verify_coverage(graph).context("coverage")?;
        verify_line_overlaps(graph).context("line overlaps")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_random_commit_dag;
    use rand::prelude::*;
    use std::sync::Arc;

    #[test]
    fn test_git_graph_merge_commits() {
        let mut rng = StdRng::seed_from_u64(42);

        let oid1 = Oid::random(&mut rng);
        let oid2 = Oid::random(&mut rng);
        let oid3 = Oid::random(&mut rng);
        let oid4 = Oid::random(&mut rng);

        let commits = vec![
            Arc::new(InitialGraphCommitData {
                sha: oid1,
                parents: smallvec![oid2, oid3],
                ref_names: vec!["HEAD".into()],
            }),
            Arc::new(InitialGraphCommitData {
                sha: oid2,
                parents: smallvec![oid4],
                ref_names: vec![],
            }),
            Arc::new(InitialGraphCommitData {
                sha: oid3,
                parents: smallvec![oid4],
                ref_names: vec![],
            }),
            Arc::new(InitialGraphCommitData {
                sha: oid4,
                parents: smallvec![],
                ref_names: vec![],
            }),
        ];

        let mut graph_data = GraphData::new(8);
        graph_data.add_commits(&commits);

        if let Err(error) = verify_all_invariants_for_test(&graph_data, &commits) {
            panic!("Graph invariant violation for merge commits:\n{}", error);
        }
    }

    #[test]
    fn test_git_graph_linear_commits() {
        let mut rng = StdRng::seed_from_u64(42);

        let oid1 = Oid::random(&mut rng);
        let oid2 = Oid::random(&mut rng);
        let oid3 = Oid::random(&mut rng);

        let commits = vec![
            Arc::new(InitialGraphCommitData {
                sha: oid1,
                parents: smallvec![oid2],
                ref_names: vec!["HEAD".into()],
            }),
            Arc::new(InitialGraphCommitData {
                sha: oid2,
                parents: smallvec![oid3],
                ref_names: vec![],
            }),
            Arc::new(InitialGraphCommitData {
                sha: oid3,
                parents: smallvec![],
                ref_names: vec![],
            }),
        ];

        let mut graph_data = GraphData::new(8);
        graph_data.add_commits(&commits);

        if let Err(error) = verify_all_invariants_for_test(&graph_data, &commits) {
            panic!("Graph invariant violation for linear commits:\n{}", error);
        }
    }

    #[test]
    fn test_git_graph_random_commits() {
        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);

            let adversarial = rng.random_bool(0.2);
            let num_commits = if adversarial {
                rng.random_range(10..100)
            } else {
                rng.random_range(5..50)
            };

            let commits = generate_random_commit_dag(&mut rng, num_commits, adversarial);

            assert_eq!(
                num_commits,
                commits.len(),
                "seed={}: Generate random commit dag didn't generate the correct amount of commits",
                seed
            );

            let mut graph_data = GraphData::new(8);
            graph_data.add_commits(&commits);

            if let Err(error) = verify_all_invariants_for_test(&graph_data, &commits) {
                panic!(
                    "Graph invariant violation (seed={}, adversarial={}, num_commits={}):\n{:#}",
                    seed, adversarial, num_commits, error
                );
            }
        }
    }

    #[test]
    fn test_graph_shared_parent_uses_leftmost_pending_lane() {
        let mut rng = StdRng::seed_from_u64(42);

        let merge = Oid::random(&mut rng);
        let first_parent = Oid::random(&mut rng);
        let second_parent = Oid::random(&mut rng);
        let shared_parent = Oid::random(&mut rng);

        let commits = vec![
            Arc::new(InitialGraphCommitData {
                sha: merge,
                parents: smallvec![first_parent, second_parent],
                ref_names: vec!["HEAD".into()],
            }),
            Arc::new(InitialGraphCommitData {
                sha: first_parent,
                parents: smallvec![shared_parent],
                ref_names: vec![],
            }),
            Arc::new(InitialGraphCommitData {
                sha: second_parent,
                parents: smallvec![shared_parent],
                ref_names: vec![],
            }),
            Arc::new(InitialGraphCommitData {
                sha: shared_parent,
                parents: smallvec![],
                ref_names: vec![],
            }),
        ];

        let mut graph_data = GraphData::new(8);
        graph_data.add_commits(&commits);

        verify_all_invariants_for_test(&graph_data, &commits)
            .expect("graph invariants should hold");

        let shared_parent_entry = graph_data
            .commits
            .iter()
            .find(|entry| entry.data.sha == shared_parent)
            .expect("shared parent should be present");
        assert_eq!(shared_parent_entry.lane, 0);
    }

    #[test]
    fn test_graph_merge_line_does_not_curve_through_occupied_parent_lane() {
        let mut rng = StdRng::seed_from_u64(42);

        let merge = Oid::random(&mut rng);
        let first_parent = Oid::random(&mut rng);
        let shared_parent = Oid::random(&mut rng);

        let commits = vec![
            Arc::new(InitialGraphCommitData {
                sha: merge,
                parents: smallvec![first_parent, shared_parent],
                ref_names: vec!["HEAD".into()],
            }),
            Arc::new(InitialGraphCommitData {
                sha: first_parent,
                parents: smallvec![shared_parent],
                ref_names: vec![],
            }),
            Arc::new(InitialGraphCommitData {
                sha: shared_parent,
                parents: smallvec![],
                ref_names: vec![],
            }),
        ];

        let mut graph_data = GraphData::new(8);
        graph_data.add_commits(&commits);

        verify_all_invariants_for_test(&graph_data, &commits)
            .expect("graph invariants should hold");

        let merge_line = graph_data
            .lines
            .iter()
            .find(|line| line.child == merge && line.parent == shared_parent)
            .expect("merge line to shared parent should be present");

        let first_segment = merge_line
            .segments
            .first()
            .expect("merge line should have at least one segment");
        let CommitLineSegment::Curve {
            to_column,
            curve_kind: CurveKind::Merge,
            ..
        } = first_segment
        else {
            panic!("merge line should start with a merge curve");
        };

        assert_eq!(*to_column, 1);
    }
}
