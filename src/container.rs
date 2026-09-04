//! Models that are built out of other whole models.
//!
//! Ported from `NAM/container.cpp` / `container.h` (the `SlimmableContainer`
//! architecture) and `NAM/sequential.cpp` / `sequential.h` (the `Sequential`
//! architecture). Both wrap a list of complete `.nam` models, each child
//! carrying its own `version`, `architecture`, `config` and `weights`, and
//! neither container carries any weights of its own.
//!
//! They differ in how the children are combined:
//!
//! * [`ContainerModel`] runs exactly *one* child, chosen by a size control.
//!   It keeps several separately trained models and switches between them,
//!   which is how a `.nam` file offers a quality-for-CPU trade. `A2.nam` in
//!   the reference's examples is this, holding a 3-channel and an 8-channel
//!   WaveNet.
//! * [`SequentialModel`] runs *all* of them, each on the previous one's
//!   output.
//!
//! One shared subtlety: neither `Reset` goes through `DSP::Reset`, so neither
//! container prewarms *itself* on reset in the usual way. The container
//! resets only its active child and lets that child prewarm; the sequential
//! model suppresses its children's prewarms, resets them all, and then pushes
//! silence through the whole chain once. Getting that wrong changes the state
//! the first real sample meets.

use crate::buffer::Buf;
use crate::engine::{self, Engine};
use crate::error::{Error, Result};
use crate::format::{ContainerConfig, SequentialConfig};

/// The reference's `NAM_UNKNOWN_EXPECTED_SAMPLE_RATE`: what a child is
/// `reset` with when neither the host nor the file has said a rate. No
/// architecture reads the value.
const UNKNOWN_SAMPLE_RATE: f64 = -1.0;

// ---------------------------------------------------------------------------
// ContainerModel
// ---------------------------------------------------------------------------

/// One entry of a [`ContainerModel`].
#[derive(Debug)]
struct Child {
    max_value: f64,
    engine: Box<dyn Engine>,
}

/// Several whole models at different sizes, one of which is active.
///
/// `nam::container::ContainerModel`, registered under the architecture name
/// `"SlimmableContainer"`. Each entry owns the half-open control range up to
/// its `max_value`; the last entry is the fallback for everything at or above
/// the final threshold, which is why the reference requires that threshold to
/// be at least 1.0.
///
/// Two details that a naive reading misses:
///
/// * The container reports 1 input and 1 output channel unconditionally, as
///   `DSP(1, 1, expected_sample_rate)`, regardless of what its children say.
/// * Only the *active* child is ever reset. A child that has never been
///   selected has never been sized or settled, so
///   [`Engine::set_slimmable_size`] resets the newly selected one before
///   putting it in the signal path. When that happens before the host's first
///   `reset`, the buffer size in hand is still 0, and the child's own prewarm
///   falls back to the default block size, and then the host's `reset`
///   settles it a second time at the real block size. That double settling is visible
///   in the output of a recurrent child, so it is reproduced rather than
///   tidied away.
#[derive(Debug)]
pub(crate) struct ContainerModel {
    children: Vec<Child>,
    active: usize,
    max_buffer: usize,
    external_sample_rate: Option<f64>,
    expected_sample_rate: Option<f64>,
    prewarm_on_reset: bool,
}

impl ContainerModel {
    /// Build from a parsed `"SlimmableContainer"` config -
    /// `ContainerConfig::create` plus the `ContainerModel` constructor.
    ///
    /// The thresholds' ordering, their coverage of 1.0 and the children's
    /// sample-rate agreement are properties of the file and are checked where
    /// the file is parsed, in [`crate::format`]. What is checked here is the
    /// one thing only a built child can answer: its shape. The container's
    /// own `weights` array is ignored: every weight belongs to a child.
    ///
    /// `expected_sample_rate` is the container's own, or `None`; it is what
    /// a newly selected child is reset with before the host has said a rate.
    pub(crate) fn new(cfg: ContainerConfig, expected_sample_rate: Option<f64>) -> Result<Self> {
        let mut children = Vec::with_capacity(cfg.submodels.len());
        for sm in cfg.submodels {
            let engine = engine::build_file(sm.model)?;
            // The reference hard-codes the container at 1x1 and passes the
            // host's buffers straight through, so a child of any other shape
            // would read or write channels that were never provided. It does
            // not check; we do, because the failure is otherwise silent.
            if engine.in_channels() != 1 || engine.out_channels() != 1 {
                return Err(Error::Config(format!(
                    "ContainerModel: submodels must be 1-in 1-out, got {} in {} out",
                    engine.in_channels(),
                    engine.out_channels()
                )));
            }
            children.push(Child {
                max_value: sm.max_value,
                engine,
            });
        }
        if children.is_empty() {
            return Err(Error::Config(
                "ContainerModel: no submodels provided".into(),
            ));
        }
        Ok(Self {
            // Default to full size, per the reference's constructor.
            active: children.len() - 1,
            children,
            max_buffer: 0,
            external_sample_rate: None,
            expected_sample_rate,
            prewarm_on_reset: true,
        })
    }

    /// `_get_index_for_slimmable_size`: the first child whose `max_value` the
    /// control is strictly below, otherwise the last.
    fn index_for_size(&self, val: f64) -> usize {
        self.children
            .iter()
            .position(|s| val < s.max_value)
            .unwrap_or(self.children.len() - 1)
    }

    /// The rate a newly selected child is reset with: the host's, else the
    /// file's, else the reference's unknown-rate sentinel.
    fn current_sample_rate(&self) -> f64 {
        self.external_sample_rate
            .or(self.expected_sample_rate)
            .unwrap_or(UNKNOWN_SAMPLE_RATE)
    }

    fn active(&mut self) -> &mut dyn Engine {
        self.children[self.active].engine.as_mut()
    }
}

impl Engine for ContainerModel {
    /// Always 1, as `DSP(1, 1, ...)` declares.
    fn in_channels(&self) -> usize {
        1
    }

    /// Always 1.
    fn out_channels(&self) -> usize {
        1
    }

    /// The active child's prewarm length: `ContainerModel::GetPrewarmSamples`.
    fn prewarm_samples(&self) -> usize {
        self.children[self.active].engine.prewarm_samples()
    }

    fn max_buffer_size(&self) -> usize {
        self.max_buffer
    }

    /// Size the active child's buffers without settling it.
    ///
    /// `ContainerModel` does not override `SetMaxBufferSize`, so the reference
    /// would only record the number here and leave every child unsized -
    /// which matters only when a container is nested as another model's
    /// `condition_dsp`. Sizing the active child makes that work; on the
    /// ordinary path [`Engine::reset`] sizes it again immediately.
    fn set_max_buffer_size(&mut self, max_buffer: usize) {
        self.max_buffer = max_buffer;
        self.active().set_max_buffer_size(max_buffer);
    }

    /// Settle the active child: `DSP::prewarm`, and nothing else.
    ///
    /// Not `reset` with the flag forced on: `SequentialModel::reset` resets
    /// *every* child and `ContainerModel::reset` resets its active one, so
    /// routing a prewarm through `reset` would wipe state the reference's
    /// one-line `_submodels[active].model->prewarm()` leaves alone. A
    /// container of a `Sequential` diverges from the reference by 5.3e-4
    /// relative if this is done any other way.
    fn prewarm(&mut self) {
        self.active().prewarm();
    }

    fn process(&mut self, input: &[&[f32]], output: &mut [&mut [f32]], n: usize) {
        self.active().process(input, output, n);
    }

    fn process_buf(&mut self, input: &Buf, output: &mut Buf, n: usize) {
        self.active().process_buf(input, output, n);
    }

    /// Propagated to *every* child, active or not.
    fn set_prewarm_on_reset(&mut self, on: bool) {
        self.prewarm_on_reset = on;
        for s in &mut self.children {
            s.engine.set_prewarm_on_reset(on);
        }
    }

    /// The value last given to [`Engine::set_prewarm_on_reset`]; true
    /// initially.
    ///
    /// [`Engine::reset`] does not consult it here, because the reference's
    /// `Reset` is not `DSP::Reset` and leaves the decision to the child it
    /// resets, but a caller holding the container can still read it back.
    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    /// `ContainerModel::Reset`: size the container, then reset only the active
    /// child. Deliberately not `DSP::Reset`, which would prewarm the child
    /// before it had been given the new buffer size.
    fn reset(&mut self, sample_rate: f64, max_buffer: usize) {
        self.external_sample_rate = Some(sample_rate);
        self.max_buffer = max_buffer;
        self.active().reset(sample_rate, max_buffer);
    }

    /// Select the child covering `val`, `0.0` smallest and `1.0` largest.
    ///
    /// `nam::SlimmableModel::SetSlimmableSize`. Not real-time safe: it resets
    /// the newly selected child, which allocates.
    fn set_slimmable_size(&mut self, val: f64) -> Result<()> {
        let index = self.index_for_size(val);
        if index == self.active {
            return Ok(());
        }
        // The reference resets before publishing the new index, so the model
        // that enters the real-time path is already sized and settled.
        let sr = self.current_sample_rate();
        let mb = self.max_buffer;
        self.children[index].engine.reset(sr, mb);
        self.active = index;
        Ok(())
    }

    /// Internal breakpoints in `(0.0, 1.0)` where the selected child changes.
    /// The last threshold is the implied upper bound 1.0 and is not listed.
    fn slimmable_size_breakpoints(&self) -> Vec<f64> {
        self.children[..self.children.len() - 1]
            .iter()
            .map(|s| s.max_value)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// SequentialModel
// ---------------------------------------------------------------------------

/// A chain of whole models, each fed the previous one's output.
///
/// `nam::sequential::SequentialModel`, architecture name `"Sequential"`.
///
/// The reference passes `NAM_SAMPLE` (by default `double`) buffers between
/// stages while every model computes in `f32`, so each hand-off narrows and
/// widens again. That round trip is exact, which is why keeping the
/// intermediate buffers in `f32` here changes nothing.
#[derive(Debug)]
pub(crate) struct SequentialModel {
    models: Vec<Box<dyn Engine>>,
    /// One output buffer per stage, `(child out_channels, max_buffer)`.
    ///
    /// The reference allocates only the *intermediate* stages and lets the
    /// last child write straight into the host's buffer. Here the last stage
    /// has a buffer too and is copied out, which costs one pass over the
    /// block and buys a uniform `process_buf` hand-off.
    stages: Vec<Buf>,
    input_buf: Buf,
    in_channels: usize,
    out_channels: usize,
    max_buffer: usize,
    prewarm_on_reset: bool,
}

impl SequentialModel {
    /// Build from a parsed `"Sequential"` config: `SequentialConfig::create`
    /// plus `build_models`.
    ///
    /// The key checks the reference performs on each child before loading
    /// it, the rule that the top-level `weights` must be empty, and the
    /// children's sample-rate agreement are applied by [`crate::format`].
    /// What is checked here needs the built children: adjacent ones must
    /// agree on channel counts.
    pub(crate) fn new(cfg: SequentialConfig) -> Result<Self> {
        let models = cfg
            .models
            .into_iter()
            .map(engine::build_file)
            .collect::<Result<Vec<_>>>()?;
        if models.is_empty() {
            return Err(Error::Config(
                "Sequential: 'models' must be a non-empty array".into(),
            ));
        }
        for i in 1..models.len() {
            let prev_out = models[i - 1].out_channels();
            let next_in = models[i].in_channels();
            if prev_out != next_in {
                return Err(Error::Config(format!(
                    "SequentialModel: channel mismatch between submodels {} and {i} ({prev_out} \
                     output channels versus {next_in} input channels)",
                    i - 1
                )));
            }
        }

        let in_channels = models[0].in_channels();
        let out_channels = models.last().expect("checked above").out_channels();
        Ok(Self {
            stages: models.iter().map(|_| Buf::new()).collect(),
            models,
            input_buf: Buf::new(),
            in_channels,
            out_channels,
            max_buffer: 0,
            prewarm_on_reset: true,
        })
    }

    /// Run every stage, leaving the result in the last stage buffer.
    fn run_chain(&mut self, n: usize) {
        debug_assert!(
            n <= self.max_buffer,
            "block of {n} exceeds max buffer {}",
            self.max_buffer
        );
        let Self {
            models,
            stages,
            input_buf,
            ..
        } = self;
        for i in 0..models.len() {
            if i == 0 {
                models[0].process_buf(input_buf, &mut stages[0], n);
            } else {
                let (done, rest) = stages.split_at_mut(i);
                models[i].process_buf(&done[i - 1], &mut rest[0], n);
            }
        }
    }

    fn last_stage(&self) -> &Buf {
        self.stages.last().expect("at least one stage")
    }
}

impl Engine for SequentialModel {
    /// The first child's input channels.
    fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// The last child's output channels.
    fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// The sum of the children's prewarm lengths, saturating -
    /// `SequentialModel::GetPrewarmSamples` saturates at `INT_MAX` rather
    /// than wrapping.
    fn prewarm_samples(&self) -> usize {
        self.models
            .iter()
            .fold(0usize, |acc, m| acc.saturating_add(m.prewarm_samples()))
    }

    fn max_buffer_size(&self) -> usize {
        self.max_buffer
    }

    /// Size the stage buffers and every child.
    ///
    /// The reference sizes only its own stage buffers here and leaves the
    /// children to `Reset`, which leaves a nested `Sequential` with unsized
    /// children when it is used as another model's `condition_dsp`, where
    /// only `SetMaxBufferSize` is called. Sizing them here as well is
    /// idempotent with the sizing [`Engine::reset`] does a moment later.
    fn set_max_buffer_size(&mut self, max_buffer: usize) {
        self.max_buffer = max_buffer;
        self.input_buf.resize(self.in_channels, max_buffer);
        let Self { models, stages, .. } = self;
        for (stage, m) in stages.iter_mut().zip(models.iter_mut()) {
            stage.resize(m.out_channels(), max_buffer);
            m.set_max_buffer_size(max_buffer);
        }
    }

    /// Push silence through the whole chain: `DSP::prewarm`, unmodified.
    fn prewarm(&mut self) {
        let max_buffer = self.max_buffer;
        engine::prewarm_with_silence(self, max_buffer);
    }

    fn process(&mut self, input: &[&[f32]], output: &mut [&mut [f32]], n: usize) {
        self.input_buf.copy_from_channels(input, n);
        self.run_chain(n);
        self.last_stage().copy_to_channels(output, n);
    }

    fn process_buf(&mut self, input: &Buf, output: &mut Buf, n: usize) {
        debug_assert_eq!(input.rows(), self.in_channels);
        debug_assert_eq!(output.rows(), self.out_channels);
        self.input_buf.left_mut(n).copy_from_slice(input.left(n));
        self.run_chain(n);
        output
            .left_mut(n)
            .copy_from_slice(self.last_stage().left(n));
    }

    /// Propagated to every child.
    fn set_prewarm_on_reset(&mut self, on: bool) {
        self.prewarm_on_reset = on;
        for m in &mut self.models {
            m.set_prewarm_on_reset(on);
        }
    }

    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    /// `SequentialModel::Reset`.
    ///
    /// The children are reset with their own prewarms suppressed and the
    /// whole chain is settled afterwards, once. Prewarming each child in
    /// isolation would settle every stage on silence rather than on what the
    /// stage before it actually emits when fed silence, which for a model
    /// with a non-zero response to zero is a different state entirely.
    ///
    /// The reference does the suppress, reset and restore as three loops over
    /// the children. One child's reset never reads another's flag, so doing
    /// the three steps per child is the same thing without the saved list.
    fn reset(&mut self, sample_rate: f64, max_buffer: usize) {
        self.set_max_buffer_size(max_buffer);
        for m in &mut self.models {
            let was = m.prewarm_on_reset();
            m.set_prewarm_on_reset(false);
            m.reset(sample_rate, max_buffer);
            m.set_prewarm_on_reset(was);
        }
        if self.prewarm_on_reset {
            self.prewarm();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{ArchConfig, NamFile};
    use crate::loader::Model;
    use serde_json::Value;

    fn parse(v: &Value) -> NamFile {
        crate::format::parse_value(v).unwrap()
    }

    fn container(v: &Value) -> ContainerModel {
        match parse(v).config {
            ArchConfig::SlimmableContainer(c) => ContainerModel::new(c, Some(48_000.0)).unwrap(),
            other => panic!("not a container: {other:?}"),
        }
    }

    fn sequential(v: &Value) -> SequentialModel {
        match parse(v).config {
            ArchConfig::Sequential(c) => SequentialModel::new(c).unwrap(),
            other => panic!("not a sequential: {other:?}"),
        }
    }

    /// A minimal LSTM `.nam` with one hidden unit, as a child model.
    fn tiny_lstm(sample_rate: f64) -> Value {
        // 1 layer, input_size 1, hidden 1: w (4x2), b (4), then h0, c0, then
        // the head weight (1) and bias (1).
        let mut weights = vec![0.5f32, -0.25, 0.75, 0.1, -0.4, 0.2, 0.3, -0.6];
        weights.extend([0.05f32, -0.05, 0.1, 0.0]);
        weights.extend([0.0f32, 0.0]);
        weights.extend([1.5f32, -0.2]);
        serde_json::json!({
            "version": "0.5.4",
            "architecture": "LSTM",
            "config": { "input_size": 1, "hidden_size": 1, "num_layers": 1 },
            "weights": weights,
            "sample_rate": sample_rate,
        })
    }

    fn container_json(max_values: &[f64]) -> Value {
        let subs: Vec<Value> = max_values
            .iter()
            .map(|v| serde_json::json!({ "max_value": v, "model": tiny_lstm(48000.0) }))
            .collect();
        serde_json::json!({
            "version": "0.7.0",
            "architecture": "SlimmableContainer",
            "config": { "submodels": subs },
            "weights": [],
            "sample_rate": 48000.0,
        })
    }

    fn sequential_json(children: Vec<Value>) -> Value {
        serde_json::json!({
            "version": "0.7.0",
            "architecture": "Sequential",
            "config": { "models": children },
            "weights": [],
            "sample_rate": 48000.0,
        })
    }

    fn run(engine: &mut dyn Engine, input: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; input.len()];
        engine.process(&[input], &mut [&mut out], input.len());
        out
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn each_submodel_owns_a_half_open_range_and_the_last_catches_the_rest() {
        let c = container(&container_json(&[0.33, 0.66, 1.0]));
        assert_eq!(c.index_for_size(0.0), 0);
        assert_eq!(c.index_for_size(0.32), 0);
        // The boundary belongs to the *next* model: the test is `val < max`.
        assert_eq!(c.index_for_size(0.33), 1);
        assert_eq!(c.index_for_size(0.65), 1);
        assert_eq!(c.index_for_size(0.66), 2);
        assert_eq!(c.index_for_size(1.0), 2);
        // Outside [0, 1] the ends simply saturate.
        assert_eq!(c.index_for_size(-1.0), 0);
        assert_eq!(c.index_for_size(7.0), 2);
        // A container defaults to its largest model.
        assert_eq!(c.active, 2);
        assert_eq!(c.slimmable_size_breakpoints(), vec![0.33, 0.66]);
    }

    /// Selecting a child resets it *twice*, and the extra settling is part of
    /// the output.
    ///
    /// `ContainerModel::SetSlimmableSize` resets the newly selected child with
    /// the buffer size the container happens to hold, which is 0 when the
    /// host has not called `Reset` yet, and the host's `Reset` then settles
    /// it again at the real block size. For a recurrent child that is twice as much
    /// silence and a different state. The recorded case
    /// `slimmable_container__slim0_0` is the oracle for this; the test pins
    /// the mechanism so a "tidying" refactor cannot quietly drop one of the
    /// two resets.
    #[test]
    fn selecting_a_child_settles_it_once_on_selection_and_once_on_reset() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/models/slimmable_container.nam");
        let raw: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.21).sin() * 0.5).collect();

        let mut c = container(&raw);
        c.set_slimmable_size(0.0).unwrap();
        assert_eq!(c.active, 0);
        c.reset(48_000.0, 64);

        // The same child, driven through the same two resets by hand.
        let mut child = Model::from_value(&raw["config"]["submodels"][0]["model"]).unwrap();
        child.reset(48_000.0, 0);
        child.reset(48_000.0, 64);

        let mut want = vec![0.0f32; 64];
        child.process(&[&input], &mut [&mut want], 64);
        assert_eq!(bits(&run(&mut c, &input)), bits(&want));

        // A child that was never selected is never touched: staying at the
        // default index must not reset anything.
        let mut c = container(&raw);
        c.set_slimmable_size(1.0).unwrap();
        assert_eq!(c.active, 2, "1.0 is at or above the last threshold");
        c.reset(48_000.0, 64);

        let mut child = Model::from_value(&raw["config"]["submodels"][2]["model"]).unwrap();
        child.reset(48_000.0, 64);
        let mut want = vec![0.0f32; 64];
        child.process(&[&input], &mut [&mut want], 64);
        assert_eq!(bits(&run(&mut c, &input)), bits(&want));
    }

    #[test]
    fn a_sequential_of_one_child_matches_that_child_bit_for_bit() {
        let child = tiny_lstm(48000.0);
        let mut a = sequential(&sequential_json(vec![child.clone()]));
        let mut b = Model::from_value(&child).unwrap();
        a.set_prewarm_on_reset(false);
        b.set_prewarm_on_reset(false);
        a.reset(48000.0, 16);
        b.reset(48000.0, 16);

        let input: Vec<f32> = (0..16).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut ob = vec![0.0f32; 16];
        b.process(&[&input], &mut [&mut ob], 16);
        assert_eq!(bits(&run(&mut a, &input)), bits(&ob));
    }

    #[test]
    fn a_sequential_chains_children_in_order() {
        // Two copies of the same child: running the chain must equal running
        // the child twice by hand, with the intermediate signal in between.
        let child = tiny_lstm(48000.0);
        let mut chain = sequential(&sequential_json(vec![child.clone(), child.clone()]));
        let mut first = Model::from_value(&child).unwrap();
        let mut second = Model::from_value(&child).unwrap();
        chain.set_prewarm_on_reset(false);
        first.set_prewarm_on_reset(false);
        second.set_prewarm_on_reset(false);
        chain.reset(48000.0, 8);
        first.reset(48000.0, 8);
        second.reset(48000.0, 8);

        let input: Vec<f32> = (0..8).map(|i| (i as f32 * 0.9).cos()).collect();
        let got = run(&mut chain, &input);

        let mut mid = vec![0.0f32; 8];
        let mut want = vec![0.0f32; 8];
        first.process(&[&input], &mut [&mut mid], 8);
        {
            let m: &[f32] = &mid;
            second.process(&[m], &mut [&mut want], 8);
        }
        assert_eq!(bits(&got), bits(&want));
    }

    #[test]
    fn a_sequential_prewarm_length_is_the_sum_of_its_children() {
        let child = tiny_lstm(48000.0);
        let one = Model::from_value(&child).unwrap();
        let chain = sequential(&sequential_json(vec![child.clone(), child.clone(), child]));
        assert_eq!(chain.prewarm_samples(), 3 * one.prewarm_samples());
    }

    #[test]
    fn a_sequential_refuses_children_whose_channels_do_not_meet() {
        let stereo_out = serde_json::json!({
            "version": "0.5.4",
            "architecture": "LSTM",
            "config": { "input_size": 1, "hidden_size": 1, "num_layers": 0, "out_channels": 2 },
            "weights": [1.0, 0.5, 0.0, 0.0],
            "sample_rate": 48000.0,
        });
        let seq = sequential_json(vec![stereo_out, tiny_lstm(48000.0)]);
        let ArchConfig::Sequential(c) = parse(&seq).config else {
            panic!()
        };
        let err = SequentialModel::new(c).unwrap_err().to_string();
        assert!(err.contains("channel mismatch"), "{err}");
    }
}
