//! The interface every runnable architecture answers to, and the factory
//! that turns a parsed file into one.
//!
//! The reference has a single base class, `nam::DSP` (`NAM/dsp.h`), from
//! which every architecture derives. [`Engine`] is that base class: the
//! methods a host calls, the two `nam::SlimmableModel` hooks that only some
//! architectures answer, and `DSP::Reset` as a trait default so that the
//! architectures which do not override it upstream do not spell it out here.
//!
//! `DSP::prewarm` is *not* a default, because every architecture does
//! something around it: caching a steady state, delegating to a child,
//! settling a whole chain at once. What they share is the loop in the middle,
//! `prewarm_with_silence`.
//!
//! `build` is `nam::get_dsp` minus the file handling: it maps a parsed
//! [`ArchConfig`] onto an engine. A `.nam` document nests, since a WaveNet
//! may carry a `condition_dsp` and the container architectures hold whole
//! child models, and each nested document comes back through `build_file`. That
//! is the one cycle in the crate's module graph, and the reason the factory
//! lives beside the trait rather than in [`crate::loader`].

use crate::buffer::Buf;
use crate::container::{ContainerModel, SequentialModel};
use crate::error::{Error, Result};
use crate::format::{ArchConfig, NamFile};
use crate::lstm::Lstm;
use crate::wavenet::WaveNet;

/// One runnable model.
///
/// `Send + Sync` is part of the contract, not incidental. A host loads on one
/// thread and processes on another, and `std::thread::spawn`, cpal's
/// `build_output_stream`, and every plugin framework's processor struct all
/// require it. Without the bound `Box<dyn Engine>` is neither, and a loaded
/// model could not legally reach the callback it exists to run in.
///
/// # Lifecycle
///
/// 1. Build, through [`crate::loader::Model`]. Allocates.
/// 2. [`Engine::reset`] with the host's sample rate and largest block.
///    Allocates, then settles the model on silence.
/// 3. [`Engine::process`] on the audio thread, any number of times, with any
///    block of up to that many frames. Allocation-free.
///
/// Step 2 may be repeated whenever the block size changes. Nothing else is
/// safe to call from the audio thread.
///
/// # Contract of `process`
///
/// `input` holds at least [`Engine::in_channels`] slices and `output` at
/// least [`Engine::out_channels`], each at least `n` frames long; surplus
/// channels are ignored. `n` is at most [`Engine::max_buffer_size`], and
/// [`Engine::reset`] or [`Engine::set_max_buffer_size`] has run. Violating
/// any of these panics: there is no error path out of an audio callback, so
/// a host that can be handed a larger block than it asked for splits it
/// before calling. `n == 0` is allowed and does nothing.
pub trait Engine: std::fmt::Debug + Send + Sync {
    /// Audio channels the model reads. At least 1.
    fn in_channels(&self) -> usize;

    /// Audio channels the model writes. At least 1.
    fn out_channels(&self) -> usize;

    /// Frames of silence the model needs to settle: `DSP::GetPrewarmSamples`.
    ///
    /// Zero for an architecture with no state to settle.
    fn prewarm_samples(&self) -> usize;

    /// The largest block [`Engine::process`] accepts: the value last given
    /// to [`Engine::set_max_buffer_size`] or [`Engine::reset`], or
    /// [`crate::DEFAULT_MAX_BUFFER_SIZE`] after a bare [`Engine::prewarm`].
    /// Zero until one of those has run.
    fn max_buffer_size(&self) -> usize;

    /// Size every buffer for blocks of up to `max_buffer` frames.
    ///
    /// Allocating. Call it before the audio thread starts, not from it. The
    /// buffers are also cleared, so the model is back in its unsettled state
    /// afterwards; [`Engine::reset`] is this followed by a prewarm.
    fn set_max_buffer_size(&mut self, max_buffer: usize);

    /// Settle the model on silence.
    ///
    /// Allocating, for the same reason. A model that has never been sized is
    /// sized to [`crate::DEFAULT_MAX_BUFFER_SIZE`] first.
    fn prewarm(&mut self);

    /// Process `n` frames. `input[ch][frame]`, `output[ch][frame]`.
    ///
    /// Allocation-free once sized; see the trait docs for what the caller
    /// guarantees.
    fn process(&mut self, input: &[&[f32]], output: &mut [&mut [f32]], n: usize);

    /// Process `n` frames from one column-major buffer into another.
    ///
    /// `input` has exactly [`Engine::in_channels`] rows and `output` exactly
    /// [`Engine::out_channels`]; both have at least `n` columns. Otherwise as
    /// [`Engine::process`]. This is how a model nested inside another one is
    /// driven: buffers rather than a slice of channel slices, so that the
    /// caller needs no per-block `Vec` of references, which would allocate
    /// on the audio thread.
    fn process_buf(&mut self, input: &Buf, output: &mut Buf, n: usize);

    /// Whether [`Engine::reset`] settles the model. True initially.
    ///
    /// The architectures that hold other models propagate it to them, which
    /// is what `WaveNet::SetPrewarmOnReset` and its siblings do upstream.
    fn set_prewarm_on_reset(&mut self, on: bool);

    /// The flag [`Engine::set_prewarm_on_reset`] last stored.
    fn prewarm_on_reset(&self) -> bool;

    /// Size the model for `max_buffer` frames and, if
    /// [`Engine::prewarm_on_reset`] is set, settle it: `DSP::Reset`.
    ///
    /// Allocating. `sample_rate` is in Hz. Nothing here resamples: the
    /// architectures that keep the rate do so only to hand it to a child
    /// they reset later, and a capture run at a rate other than
    /// [`crate::loader::Model::expected_sample_rate`] simply sounds wrong.
    fn reset(&mut self, sample_rate: f64, max_buffer: usize) {
        let _ = sample_rate;
        self.set_max_buffer_size(max_buffer);
        if self.prewarm_on_reset() {
            self.prewarm();
        }
    }

    /// Select the size of a size-switchable model, `0.0` smallest to `1.0`
    /// largest: `nam::SlimmableModel::SetSlimmableSize`.
    ///
    /// Only `SlimmableContainer` answers it; every other architecture
    /// returns [`Error::Config`]. Not real-time safe: selecting a child
    /// resets it, which allocates.
    fn set_slimmable_size(&mut self, _val: f64) -> Result<()> {
        Err(Error::Config("this model is not slimmable".into()))
    }

    /// The control values in `(0.0, 1.0)` at which the selected model
    /// changes, ascending. Empty for an architecture that does not switch.
    ///
    /// `0.0` and `1.0` are implied bounds and are not listed. A host uses
    /// this to quantise a continuous control to the sizes that exist.
    fn slimmable_size_breakpoints(&self) -> Vec<f64> {
        Vec::new()
    }
}

/// Push blocks of silence through `engine` until it has settled.
///
/// The body of `nam::DSP::prewarm` (`NAM/dsp.cpp`), shared by every
/// architecture that implements [`Engine::prewarm`] by running the model.
/// `max_buffer` is the size the caller was last given; 0 means nothing has
/// been sized yet, and the default block size is installed first, as
/// upstream does.
///
/// The overshoot is load-bearing. The reference pushes *whole* blocks and
/// stops once the running total reaches the target, so it settles on a
/// multiple of the block size rather than on the exact prewarm length. For a
/// recurrent model those extra samples land in the state and change the
/// first real output, so the loop is reproduced rather than tidied into an
/// exact count.
///
/// Allocating; never call it from the audio thread.
pub(crate) fn prewarm_with_silence(engine: &mut dyn Engine, max_buffer: usize) {
    let max_buffer = if max_buffer == 0 {
        engine.set_max_buffer_size(crate::DEFAULT_MAX_BUFFER_SIZE);
        crate::DEFAULT_MAX_BUFFER_SIZE
    } else {
        max_buffer
    };
    // `GetPrewarmSamples` is read here, after the sizing and before the loop,
    // which is the reference's order. No architecture in this crate changes
    // its answer as a result, but one that asks a child might.
    let target = engine.prewarm_samples();
    if target == 0 {
        return;
    }

    let n = max_buffer.max(1);
    let in_channels = engine.in_channels();
    let zeros = vec![0.0f32; n];
    let mut out = vec![vec![0.0f32; n]; engine.out_channels()];
    let mut processed = 0usize;
    while processed < target {
        {
            let ins: Vec<&[f32]> = (0..in_channels).map(|_| zeros.as_slice()).collect();
            let mut outs: Vec<&mut [f32]> = out.iter_mut().map(|v| v.as_mut_slice()).collect();
            engine.process(&ins, &mut outs, n);
        }
        processed += n;
    }
}

/// Build the engine for a parsed `config` block, consuming `weights` in the
/// architecture's order.
///
/// `sample_rate` is the file's, or `None` when the file does not say; the
/// architectures that need it (the LSTM's prewarm length, the container's
/// child resets) take it from here.
///
/// Fails with [`Error::WeightCount`] when `weights` is not exactly as long
/// as the architecture consumes, [`Error::Config`] when a shape does not fit
/// together or exceeds the crate's limits, and
/// [`Error::UnsupportedArchitecture`] for a file whose architecture is
/// recognised but not run.
pub(crate) fn build(
    config: ArchConfig,
    weights: &[f32],
    sample_rate: Option<f64>,
) -> Result<Box<dyn Engine>> {
    Ok(match config {
        // A slimmable WaveNet is still `architecture: "WaveNet"`, read and
        // run at its full width.
        ArchConfig::WaveNet(cfg) => crate::a2_fast::select(WaveNet::new(cfg, weights)?),
        ArchConfig::Lstm(cfg) => Box::new(Lstm::new(&cfg, weights, sample_rate)?),
        // ConvNet and Linear are pre-WaveNet formats. The file is still
        // recognised, since a `.nam` tool should be able to say what it
        // holds, but this crate carries no engine for them.
        ArchConfig::Dropped(name) => {
            return Err(Error::UnsupportedArchitecture(format!(
                "{name}: a pre-WaveNet architecture this crate recognises but does not run"
            )));
        }
        ArchConfig::SlimmableContainer(cfg) => Box::new(ContainerModel::new(cfg, sample_rate)?),
        ArchConfig::Sequential(cfg) => Box::new(SequentialModel::new(cfg)?),
    })
}

/// [`build`] for a whole nested document.
pub(crate) fn build_file(file: NamFile) -> Result<Box<dyn Engine>> {
    build(file.config, &file.weights, file.sample_rate)
}
