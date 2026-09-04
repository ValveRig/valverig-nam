//! Typed model configuration: the `config` block of a `.nam` file.
//!
//! One struct per architecture, transliterated from the reference's
//! `parse_config_json` functions:
//!
//! | architecture        | reference                                        |
//! |---------------------|--------------------------------------------------|
//! | `WaveNet`           | `NAM/wavenet/model.cpp`, `NAM/wavenet/params.h`  |
//! | `LSTM`              | `NAM/lstm.cpp`                                   |
//! | `Linear`            | `NAM/linear.cpp`                                 |
//! | `ConvNet`           | `NAM/convnet.cpp`                                |
//! | `Sequential`        | `NAM/sequential.cpp`                             |
//! | `SlimmableContainer`| `NAM/container.cpp`                              |
//!
//! Those six are the complete set: they are exactly the names passed to
//! `ConfigParserHelper` across the reference tree.
//!
//! Where the reference defers work to a `ModelConfig::create()` body, as the
//! `Sequential` and `SlimmableContainer` architectures do by keeping the raw
//! JSON and only looking at it when the DSP is built, this module does the
//! work at parse time instead, so that a `NamFile` is always fully typed. The errors are the
//! same errors; only the moment they surface moves earlier. Validation
//! performed by the *DSP constructors* is not reproduced here and is called
//! out on the types that need it.
//!
//! # How a block is read
//!
//! Every key whose schema is fixed is a `Deserialize` field, so its name is
//! written once, next to what it fills. The reference's conversion rules are
//! not serde's, and [`super::de`] is where the difference is spelled out; the
//! `#[serde(…)]` attributes here name the rule rather than restate it.
//!
//! What is left over is what the reference does *not* express as a schema, and
//! that is still written out by hand:
//!
//! * defaults that read another field: `bottleneck` falls back to `channels`,
//!   `head1x1.out_channels` to the layer array's;
//! * keys that are alternatives: `head` versus `head_size`/`head_bias`,
//!   `kernel_size` versus `kernel_sizes`, `gating_mode` versus `gated`;
//! * keys holding either one value or one per layer (`activation`,
//!   `gating_mode`, `secondary_activation`), whose lengths are then checked
//!   against `dilations`;
//! * the eight FiLM sites, which are read through [`FILM_KEYS`] so that the
//!   one place they are spelled out stays the one place.
//!
//! Those live in a `*Raw` struct plus a function, rather than in a `Deserialize`
//! impl, because the checks need a whole layer array in hand.

use crate::activations::Activation;
use crate::error::{Error, Result};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

use super::NamFile;
use super::de::{self, Count, F32};

/// The parsed `config` block, discriminated by the file's `architecture`.
///
/// The reference has no `"Container"` architecture: `nam::container::ContainerConfig`
/// is registered under the name `"SlimmableContainer"`, and a slimmable
/// *WaveNet* is still architecture `"WaveNet"`. [`ContainerConfig`] is
/// therefore reachable only through [`ArchConfig::SlimmableContainer`].
#[derive(Debug, Clone, PartialEq)]
pub enum ArchConfig {
    /// `architecture: "WaveNet"`.
    WaveNet(WaveNetConfig),
    /// `architecture: "LSTM"`.
    Lstm(LstmConfig),
    /// A pre-WaveNet architecture this crate reads but does not run, held
    /// as its name from the file: `"Linear"` or `"ConvNet"`.
    ///
    /// The `config` block is not parsed. Its fields exist only to build an
    /// engine this crate does not have, and refusing the file on a missing
    /// `receptive_field` would name the wrong reason.
    Dropped(String),
    /// `architecture: "Sequential"`.
    Sequential(SequentialConfig),
    /// `architecture: "SlimmableContainer"`.
    SlimmableContainer(ContainerConfig),
}

/// Dispatch on the architecture name, as `ConfigParserRegistry::parse` does.
///
/// `weights` is the file's top-level weight array; the container architectures
/// inspect it (`Sequential` requires it to be empty) but never consume it.
pub(super) fn parse_arch_config(
    architecture: &str,
    config: &Value,
    sample_rate: Option<f64>,
    weights: &[f32],
) -> Result<ArchConfig> {
    match architecture {
        "WaveNet" => Ok(ArchConfig::WaveNet(WaveNetConfig::from_json(
            config,
            sample_rate,
        )?)),
        "LSTM" => Ok(ArchConfig::Lstm(de::from_value(config, "LSTM config")?)),
        "Linear" | "ConvNet" => Ok(ArchConfig::Dropped(architecture.to_string())),
        "Sequential" => Ok(ArchConfig::Sequential(SequentialConfig::from_json(
            config, weights,
        )?)),
        "SlimmableContainer" => Ok(ArchConfig::SlimmableContainer(ContainerConfig::from_json(
            config,
        )?)),
        other => Err(Error::UnsupportedArchitecture(other.to_string())),
    }
}

// ============================================================================
// WaveNet
// ============================================================================

/// `architecture: "WaveNet"`: `nam::wavenet::WaveNetConfig`.
#[derive(Debug, Clone, PartialEq)]
pub struct WaveNetConfig {
    /// Audio input channels. `config.in_channels`, default 1.
    pub in_channels: usize,
    /// One entry per element of `config.layers`.
    pub layer_arrays: Vec<LayerArrayConfig>,
    /// `config.head_scale`, required. Held as `f32` because the reference's
    /// `WaveNetConfig::head_scale` is a `float`, so the JSON double is narrowed
    /// once here and never again.
    pub head_scale: f32,
    /// The optional post-stack head: `config.head`, when present and non-null.
    /// The reference also keeps a separate `with_head` flag; it can only ever
    /// disagree with this through a bug, so it is not carried.
    pub head: Option<PostStackHeadConfig>,
    /// `config.condition_dsp`: a complete nested model whose output feeds the
    /// layer arrays' conditioning input.
    pub condition_dsp: Option<Box<NamFile>>,
}

/// One entry of `config.layers`: `nam::wavenet::LayerArrayParams`.
///
/// The reference stores the per-layer quantities as parallel vectors on this
/// struct; they live in [`Self::layers`] here, one [`LayerConfig`] per element
/// of `dilations`.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerArrayConfig {
    /// Channels entering the array's rechannel convolution.
    pub input_size: usize,
    /// Width of the conditioning signal.
    pub condition_size: usize,
    /// Channels carried from layer to layer.
    pub channels: usize,
    /// Internal channel count. Defaults to [`Self::channels`].
    pub bottleneck: usize,
    /// Output channels of the array's head rechannel convolution
    /// (`head.out_channels`, or legacy `head_size`).
    pub head_size: usize,
    /// Dilation of the head rechannel convolution. Only the nested `head`
    /// object can set it; defaults to 1.
    pub head_dilation: usize,
    /// Kernel size of the head rechannel convolution. 1 on the legacy path.
    pub head_kernel_size: usize,
    /// Whether the head rechannel convolution has a bias.
    pub head_bias: bool,
    /// Groups for each layer's dilated input convolution. Default 1.
    pub groups_input: usize,
    /// Groups for each layer's input-mixin convolution. Default 1.
    pub groups_input_mixin: usize,
    /// The optional post-activation 1x1 on the residual path.
    pub layer1x1: Layer1x1Config,
    /// The optional 1x1 that feeds the head directly.
    pub head1x1: Head1x1Config,
    /// The eight FiLM sites, indexed by [`film_site`] and named in the file by
    /// [`FILM_KEYS`]. `wavenet::Layer` reads this array by the same indices.
    pub films: [FilmConfig; 8],
    /// One entry per dilation.
    pub layers: Vec<LayerConfig>,
}

/// The per-layer quantities of a layer array: `nam::wavenet::LayerParams`.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerConfig {
    /// Dilation of this layer's input convolution.
    pub dilation: usize,
    /// Kernel size of this layer's input convolution.
    pub kernel_size: usize,
    /// Primary activation.
    pub activation: Activation,
    /// How the doubled bottleneck is combined.
    pub gating_mode: GatingMode,
    /// Activation applied to the gate half.
    ///
    /// Never read when [`Self::gating_mode`] is [`GatingMode::None`], and the
    /// reference does not agree with itself about what it holds there. Two of
    /// its three ungated paths push a value-initialised `ActivationConfig{}`,
    /// whose `type` is the zero of the enum, `Tanh`. The third, a scalar
    /// `"gating_mode": "none"`, default-initialises a local
    /// `ActivationConfig` and copies it, leaving `type` indeterminate; a
    /// reference build with assertions on prints garbage for it. This crate
    /// uses `Tanh` throughout.
    pub secondary_activation: Activation,
}

/// `nam::wavenet::GatingMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatingMode {
    /// Plain activation; the bottleneck is not doubled.
    None,
    /// Element-wise product of the activated and gate halves.
    Gated,
    /// Weighted average between activated and pre-activation values.
    Blended,
}

impl GatingMode {
    /// `parse_gating_mode_str` in `NAM/wavenet/model.cpp`.
    fn from_name(name: &str) -> Result<Self> {
        match name {
            "gated" => Ok(GatingMode::Gated),
            "blended" => Ok(GatingMode::Blended),
            "none" => Ok(GatingMode::None),
            other => Err(Error::Config(format!("Invalid gating_mode: {other}"))),
        }
    }
}

impl<'de> Deserialize<'de> for GatingMode {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let name = String::deserialize(d)?;
        GatingMode::from_name(&name).map_err(serde::de::Error::custom)
    }
}

/// `nam::wavenet::Layer1x1Params`. Absent from the JSON means active with one
/// group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct Layer1x1Config {
    /// Whether the convolution exists at all.
    pub active: bool,
    /// Groups for the grouped convolution.
    #[serde(deserialize_with = "de::count")]
    pub groups: usize,
}

impl Layer1x1Config {
    /// The value a layer array without a `layer1x1` key gets.
    const ABSENT: Layer1x1Config = Layer1x1Config {
        active: true,
        groups: 1,
    };
}

/// `nam::wavenet::Head1x1Params`. Absent from the JSON means inactive, with
/// `out_channels` defaulting to the layer array's `channels`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct Head1x1Config {
    /// Whether the convolution exists at all.
    pub active: bool,
    /// Output channels.
    #[serde(deserialize_with = "de::count")]
    pub out_channels: usize,
    /// Groups for the grouped convolution.
    #[serde(deserialize_with = "de::count")]
    pub groups: usize,
}

impl Head1x1Config {
    /// The value a layer array without a `head1x1` key gets: inactive, and as
    /// wide as the array it sits in.
    const fn absent(channels: usize) -> Self {
        Head1x1Config {
            active: false,
            out_channels: channels,
            groups: 1,
        }
    }
}

/// `nam::wavenet::_FiLMParams`: one of the eight modulation sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilmConfig {
    /// Whether this site modulates at all.
    pub active: bool,
    /// `true` to apply scale *and* shift, `false` for scale alone. This drives
    /// the weight count, so it matters even when [`Self::active`] is false in
    /// a config that sets both.
    pub shift: bool,
    /// Groups for the condition-to-scale-shift convolution.
    pub groups: usize,
}

impl FilmConfig {
    /// The all-off value the reference builds as `_FiLMParams(false, false)`,
    /// whose `groups` defaults to 1.
    const OFF: FilmConfig = FilmConfig {
        active: false,
        shift: false,
        groups: 1,
    };
}

/// One FiLM site.
///
/// The reference's shorthand is `layer_config[key] == false`, a comparison
/// against a JSON boolean: only a literal `false` disables the site this way.
/// A missing key does too, but that is handled a level up, by the caller that
/// looks the key up; this impl never sees one.
impl<'de> Deserialize<'de> for FilmConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        /// The object form. Every key defaults, so `{}` is a shift-and-scale
        /// site with one group.
        #[derive(Deserialize)]
        struct Fields {
            #[serde(default = "de::yes")]
            active: bool,
            #[serde(default = "de::yes")]
            shift: bool,
            #[serde(default = "de::one", deserialize_with = "de::count")]
            groups: usize,
        }

        let v = Value::deserialize(d)?;
        if v == Value::Bool(false) {
            return Ok(FilmConfig::OFF);
        }
        if !v.is_object() {
            return Err(serde::de::Error::custom(format!(
                "expected an object or false, found {}",
                de::kind(&v)
            )));
        }
        let f = Fields::deserialize(&v).map_err(serde::de::Error::custom)?;
        Ok(FilmConfig {
            active: f.active,
            shift: f.shift,
            groups: f.groups,
        })
    }
}

/// The optional post-stack head: `nam::wavenet::HeadParams`.
#[derive(Debug, Clone, PartialEq)]
pub struct PostStackHeadConfig {
    /// Implied by the last layer array's `head_size`; never read from the JSON
    /// even when a legacy file carries it.
    pub in_channels: usize,
    /// Internal channel count.
    pub channels: usize,
    /// Output channels of the whole model.
    pub out_channels: usize,
    /// One kernel size per convolution in the head stack. Non-empty.
    pub kernel_sizes: Vec<usize>,
    /// Activation between the head's convolutions.
    pub activation: Activation,
}

const SLIMMABLE_METHOD: &str = "slice_channels_uniform";

impl WaveNetConfig {
    /// `nam::wavenet::parse_config_json`, plus the slimmable detection that
    /// `create_config` performs around it.
    ///
    /// A slimmable WaveNet, one whose layer arrays carry a `slimmable`
    /// block, is read at its full width, the width the reference itself
    /// starts at. The narrower widths the block offers are not built.
    ///
    /// `expected_sample_rate` is the file's `sample_rate`; the reference
    /// compares the condition DSP's against it and refuses a mismatch.
    fn from_json(config: &Value, expected_sample_rate: Option<f64>) -> Result<Self> {
        // The reference checks slimmability first, in create_config, and
        // routes to a different ModelConfig. Detection can itself fail on a
        // bad `method`, so it has to run before anything else.
        let slimmable = detect_slimmable(config)?;
        // SlimmableWavenetConfig::create accepts a {"model": {...}} wrapper.
        // Detection above never looks inside it, so the wrapper is only
        // reachable for a config that carries both keys.
        let model = if slimmable {
            config.get("model").unwrap_or(config)
        } else {
            config
        };

        let raw: WaveNetRaw = de::from_value(model, "WaveNet config")?;

        let condition_dsp = match de::non_null(model, "condition_dsp") {
            Some(cd) => {
                let nested = super::parse_value(cd)?;
                // The reference compares the raw doubles, unknown (-1) included,
                // so a nested model that does not state its rate under an outer
                // one that does is a mismatch there and here.
                if nested.sample_rate != expected_sample_rate {
                    return Err(Error::Config(format!(
                        "Condition DSP expected sample rate ({}) doesn't match WaveNet expected sample rate ({})",
                        rate_text(nested.sample_rate),
                        rate_text(expected_sample_rate)
                    )));
                }
                Some(Box::new(nested))
            }
            None => None,
        };

        const NO_LAYERS: &[Value] = &[];
        let layers_json: &[Value] = match model.get("layers") {
            None => {
                return Err(Error::Config(
                    "WaveNet config: missing required key \"layers\"".into(),
                ));
            }
            // nlohmann's `size()` on null is 0, so a null `layers` yields no
            // layer arrays and falls through to the emptiness check below.
            Some(Value::Null) => NO_LAYERS,
            Some(Value::Array(a)) => a,
            Some(other) => {
                return Err(Error::Config(format!(
                    "WaveNet config: \"layers\" must be an array, found {}",
                    de::kind(other)
                )));
            }
        };

        let mut layer_arrays = Vec::with_capacity(layers_json.len());
        for (i, lc) in layers_json.iter().enumerate() {
            layer_arrays.push(parse_layer_array(lc, i)?);
        }

        if layer_arrays.is_empty() {
            return Err(Error::Config(
                "WaveNet config requires at least one layer array".into(),
            ));
        }

        let head = match de::non_null(model, "head") {
            Some(hj) => Some(parse_post_stack_head(
                hj,
                layer_arrays.last().unwrap().head_size,
            )?),
            None => None,
        };

        Ok(WaveNetConfig {
            in_channels: raw.in_channels,
            layer_arrays,
            head_scale: raw.head_scale,
            head,
            condition_dsp,
        })
    }
}

/// A sample rate for an error message: the reference's `-1` when unknown.
fn rate_text(rate: Option<f64>) -> String {
    match rate {
        Some(r) => r.to_string(),
        None => "-1".to_string(),
    }
}

/// The scalar `config` keys of a WaveNet.
///
/// `layers`, `head` and `condition_dsp` are read from the same object
/// afterwards: a `null` in `layers` is nlohmann's zero-length array rather
/// than a type error, `head` needs a channel count only the layer arrays know,
/// and `condition_dsp` is a whole nested document that [`de::non_null`] hands
/// over by reference instead of copying.
#[derive(Deserialize)]
struct WaveNetRaw {
    #[serde(default = "de::one", deserialize_with = "de::count")]
    in_channels: usize,
    #[serde(deserialize_with = "de::float")]
    head_scale: f32,
}

/// `config_is_slimmable_wavenet` in `NAM/wavenet/model.cpp`.
///
/// Returns at the *first* layer whose method matches, so a later layer with a
/// bad method is not seen here. A non-empty method that is not the supported
/// one aborts detection; an empty one is skipped.
///
/// The check is kept even though the widths are not: a method the reference
/// cannot slice is a file it refuses, and refusing what it refuses is the
/// contract.
fn detect_slimmable(config: &Value) -> Result<bool> {
    let Some(Value::Array(layers)) = config.get("layers") else {
        return Ok(false);
    };
    for lc in layers {
        let Some(slim) = lc.get("slimmable").filter(|s| s.is_object()) else {
            continue;
        };
        let method = de::from_value::<SlimmableRaw>(slim, "slimmable")?.method;
        if method != SLIMMABLE_METHOD {
            if !method.is_empty() {
                return Err(Error::Config(format!(
                    "SlimmableWavenet: unsupported slimmable method '{method}'"
                )));
            }
            continue;
        }
        return Ok(true);
    }
    Ok(false)
}

/// `layers[i].slimmable`, as far as this crate reads it.
///
/// Only the method name matters here. The reference also reads
/// `kwargs.allowed_channels` to build the narrower widths; those are not
/// built, so the key is ignored.
#[derive(Deserialize)]
struct SlimmableRaw {
    #[serde(default)]
    method: String,
}

/// One iteration of the `config["layers"]` loop in
/// `nam::wavenet::parse_config_json`, in the reference's order so that a
/// config with several problems reports the same one it does.
fn parse_layer_array(lc: &Value, index: usize) -> Result<LayerArrayConfig> {
    let ctx = format!("Layer array {index}");
    let raw: LayerArrayRaw = de::from_value(lc, &ctx)?;

    let channels = raw.channels;
    let bottleneck = raw.bottleneck.map_or(channels, |c| c.0);
    let layer1x1 = raw.layer1x1.unwrap_or(Layer1x1Config::ABSENT);

    // Nested "head" wins over the legacy head_size/head_bias pair.
    let (head_size, head_dilation, head_kernel_size, head_bias) = match (&raw.head, &raw.head_size)
    {
        (Some(hj), _) => {
            // Checked here rather than by `LayerHeadRaw`, which would report
            // a bare "invalid type" for a key that has two spellings.
            if !hj.is_object() {
                return Err(Error::Config(format!(
                    "{ctx}: 'head' must be a JSON object"
                )));
            }
            let h: LayerHeadRaw = de::from_value(hj, &format!("{ctx}.head"))?;
            (h.out_channels, h.head_dilation, h.kernel_size, h.bias)
        }
        (None, Some(hs)) => {
            let Count(head_size) = de::from_value(hs, &format!("{ctx}.head_size"))?;
            let Some(hb) = &raw.head_bias else {
                return Err(Error::Config(format!(
                    "{ctx}: missing required key \"head_bias\""
                )));
            };
            (
                head_size,
                1,
                1,
                de::from_value(hb, &format!("{ctx}.head_bias"))?,
            )
        }
        (None, None) => {
            return Err(Error::Config(format!(
                "{ctx}: expected 'head' object with out_channels, kernel_size, and bias, \
                 or legacy 'head_size' and 'head_bias'"
            )));
        }
    };

    if head_kernel_size < 1 {
        return Err(Error::Config(format!(
            "{ctx}: head.kernel_size must be >= 1"
        )));
    }

    let dilations: Vec<usize> = raw.dilations.iter().flatten().map(|c| c.0).collect();
    let num_layers = dilations.len();

    let kernel_sizes = resolve_kernel_sizes(&raw, num_layers, &ctx)?;
    let activations = parse_activations(lc, num_layers, &ctx)?;
    let (gating_modes, secondary_activations) = parse_gating(lc, num_layers, &ctx)?;

    let head1x1 = raw.head1x1.unwrap_or(Head1x1Config::absent(channels));

    let mut films = [FilmConfig::OFF; 8];
    for (site, key) in FILM_KEYS.iter().enumerate() {
        films[site] = parse_film(lc, key, &ctx)?;
    }

    if films[film_site::LAYER1X1_POST].active && !layer1x1.active {
        return Err(Error::Config(format!(
            "{ctx}: layer1x1_post_film cannot be active when layer1x1.active is false"
        )));
    }

    // LayerArrayParams' own constructor checks, which run last.
    if kernel_sizes.is_empty() {
        return Err(Error::Config(
            "LayerArrayParams: kernel_sizes must not be empty".into(),
        ));
    }

    let layers = (0..num_layers)
        .map(|i| LayerConfig {
            dilation: dilations[i],
            kernel_size: kernel_sizes[i],
            activation: activations[i].clone(),
            gating_mode: gating_modes[i],
            secondary_activation: secondary_activations[i].clone(),
        })
        .collect();

    Ok(LayerArrayConfig {
        input_size: raw.input_size,
        condition_size: raw.condition_size,
        channels,
        bottleneck,
        head_size,
        head_dilation,
        head_kernel_size,
        head_bias,
        groups_input: raw.groups_input,
        groups_input_mixin: raw.groups_input_mixin,
        layer1x1,
        head1x1,
        films,
        layers,
    })
}

/// The fixed-schema keys of one entry of `config.layers`.
///
/// The polymorphic ones, `activation`, `gating_mode` and
/// `secondary_activation`, each either one value for the array or one per
/// layer, and the eight FiLM sites are read from the same object afterwards,
/// by the functions that know their shapes. Unknown keys are ignored, so the
/// two passes coexist.
#[derive(Deserialize)]
struct LayerArrayRaw {
    #[serde(deserialize_with = "de::count")]
    input_size: usize,
    #[serde(deserialize_with = "de::count")]
    condition_size: usize,
    #[serde(deserialize_with = "de::count")]
    channels: usize,
    #[serde(default, deserialize_with = "de::present")]
    bottleneck: Option<Count>,
    #[serde(default = "de::one", deserialize_with = "de::count")]
    groups_input: usize,
    #[serde(default = "de::one", deserialize_with = "de::count")]
    groups_input_mixin: usize,
    /// Wins over the legacy `head_size`/`head_bias` pair when present and
    /// non-null; see [`LayerHeadRaw`].
    #[serde(default)]
    head: Option<Value>,
    /// Both are converted only on the branch that reads them, so a file
    /// carrying a `head` object next to a stale `"head_size": null` is
    /// accepted, as it is by the reference.
    #[serde(default, deserialize_with = "de::present")]
    head_size: Option<Value>,
    #[serde(default, deserialize_with = "de::present")]
    head_bias: Option<Value>,
    /// Plain `Option`, because a `null` here is nlohmann's zero-length array:
    /// it leaves the array with no layers and lands on the empty-kernel-sizes
    /// error, not a type error.
    #[serde(default)]
    dilations: Option<Vec<Count>>,
    #[serde(default, deserialize_with = "de::present")]
    kernel_size: Option<Count>,
    #[serde(default, deserialize_with = "de::present")]
    kernel_sizes: Option<Vec<Count>>,
    #[serde(default, deserialize_with = "de::present")]
    layer1x1: Option<Layer1x1Config>,
    #[serde(default, deserialize_with = "de::present")]
    head1x1: Option<Head1x1Config>,
}

/// The nested `head` object of a layer array: the modern spelling of
/// `head_size`/`head_bias`, and the only one that can set a dilation or a
/// kernel size.
#[derive(Deserialize)]
struct LayerHeadRaw {
    #[serde(deserialize_with = "de::count")]
    out_channels: usize,
    #[serde(default = "de::one", deserialize_with = "de::count")]
    head_dilation: usize,
    #[serde(deserialize_with = "de::count")]
    kernel_size: usize,
    bias: bool,
}

/// `kernel_size` (one int for the whole array) exclusive-or `kernel_sizes`
/// (one per layer).
fn resolve_kernel_sizes(raw: &LayerArrayRaw, num_layers: usize, ctx: &str) -> Result<Vec<usize>> {
    match (raw.kernel_size, &raw.kernel_sizes) {
        (Some(_), Some(_)) => Err(Error::Config(format!(
            "{ctx}: only one of kernel_size (int) or kernel_sizes (array) may be provided"
        ))),
        (None, Some(ks)) => {
            if ks.len() != num_layers {
                return Err(Error::Config(format!(
                    "{ctx}: kernel_sizes array size ({}) must match dilations size ({num_layers})",
                    ks.len()
                )));
            }
            Ok(ks.iter().map(|c| c.0).collect())
        }
        (Some(k), None) => Ok(vec![k.0; num_layers]),
        (None, None) => Err(Error::Config(format!(
            "{ctx}: either kernel_size (int) or kernel_sizes (array) must be provided"
        ))),
    }
}

/// `activation`: one config for the whole array, or one per layer.
fn parse_activations(lc: &Value, num_layers: usize, ctx: &str) -> Result<Vec<Activation>> {
    // The reference indexes a *mutable copy* of the layer config here, so a
    // missing "activation" materialises as null and fails inside from_json
    // with "Invalid activation config".
    let null = Value::Null;
    let j = lc.get("activation").unwrap_or(&null);
    if let Value::Array(items) = j {
        let mut out = Vec::with_capacity(items.len());
        for a in items {
            out.push(activation_from_json(a)?);
        }
        if out.len() != num_layers {
            return Err(Error::Config(format!(
                "{ctx}: activation array size ({}) must match dilations size ({num_layers})",
                out.len()
            )));
        }
        Ok(out)
    } else {
        let a = activation_from_json(j)?;
        Ok(vec![a; num_layers])
    }
}

/// `gating_mode` (string or per-layer array), legacy `gated` bool, or neither.
///
/// The secondary activation is only ever read for a layer whose gating is on,
/// which is what lets `A2.nam` ship `"secondary_activation": [null, ...]`
/// alongside an all-`"none"` `gating_mode` array.
fn parse_gating(
    lc: &Value,
    num_layers: usize,
    ctx: &str,
) -> Result<(Vec<GatingMode>, Vec<Activation>)> {
    // The value the reference leaves in an ungated layer's secondary slot is
    // Tanh on two paths and indeterminate on the third; see LayerConfig.
    let default_secondary = Activation::Tanh;

    if let Some(gm) = lc.get("gating_mode") {
        if let Value::Array(items) = gm {
            let mut modes = Vec::with_capacity(items.len());
            let mut secondaries = Vec::with_capacity(items.len());
            for item in items {
                let mode = de::from_value(item, &format!("{ctx}.gating_mode"))?;
                modes.push(mode);
                if mode != GatingMode::None {
                    secondaries.push(match lc.get("secondary_activation") {
                        Some(Value::Array(sa)) => {
                            if modes.len() > sa.len() {
                                return Err(Error::Config(format!(
                                    "{ctx}: secondary_activation array size must be at least {}",
                                    modes.len()
                                )));
                            }
                            activation_from_json(&sa[modes.len() - 1])?
                        }
                        Some(single) => activation_from_json(single)?,
                        None => Activation::Sigmoid,
                    });
                } else {
                    secondaries.push(default_secondary.clone());
                }
            }
            if modes.len() != num_layers {
                return Err(Error::Config(format!(
                    "{ctx}: gating_mode array size ({}) must match dilations size ({num_layers})",
                    modes.len()
                )));
            }
            if let Some(Value::Array(sa)) = lc.get("secondary_activation")
                && sa.len() != num_layers
            {
                return Err(Error::Config(format!(
                    "{ctx}: secondary_activation array size ({}) must match dilations size ({num_layers})",
                    sa.len()
                )));
            }
            return Ok((modes, secondaries));
        }

        let mode: GatingMode = de::from_value(gm, &format!("{ctx}.gating_mode"))?;
        let secondary = if mode != GatingMode::None {
            match lc.get("secondary_activation") {
                // Note this takes the whole value: with a scalar gating_mode,
                // an array-valued secondary_activation is an error, not a
                // per-layer list.
                Some(v) => activation_from_json(v)?,
                None => Activation::Sigmoid,
            }
        } else {
            default_secondary
        };
        return Ok((vec![mode; num_layers], vec![secondary; num_layers]));
    }

    if let Some(g) = lc.get("gated") {
        let gated: bool = de::from_value(g, &format!("{ctx}.gated"))?;
        let mode = if gated {
            GatingMode::Gated
        } else {
            GatingMode::None
        };
        let secondary = if gated {
            Activation::Sigmoid
        } else {
            default_secondary
        };
        return Ok((vec![mode; num_layers], vec![secondary; num_layers]));
    }

    Ok((
        vec![GatingMode::None; num_layers],
        vec![default_secondary; num_layers],
    ))
}

/// Indices into [`LayerArrayConfig::films`], and into the matching array of
/// built sites in `wavenet::Layer`.
pub mod film_site {
    /// Before the dilated convolution.
    pub const CONV_PRE: usize = 0;
    /// After the dilated convolution.
    pub const CONV_POST: usize = 1;
    /// Before the condition mixin.
    pub const INPUT_MIXIN_PRE: usize = 2;
    /// After the condition mixin.
    pub const INPUT_MIXIN_POST: usize = 3;
    /// On the summed pre-activation.
    pub const ACTIVATION_PRE: usize = 4;
    /// On the activation output.
    pub const ACTIVATION_POST: usize = 5;
    /// After the residual-path 1x1.
    pub const LAYER1X1_POST: usize = 6;
    /// After the skip-path 1x1.
    pub const HEAD1X1_POST: usize = 7;
}

/// The file's FiLM field names, in [`film_site`] index order.
///
/// The single place the eight sites are spelled out. The order is a contract
/// with [`film_site`] just above, and `film_keys_are_in_site_order` in the
/// format tests is what holds the two together.
pub const FILM_KEYS: [&str; 8] = [
    "conv_pre_film",
    "conv_post_film",
    "input_mixin_pre_film",
    "input_mixin_post_film",
    "activation_pre_film",
    "activation_post_film",
    "layer1x1_post_film",
    "head1x1_post_film",
];

/// One FiLM site: absent is off, and everything else is [`FilmConfig`]'s own
/// `false`-or-object rule.
fn parse_film(lc: &Value, key: &str, ctx: &str) -> Result<FilmConfig> {
    match lc.get(key) {
        None => Ok(FilmConfig::OFF),
        Some(j) => de::from_value(j, &format!("{ctx}.{key}")),
    }
}

/// The top-level `head` object.
#[derive(Deserialize)]
struct PostStackHeadRaw {
    /// A legacy file may carry it; it is checked against the last layer
    /// array's `head_size` and then discarded.
    #[serde(default)]
    in_channels: Option<Count>,
    #[serde(deserialize_with = "de::count")]
    channels: usize,
    #[serde(deserialize_with = "de::count")]
    out_channels: usize,
    #[serde(deserialize_with = "de::count_vec")]
    kernel_sizes: Vec<usize>,
    activation: Activation,
}

fn parse_post_stack_head(hj: &Value, implied_in_channels: usize) -> Result<PostStackHeadConfig> {
    let raw: PostStackHeadRaw = de::from_value(hj, "WaveNet config.head")?;
    if let Some(Count(legacy)) = raw.in_channels
        && legacy != implied_in_channels
    {
        return Err(Error::Config(format!(
            "WaveNet config: head.in_channels ({legacy}) must equal last layer's head_size ({implied_in_channels})"
        )));
    }
    if raw.kernel_sizes.is_empty() {
        return Err(Error::Config(
            "WaveNet config: head.kernel_sizes must be non-empty".into(),
        ));
    }
    Ok(PostStackHeadConfig {
        in_channels: implied_in_channels,
        channels: raw.channels,
        out_channels: raw.out_channels,
        kernel_sizes: raw.kernel_sizes,
        activation: raw.activation,
    })
}

// ============================================================================
// Activations
// ============================================================================

/// `nam::activations::ActivationConfig::from_json`, folded into the crate's
/// [`Activation`].
///
/// The reference splits config from behaviour: `ActivationConfig` keeps the
/// optional parameters, and `Activation::get_activation` later substitutes the
/// defaults for the ones that were left out. Those defaults are applied here,
/// so that every [`Activation`] is fully determined:
///
/// * `LeakyReLU` without `negative_slope` → 0.01
/// * `PReLU` with neither `negative_slope` nor `negative_slopes` → `[0.01]`
/// * `LeakyHardtanh` → `min_val` -1, `max_val` 1, both slopes 0.01
pub fn activation_from_json(j: &Value) -> Result<Activation> {
    if let Some(name) = j.as_str() {
        return Activation::from_name(name);
    }
    if !j.is_object() {
        return Err(Error::Config(
            "Invalid activation config: expected string or object".into(),
        ));
    }
    // `from_name` validates the type and supplies the same defaults the
    // bare-string form gets, which the parameter branches below override.
    let named: ActivationType = de::from_value(j, "activation")?;
    let base = Activation::from_name(&named.kind)?;
    // One struct per parameter set, read only on the branch that wants it:
    // the reference never looks at a parameter its activation does not take,
    // so neither does this: a stale `"negative_slope": null` on a `Tanh` is
    // as harmless here as it is there.
    Ok(match base {
        Activation::PReLU { .. } => {
            let p: PReLUParams = de::from_value(j, "activation")?;
            match (p.negative_slope, p.negative_slopes) {
                (Some(F32(s)), _) => Activation::PReLU {
                    negative_slopes: vec![s],
                },
                (None, Some(slopes)) => Activation::PReLU {
                    negative_slopes: slopes.into_iter().map(|s| s.0).collect(),
                },
                (None, None) => base,
            }
        }
        Activation::LeakyReLU { .. } => {
            let p: LeakyReLUParams = de::from_value(j, "activation")?;
            Activation::LeakyReLU {
                negative_slope: p.negative_slope.map_or(0.01, |s| s.0),
            }
        }
        Activation::LeakyHardtanh { .. } => {
            let p: LeakyHardtanhParams = de::from_value(j, "activation")?;
            Activation::LeakyHardtanh {
                min_val: p.min_val.map_or(-1.0, |v| v.0),
                max_val: p.max_val.map_or(1.0, |v| v.0),
                min_slope: p.min_slope.map_or(0.01, |v| v.0),
                max_slope: p.max_slope.map_or(0.01, |v| v.0),
            }
        }
        other => other,
    })
}

/// The one key every activation object has.
#[derive(Deserialize)]
struct ActivationType {
    #[serde(rename = "type")]
    kind: String,
}

/// `negative_slope` wins over `negative_slopes`; with neither, the slope table
/// stays the single 0.01 the bare-string form gets.
#[derive(Deserialize)]
struct PReLUParams {
    #[serde(default, deserialize_with = "de::present")]
    negative_slope: Option<F32>,
    #[serde(default, deserialize_with = "de::present")]
    negative_slopes: Option<Vec<F32>>,
}

#[derive(Deserialize)]
struct LeakyReLUParams {
    #[serde(default, deserialize_with = "de::present")]
    negative_slope: Option<F32>,
}

#[derive(Deserialize)]
struct LeakyHardtanhParams {
    #[serde(default, deserialize_with = "de::present")]
    min_val: Option<F32>,
    #[serde(default, deserialize_with = "de::present")]
    max_val: Option<F32>,
    #[serde(default, deserialize_with = "de::present")]
    min_slope: Option<F32>,
    #[serde(default, deserialize_with = "de::present")]
    max_slope: Option<F32>,
}

/// Both spellings of an activation, so that any config struct can hold one.
impl<'de> Deserialize<'de> for Activation {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let v = Value::deserialize(d)?;
        activation_from_json(&v).map_err(serde::de::Error::custom)
    }
}

// ============================================================================
// LSTM
// ============================================================================

/// `architecture: "LSTM"`: `nam::lstm::LSTMConfig`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct LstmConfig {
    /// Number of stacked cells.
    #[serde(deserialize_with = "de::count")]
    pub num_layers: usize,
    /// Input width of the first cell.
    #[serde(deserialize_with = "de::count")]
    pub input_size: usize,
    /// Hidden-state width of every cell.
    #[serde(deserialize_with = "de::count")]
    pub hidden_size: usize,
    /// Audio input channels. Default 1.
    #[serde(default = "de::one", deserialize_with = "de::count")]
    pub in_channels: usize,
    /// Audio output channels. Default 1.
    #[serde(default = "de::one", deserialize_with = "de::count")]
    pub out_channels: usize,
}

// ============================================================================
// Sequential
// ============================================================================

/// `architecture: "Sequential"`: `nam::sequential::SequentialConfig`.
///
/// The reference's `SequentialModel` constructor additionally checks that
/// adjacent children agree on channel counts; that needs each child's
/// realised channel counts, so it belongs to the model layer. Its
/// sample-rate check is a property of the files and runs here, through
/// `SequentialConfig::check_sample_rates`.
#[derive(Debug, Clone, PartialEq)]
pub struct SequentialConfig {
    /// Children in processing order; each is a complete `.nam` model.
    pub models: Vec<NamFile>,
}

impl SequentialConfig {
    /// `build_models` plus the weight check at the head of
    /// `SequentialConfig::create`, in that order.
    fn from_json(config: &Value, weights: &[f32]) -> Result<Self> {
        if !weights.is_empty() {
            return Err(Error::Config(
                "Sequential: top-level weights must be empty; weights belong to the child models"
                    .into(),
            ));
        }
        let Some(models_json) = config.get("models") else {
            return Err(Error::Config(
                "Sequential: config must contain a 'models' array".into(),
            ));
        };
        let non_empty_array = models_json.as_array().filter(|a| !a.is_empty());
        let Some(items) = non_empty_array else {
            return Err(Error::Config(
                "Sequential: 'models' must be a non-empty array".into(),
            ));
        };
        let mut models = Vec::with_capacity(items.len());
        for child in items {
            if !is_complete_model(child) {
                return Err(Error::Config(
                    "Sequential: each child must be a complete NAM model with version, architecture, config, and weights"
                        .into(),
                ));
            }
            models.push(super::parse_value(child)?);
        }
        Ok(SequentialConfig { models })
    }
}

/// Whether `v` is an object with the four keys `validate_nam_file` requires
/// of every `.nam` document, which `build_models` re-checks for each
/// `Sequential` child.
fn is_complete_model(v: &Value) -> bool {
    v.is_object() && super::REQUIRED_KEYS.iter().all(|k| v.get(k).is_some())
}

impl SequentialConfig {
    /// The `SequentialModel` constructor's sample-rate agreement check.
    ///
    /// `expected` is the chain's own `sample_rate`. A child that does not
    /// state a rate is exempt; the first stated rate, the chain's or a
    /// child's, becomes the one every later child must match.
    pub(super) fn check_sample_rates(&self, expected: Option<f64>) -> Result<()> {
        let mut resolved = expected;
        for m in &self.models {
            let Some(sr) = m.sample_rate else { continue };
            match resolved {
                None => resolved = Some(sr),
                Some(r) if r != sr => {
                    return Err(Error::Config(format!(
                        "SequentialModel: submodel sample rate mismatch (expected {r}, got {sr})"
                    )));
                }
                Some(_) => {}
            }
        }
        Ok(())
    }
}

// ============================================================================
// SlimmableContainer
// ============================================================================

/// One entry of a container's `submodels` array: `nam::container::Submodel`.
#[derive(Debug, Clone, PartialEq)]
pub struct Submodel {
    /// This submodel covers size-control values below `max_value`.
    pub max_value: f64,
    /// A complete `.nam` model.
    pub model: NamFile,
}

/// The scalar half of one entry of a container's `submodels` array. Its
/// `model` is a complete `.nam` document, read by reference rather than
/// deserialized into a field that would copy it.
#[derive(Deserialize)]
struct SubmodelRaw {
    max_value: f64,
}

/// `architecture: "SlimmableContainer"`: `nam::container::ContainerConfig`.
#[derive(Debug, Clone, PartialEq)]
pub struct ContainerConfig {
    /// Submodels ordered by ascending `max_value`.
    pub submodels: Vec<Submodel>,
}

impl ContainerConfig {
    /// `ContainerConfig::create` plus the `ContainerModel` constructor's three
    /// checks.
    ///
    /// Those three are constructor checks in the reference, but they are pure
    /// predicates over the parsed config, namely ordering, coverage of 1.0
    /// and sample-rate agreement, so they run here where the config is
    /// built.
    fn from_json(config: &Value) -> Result<Self> {
        let submodels_json = config
            .get("submodels")
            .and_then(|v| v.as_array())
            .filter(|a| !a.is_empty())
            .ok_or_else(|| {
                Error::Config("SlimmableContainer: 'submodels' must be a non-empty array".into())
            })?;

        let mut submodels = Vec::with_capacity(submodels_json.len());
        for (i, entry) in submodels_json.iter().enumerate() {
            let ctx = format!("SlimmableContainer submodel {i}");
            let raw: SubmodelRaw = de::from_value(entry, &ctx)?;
            // The reference calls get_dsp() straight on this object, which
            // skips validate_nam_file and is undefined behaviour when a key is
            // missing. Check what Sequential checks instead.
            let model_json = entry.get("model").filter(|m| is_complete_model(m)).ok_or_else(|| {
                Error::Config(format!(
                    "{ctx}: 'model' must be a complete NAM model with version, architecture, config, and weights"
                ))
            })?;
            submodels.push(Submodel {
                max_value: raw.max_value,
                model: super::parse_value(model_json)?,
            });
        }

        for i in 1..submodels.len() {
            if submodels[i].max_value <= submodels[i - 1].max_value {
                return Err(Error::Config(
                    "ContainerModel: submodels must be sorted by ascending max_value".into(),
                ));
            }
        }
        if submodels.last().unwrap().max_value < 1.0 {
            return Err(Error::Config(
                "ContainerModel: last submodel max_value must be >= 1.0".into(),
            ));
        }
        Ok(ContainerConfig { submodels })
    }

    /// The `ContainerModel` constructor's sample-rate agreement check.
    ///
    /// Separate from parsing because it needs the container's own
    /// `sample_rate`, which lives a level up in the file. A child or a
    /// container that does not state a rate is exempt.
    pub(super) fn check_sample_rates(&self, expected: Option<f64>) -> Result<()> {
        let Some(expected) = expected else {
            return Ok(());
        };
        for sm in &self.submodels {
            if let Some(sr) = sm.model.sample_rate
                && sr != expected
            {
                return Err(Error::Config(format!(
                    "ContainerModel: submodel sample rate mismatch (expected {expected}, got {sr})"
                )));
            }
        }
        Ok(())
    }
}
