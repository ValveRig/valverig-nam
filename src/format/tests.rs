//! Tests for the `.nam` parser.
//!
//! Three kinds:
//!
//! * every real model file is parsed, then cross-checked field by field
//!   against a second, independent walk of the raw JSON tree, consulting no
//!   parser internals, only `serde_json` accessors;
//! * pinned facts (shapes, weight-array checksums) obtained by reading the
//!   files with CPython, whose `float()` and `struct.pack('<f', …)` are an
//!   independent decimal → f64 → f32 implementation;
//! * negative tests, one per validation the reference performs.

use super::*;
use crate::activations::Activation;
use serde_json::json;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Real files
// ---------------------------------------------------------------------------

fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models")
}

fn nam_files_in(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "nam"))
        .collect();
    out.sort();
    out
}

/// Re-derive every parsed field straight from the JSON tree.
///
/// Deliberately written against `serde_json` rather than against anything in
/// this module, so that a defaulting bug shows up as a disagreement instead of
/// being reproduced on both sides.
fn cross_check(parsed: &NamFile, raw: &Value, what: &str) {
    assert_eq!(
        parsed.version,
        raw["version"].as_str().unwrap(),
        "{what}: version"
    );
    let arch = raw["architecture"].as_str().unwrap();
    let variant = match &parsed.config {
        ArchConfig::WaveNet(_) => "WaveNet",
        ArchConfig::Lstm(_) => "LSTM",
        ArchConfig::Dropped(name) => name.as_str(),
        ArchConfig::Sequential(_) => "Sequential",
        ArchConfig::SlimmableContainer(_) => "SlimmableContainer",
    };
    assert_eq!(variant, arch, "{what}: architecture");
    assert_eq!(
        parsed.weights.len(),
        raw["weights"].as_array().unwrap().len(),
        "{what}: weight count"
    );
    let sr = raw
        .get("sample_rate")
        .and_then(Value::as_f64)
        .filter(|r| *r != -1.0);
    assert_eq!(parsed.sample_rate, sr, "{what}: sample_rate");
    assert_eq!(
        parsed.metadata.loudness,
        raw.get("metadata")
            .and_then(|m| m.get("loudness"))
            .and_then(Value::as_f64),
        "{what}: loudness"
    );

    let cfg = &raw["config"];
    match &parsed.config {
        ArchConfig::WaveNet(w) => cross_check_wavenet(w, cfg, what),
        ArchConfig::Lstm(l) => {
            assert_eq!(l.num_layers as u64, cfg["num_layers"].as_u64().unwrap());
            assert_eq!(l.input_size as u64, cfg["input_size"].as_u64().unwrap());
            assert_eq!(l.hidden_size as u64, cfg["hidden_size"].as_u64().unwrap());
            assert_eq!(
                l.in_channels as u64,
                cfg.get("in_channels").and_then(Value::as_u64).unwrap_or(1)
            );
            assert_eq!(
                l.out_channels as u64,
                cfg.get("out_channels").and_then(Value::as_u64).unwrap_or(1)
            );
        }
        ArchConfig::SlimmableContainer(c) => {
            let subs = cfg["submodels"].as_array().unwrap();
            assert_eq!(c.submodels.len(), subs.len(), "{what}: submodel count");
            for (i, (sm, raw_sm)) in c.submodels.iter().zip(subs).enumerate() {
                assert_eq!(sm.max_value, raw_sm["max_value"].as_f64().unwrap());
                cross_check(&sm.model, &raw_sm["model"], &format!("{what}/submodel {i}"));
            }
        }
        ArchConfig::Sequential(s) => {
            let children = cfg["models"].as_array().unwrap();
            assert_eq!(s.models.len(), children.len());
            for (i, (m, raw_m)) in s.models.iter().zip(children).enumerate() {
                cross_check(m, raw_m, &format!("{what}/model {i}"));
            }
        }
        // Nothing to cross-check: a dropped architecture keeps its name and
        // discards the config block.
        ArchConfig::Dropped(_) => {}
    }
}

fn cross_check_wavenet(w: &WaveNetConfig, cfg: &Value, what: &str) {
    assert_eq!(
        w.head_scale,
        cfg["head_scale"].as_f64().unwrap() as f32,
        "{what}: head_scale"
    );
    assert_eq!(
        w.head.is_some(),
        cfg.get("head").is_some_and(|h| !h.is_null()),
        "{what}: head"
    );
    assert_eq!(
        w.in_channels as u64,
        cfg.get("in_channels").and_then(Value::as_u64).unwrap_or(1),
        "{what}: in_channels"
    );

    match cfg.get("condition_dsp").filter(|c| !c.is_null()) {
        Some(cd) => cross_check(
            w.condition_dsp.as_ref().expect("condition_dsp parsed"),
            cd,
            &format!("{what}/condition_dsp"),
        ),
        None => assert!(
            w.condition_dsp.is_none(),
            "{what}: unexpected condition_dsp"
        ),
    }

    let layers = cfg["layers"].as_array().unwrap();
    assert_eq!(
        w.layer_arrays.len(),
        layers.len(),
        "{what}: layer array count"
    );
    for (i, (la, raw_la)) in w.layer_arrays.iter().zip(layers).enumerate() {
        let at = format!("{what}/layer array {i}");
        let u = |k: &str| raw_la[k].as_u64().unwrap() as usize;
        assert_eq!(la.input_size, u("input_size"), "{at}: input_size");
        assert_eq!(
            la.condition_size,
            u("condition_size"),
            "{at}: condition_size"
        );
        assert_eq!(la.channels, u("channels"), "{at}: channels");
        assert_eq!(
            la.bottleneck,
            raw_la
                .get("bottleneck")
                .and_then(Value::as_u64)
                .map_or(la.channels, |b| b as usize),
            "{at}: bottleneck"
        );
        assert_eq!(
            la.groups_input,
            raw_la
                .get("groups_input")
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize,
            "{at}: groups_input"
        );
        assert_eq!(
            la.groups_input_mixin,
            raw_la
                .get("groups_input_mixin")
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize,
            "{at}: groups_input_mixin"
        );

        match raw_la.get("head").filter(|h| !h.is_null()) {
            Some(h) => {
                assert_eq!(
                    la.head_size,
                    h["out_channels"].as_u64().unwrap() as usize,
                    "{at}: head_size"
                );
                assert_eq!(
                    la.head_kernel_size,
                    h["kernel_size"].as_u64().unwrap() as usize,
                    "{at}: head ks"
                );
                assert_eq!(
                    la.head_bias,
                    h["bias"].as_bool().unwrap(),
                    "{at}: head bias"
                );
                assert_eq!(
                    la.head_dilation,
                    h.get("head_dilation").and_then(Value::as_u64).unwrap_or(1) as usize,
                    "{at}: head dilation"
                );
            }
            None => {
                assert_eq!(la.head_size, u("head_size"), "{at}: legacy head_size");
                assert_eq!(la.head_kernel_size, 1, "{at}: legacy head kernel size");
                assert_eq!(la.head_dilation, 1, "{at}: legacy head dilation");
                assert_eq!(
                    la.head_bias,
                    raw_la["head_bias"].as_bool().unwrap(),
                    "{at}: legacy head bias"
                );
            }
        }

        let dilations = raw_la["dilations"].as_array().unwrap();
        assert_eq!(la.layers.len(), dilations.len(), "{at}: layer count");
        for (l, (layer, d)) in la.layers.iter().zip(dilations).enumerate() {
            assert_eq!(
                layer.dilation,
                d.as_u64().unwrap() as usize,
                "{at}/layer {l}: dilation"
            );
            let expected_ks = match raw_la.get("kernel_sizes") {
                Some(ks) => ks[l].as_u64().unwrap(),
                None => raw_la["kernel_size"].as_u64().unwrap(),
            };
            assert_eq!(
                layer.kernel_size as u64, expected_ks,
                "{at}/layer {l}: kernel_size"
            );

            let expected_mode = match raw_la.get("gating_mode") {
                Some(Value::Array(a)) => a[l].as_str().unwrap(),
                Some(v) => v.as_str().unwrap(),
                None => match raw_la.get("gated").and_then(Value::as_bool) {
                    Some(true) => "gated",
                    _ => "none",
                },
            };
            let expected_mode = match expected_mode {
                "gated" => GatingMode::Gated,
                "blended" => GatingMode::Blended,
                _ => GatingMode::None,
            };
            assert_eq!(
                layer.gating_mode, expected_mode,
                "{at}/layer {l}: gating_mode"
            );
            if expected_mode == GatingMode::None {
                assert_eq!(
                    layer.secondary_activation,
                    Activation::Tanh,
                    "{at}/layer {l}: ungated secondary activation"
                );
            }
        }
    }
}

/// Sum of the IEEE-754 bit patterns of every weight in the file, including
/// nested models, modulo 2^32.
fn weight_bitsum(f: &NamFile) -> (usize, u32) {
    fn walk(f: &NamFile, count: &mut usize, acc: &mut u32) {
        *count += f.weights.len();
        for w in &f.weights {
            *acc = acc.wrapping_add(w.to_bits());
        }
        match &f.config {
            ArchConfig::WaveNet(w) => {
                if let Some(cd) = &w.condition_dsp {
                    walk(cd, count, acc);
                }
            }
            ArchConfig::SlimmableContainer(c) => {
                for sm in &c.submodels {
                    walk(&sm.model, count, acc);
                }
            }
            ArchConfig::Sequential(s) => {
                for m in &s.models {
                    walk(m, count, acc);
                }
            }
            _ => {}
        }
    }
    let (mut count, mut acc) = (0usize, 0u32);
    walk(f, &mut count, &mut acc);
    (count, acc)
}

#[test]
fn every_bundled_model_parses_and_agrees_with_its_json() {
    let dir = models_dir();
    let files = nam_files_in(&dir);
    assert_eq!(
        files.len(),
        12,
        "expected the bundled model set, found {files:?}"
    );
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(path).unwrap();
        let parsed = parse_json(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            parsed,
            load_file(path).unwrap(),
            "{name}: load_file differs from parse_json"
        );
        let raw: Value = serde_json::from_str(&text).unwrap();
        cross_check(&parsed, &raw, &name);
    }
}

/// The upstream `example_models/` directory, when a checkout is around.
///
/// `assets/models/` holds eight of its files; this also covers
/// `my_model.nam`, which is not committed to this crate. Point
/// `NAM_REF_EXAMPLE_MODELS` at `<NeuralAmpModelerCore>/example_models` to
/// run it.
#[test]
fn upstream_example_models_parse() {
    let Some(dir) = std::env::var_os("NAM_REF_EXAMPLE_MODELS").map(PathBuf::from) else {
        eprintln!("NAM_REF_EXAMPLE_MODELS not set; skipping upstream example models");
        return;
    };
    let files = nam_files_in(&dir);
    assert!(!files.is_empty(), "no .nam files in {}", dir.display());
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(path).unwrap();
        let parsed = parse_json(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        let raw: Value = serde_json::from_str(&text).unwrap();
        cross_check(&parsed, &raw, &name);
    }
}

#[test]
fn bundled_model_shapes_are_what_the_json_says() {
    // (file, version, sample_rate, top-level weight count, recursive weight
    //  count, recursive sum of f32 bit patterns).
    //
    // Produced by reading each file with CPython:
    //   struct.unpack('<I', struct.pack('<f', w))[0], summed mod 2**32.
    // Python's decimal->double conversion is correctly rounded and its
    // double->float narrowing is round-to-nearest-even, the same pair of
    // roundings the reference performs, so this pins the weight decoding
    // against an implementation that shares no code with this crate.
    type Pinned = (&'static str, &'static str, Option<f64>, usize, usize, u32);
    let expected: [Pinned; 7] = [
        ("lstm.nam", "0.5.4", Some(48000.0), 70, 70, 0x9E3A_77FB),
        (
            "slimmable_container.nam",
            "0.7.0",
            Some(48000.0),
            0,
            1043,
            0x2E33_F4F7,
        ),
        (
            "slimmable_wavenet.nam",
            "0.7.0",
            Some(48000.0),
            457,
            457,
            0x997C_156A,
        ),
        ("wavenet.nam", "0.5.4", Some(48000.0), 131, 131, 0x7EF3_A9B5),
        (
            "wavenet_a1_standard.nam",
            "0.5.0",
            None,
            13802,
            13802,
            0xEC2E_B2CD,
        ),
        (
            "wavenet_a2_feature_test.nam",
            "0.6.0",
            Some(48000.0),
            818,
            1870,
            0xEC94_6EE7,
        ),
        (
            "wavenet_condition_dsp.nam",
            "0.6.0",
            Some(48000.0),
            147,
            284,
            0x1074_B9EE,
        ),
    ];
    let dir = models_dir();
    for (file, version, sample_rate, top_weights, all_weights, bitsum) in expected {
        let f = load_file(dir.join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(f.version, version, "{file}: version");
        assert_eq!(f.sample_rate, sample_rate, "{file}: sample_rate");
        assert_eq!(
            f.weights.len(),
            top_weights,
            "{file}: top-level weight count"
        );
        assert_eq!(
            weight_bitsum(&f),
            (all_weights, bitsum),
            "{file}: recursive weight bits"
        );
    }
}

#[test]
fn wavenet_weights_round_exactly_as_python_does() {
    // Individual weights of wavenet.nam, with the f32 bits CPython produces.
    // A double-rounding bug (parsing to f32 directly, say) moves these.
    let expected: [(usize, u32); 8] = [
        (0, 0xBF1E_367B),
        (1, 0x3F84_04FF),
        (2, 0xBF81_6D90),
        (3, 0xBEC4_ECF1),
        (32, 0x394B_64C7),
        (48, 0xBD68_9B5D),
        (109, 0xBD1C_23DF),
        (130, 0x3CA3_D70A),
    ];
    let f = load_file(models_dir().join("wavenet.nam")).unwrap();
    for (i, bits) in expected {
        assert_eq!(
            f.weights[i].to_bits(),
            bits,
            "weights[{i}] = {}",
            f.weights[i]
        );
    }
}

#[test]
fn feature_test_model_exercises_every_wavenet_field() {
    // wavenet_a2_feature_test.nam is upstream's kitchen sink. Reading a few
    // specific values out of it checks that the exotic branches actually fire
    // rather than silently falling back to a default.
    let f = load_file(models_dir().join("wavenet_a2_feature_test.nam")).unwrap();
    let ArchConfig::WaveNet(w) = &f.config else {
        panic!("expected a WaveNet")
    };

    let cond = w.condition_dsp.as_ref().expect("condition_dsp");
    assert_eq!(cond.sample_rate, f.sample_rate);
    let ArchConfig::WaveNet(cw) = &cond.config else {
        panic!("expected a WaveNet condition DSP")
    };

    // Array 0 of the condition DSP: bottleneck != channels, grouped input
    // convolution, an active head1x1, gated with a named secondary activation,
    // and all eight FiLM sites on.
    let la = &cw.layer_arrays[0];
    assert_eq!((la.channels, la.bottleneck), (3, 6));
    assert_eq!(la.groups_input, 3);
    assert_eq!(
        la.head1x1,
        Head1x1Config {
            active: true,
            out_channels: 6,
            groups: 3
        }
    );
    assert_eq!(
        la.layer1x1,
        Layer1x1Config {
            active: true,
            groups: 3
        }
    );
    for l in &la.layers {
        assert_eq!(l.gating_mode, GatingMode::Gated);
        assert_eq!(l.activation, Activation::SiLU);
        assert_eq!(l.secondary_activation, Activation::Hardswish);
    }
    let on = FilmConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    assert_eq!(
        [
            la.films[film_site::CONV_PRE],
            la.films[film_site::CONV_POST],
            la.films[film_site::INPUT_MIXIN_PRE],
            la.films[film_site::INPUT_MIXIN_POST],
            la.films[film_site::ACTIVATION_PRE],
            la.films[film_site::ACTIVATION_POST],
            la.films[film_site::LAYER1X1_POST],
            la.films[film_site::HEAD1X1_POST],
        ],
        [on; 8]
    );

    // Array 1: per-layer activations, per-layer gating, per-layer secondary
    // activations, and shift = false on every FiLM site.
    let la = &cw.layer_arrays[1];
    assert_eq!(la.layers.len(), 3);
    assert_eq!(
        la.layers[0].activation,
        Activation::PReLU {
            negative_slopes: vec![0.04, 0.05]
        }
    );
    assert_eq!(la.layers[2].activation, Activation::Softsign);
    assert_eq!(
        [
            la.layers[0].gating_mode,
            la.layers[1].gating_mode,
            la.layers[2].gating_mode
        ],
        [GatingMode::Blended, GatingMode::Gated, GatingMode::Gated]
    );
    assert_eq!(
        la.layers[0].secondary_activation,
        Activation::LeakyHardtanh {
            min_val: 0.0,
            max_val: 0.9,
            min_slope: 0.0,
            max_slope: 0.02
        }
    );
    assert_eq!(la.layers[1].secondary_activation, Activation::ReLU);
    assert_eq!(la.layers[2].secondary_activation, Activation::Sigmoid);
    assert!(!la.films[film_site::CONV_PRE].shift && la.films[film_site::CONV_PRE].active);

    // The outer model's layer array uses per-layer kernel sizes via the
    // nested head object and grouped FiLM.
    let la = &w.layer_arrays[0];
    assert_eq!(la.condition_size, 8);
    assert_eq!(la.films[film_site::LAYER1X1_POST].groups, 8);
    assert_eq!(la.groups_input_mixin, 4);
}

#[test]
fn a2_ships_null_secondary_activations_next_to_ungated_layers() {
    // A2.nam carries "secondary_activation": [null, null, ...] alongside an
    // all-"none" gating_mode array. The reference never parses those nulls
    // because the gating branch short-circuits, and neither may this parser.
    let cfg = json!({
        "version": "0.7.0", "architecture": "WaveNet", "weights": [], "sample_rate": 48000,
        "config": {
            "layers": [{
                "input_size": 1, "condition_size": 1, "channels": 3,
                "head": {"out_channels": 1, "kernel_size": 16, "bias": true},
                "kernel_sizes": [1, 2], "dilations": [1, 2],
                "activation": "ReLU",
                "gating_mode": ["none", "none"],
                "secondary_activation": [null, null]
            }],
            "head": null, "head_scale": 0.01
        }
    });
    let f = parse_value(&cfg).unwrap();
    let ArchConfig::WaveNet(w) = &f.config else {
        panic!()
    };
    assert!(
        w.layer_arrays[0]
            .layers
            .iter()
            .all(|l| l.gating_mode == GatingMode::None)
    );
}

#[test]
fn a_slimmable_file_is_read_at_full_width() {
    let f = load_file(models_dir().join("slimmable_wavenet.nam")).unwrap();
    let ArchConfig::WaveNet(w) = &f.config else {
        panic!()
    };
    // An ordinary WaveNet, at the channel counts the file states. The
    // narrower widths the `slimmable` block offers are not built; full width
    // is where the reference starts and where the reference vectors are
    // recorded.
    let raw = std::fs::read_to_string(models_dir().join("slimmable_wavenet.nam")).unwrap();
    let raw: Value = serde_json::from_str(&raw).unwrap();
    for (i, la) in w.layer_arrays.iter().enumerate() {
        assert_eq!(
            la.channels,
            raw["config"]["layers"][i]["channels"].as_u64().unwrap() as usize
        );
    }
}

#[test]
fn a_slimmable_block_needs_nothing_but_its_method() {
    // The reference reads `kwargs.allowed_channels` here and falls back to
    // 1..=channels; neither width is built, so a method on its own must not
    // be an error.
    let mut cfg = json!({
        "version": "0.7.0", "architecture": "WaveNet", "weights": [], "sample_rate": 48000,
        "config": {
            "layers": [legacy_layer(4)],
            "head": null, "head_scale": 0.02
        }
    });
    cfg["config"]["layers"][0]["slimmable"] = json!({"method": "slice_channels_uniform"});
    let f = parse_value(&cfg).unwrap();
    let ArchConfig::WaveNet(w) = &f.config else {
        panic!()
    };
    assert_eq!(w.layer_arrays[0].channels, 4);
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

/// A pre-0.6 layer array: no bottleneck, no groups, no FiLM, no head object.
fn legacy_layer(channels: u64) -> Value {
    json!({
        "input_size": 1, "condition_size": 1, "head_size": 2, "channels": channels,
        "kernel_size": 3, "dilations": [1, 2], "activation": "Tanh",
        "gated": false, "head_bias": false
    })
}

fn legacy_wavenet() -> Value {
    json!({
        "version": "0.5.4", "architecture": "WaveNet", "weights": [], "sample_rate": 48000,
        "config": {"layers": [legacy_layer(3)], "head": null, "head_scale": 0.02}
    })
}

fn wavenet_with_layer(layer: Value) -> Value {
    let mut v = legacy_wavenet();
    v["config"]["layers"] = json!([layer]);
    v
}

fn err_of(v: &Value) -> String {
    match parse_value(v) {
        Ok(_) => panic!("expected an error, parsed successfully"),
        Err(e) => e.to_string(),
    }
}

/// The reference converts a key only on the branch that reads it, so a key
/// the branch it belongs to never takes may hold anything, `null` included.
/// Trainers do ship such leftovers (`A2.nam` carries `"slimmable": null`), and
/// reading the whole object up front would refuse files the reference loads.
#[test]
fn a_key_the_taken_branch_never_reads_is_never_converted() {
    // `head` wins, so a stale legacy pair beside it is not looked at.
    let mut layer = legacy_layer(3);
    layer["head"] = json!({"out_channels": 2, "kernel_size": 1, "bias": false});
    layer["head_size"] = json!(null);
    layer["head_bias"] = json!(null);
    let f = parse_value(&wavenet_with_layer(layer)).unwrap();
    let ArchConfig::WaveNet(w) = &f.config else {
        panic!()
    };
    assert_eq!(w.layer_arrays[0].head_size, 2);

    // ...and an activation never reads another activation's parameters.
    let mut layer = legacy_layer(3);
    layer["activation"] = json!({"type": "Tanh", "negative_slope": null, "min_val": null});
    let f = parse_value(&wavenet_with_layer(layer)).unwrap();
    let ArchConfig::WaveNet(w) = &f.config else {
        panic!()
    };
    assert_eq!(w.layer_arrays[0].layers[0].activation, Activation::Tanh);

    // The same key on the branch that *does* read it is still a type error.
    // Serde does not name the field it choked on, so the message says where
    // in the tree it was and what it wanted; only `head_size`, read on its
    // own, can name itself.
    let mut layer = legacy_layer(3);
    layer["activation"] = json!({"type": "LeakyReLU", "negative_slope": null});
    let msg = err_of(&wavenet_with_layer(layer));
    assert!(msg.contains("activation") && msg.contains("null"), "{msg}");

    let mut layer = legacy_layer(3);
    layer["head_size"] = json!(null);
    assert!(err_of(&wavenet_with_layer(layer)).contains("head_size"));
}

#[test]
fn a_legacy_layer_array_gets_the_reference_defaults() {
    let f = parse_value(&legacy_wavenet()).unwrap();
    let ArchConfig::WaveNet(w) = &f.config else {
        panic!()
    };
    assert!(w.head.is_none());
    assert_eq!(w.in_channels, 1);
    assert_eq!(w.head_scale, 0.02f64 as f32);

    let la = &w.layer_arrays[0];
    assert_eq!(
        la.bottleneck, la.channels,
        "bottleneck defaults to channels"
    );
    assert_eq!(la.groups_input, 1);
    assert_eq!(la.groups_input_mixin, 1);
    assert_eq!(
        la.layer1x1,
        Layer1x1Config {
            active: true,
            groups: 1
        }
    );
    assert_eq!(
        la.head1x1,
        Head1x1Config {
            active: false,
            out_channels: 3,
            groups: 1
        }
    );
    assert_eq!(
        (
            la.head_size,
            la.head_kernel_size,
            la.head_dilation,
            la.head_bias
        ),
        (2, 1, 1, false)
    );
    let off = FilmConfig {
        active: false,
        shift: false,
        groups: 1,
    };
    assert_eq!(la.films[film_site::CONV_PRE], off);
    assert_eq!(la.films[film_site::HEAD1X1_POST], off);
    assert_eq!(la.layers.len(), 2);
    for l in &la.layers {
        assert_eq!(l.kernel_size, 3);
        assert_eq!(l.activation, Activation::Tanh);
        assert_eq!(l.gating_mode, GatingMode::None);
        assert_eq!(l.secondary_activation, Activation::Tanh);
    }
}

#[test]
fn gating_defaults_the_secondary_activation_to_sigmoid() {
    for gating in [json!(true), json!("gated"), json!(["gated", "blended"])] {
        let mut layer = legacy_layer(3);
        if gating == json!(true) {
            layer["gated"] = json!(true);
        } else {
            layer.as_object_mut().unwrap().remove("gated");
            layer["gating_mode"] = gating.clone();
        }
        let f = parse_value(&wavenet_with_layer(layer)).unwrap();
        let ArchConfig::WaveNet(w) = &f.config else {
            panic!()
        };
        for l in &w.layer_arrays[0].layers {
            assert_ne!(l.gating_mode, GatingMode::None, "{gating}");
            assert_eq!(l.secondary_activation, Activation::Sigmoid, "{gating}");
        }
    }
}

#[test]
fn film_false_is_shorthand_for_inactive() {
    let mut layer = legacy_layer(3);
    layer["conv_pre_film"] = json!(false);
    layer["conv_post_film"] = json!({"active": true});
    let f = parse_value(&wavenet_with_layer(layer)).unwrap();
    let ArchConfig::WaveNet(w) = &f.config else {
        panic!()
    };
    let la = &w.layer_arrays[0];
    assert_eq!(
        la.films[film_site::CONV_PRE],
        FilmConfig {
            active: false,
            shift: false,
            groups: 1
        }
    );
    // An object supplies only what it names; shift and groups default on.
    assert_eq!(
        la.films[film_site::CONV_POST],
        FilmConfig {
            active: true,
            shift: true,
            groups: 1
        }
    );
}

#[test]
fn an_inactive_film_object_keeps_its_shift_flag() {
    // A2.nam writes {"active": false, "shift": true, "groups": 1}. The
    // reference stores shift as given, so a later reactivation of the site
    // would not silently change the weight layout.
    let mut layer = legacy_layer(3);
    layer["activation_post_film"] = json!({"active": false, "shift": true, "groups": 2});
    let f = parse_value(&wavenet_with_layer(layer)).unwrap();
    let ArchConfig::WaveNet(w) = &f.config else {
        panic!()
    };
    assert_eq!(
        w.layer_arrays[0].films[film_site::ACTIVATION_POST],
        FilmConfig {
            active: false,
            shift: true,
            groups: 2
        }
    );
}

#[test]
fn activation_objects_get_the_references_parameter_defaults() {
    let cases: [(Value, Activation); 8] = [
        (json!("Tanh"), Activation::Tanh),
        (
            json!({"type": "LeakyReLU"}),
            Activation::LeakyReLU {
                negative_slope: 0.01,
            },
        ),
        (
            json!({"type": "LeakyReLU", "negative_slope": 0.2}),
            Activation::LeakyReLU {
                negative_slope: 0.2,
            },
        ),
        (
            json!({"type": "PReLU"}),
            Activation::PReLU {
                negative_slopes: vec![0.01],
            },
        ),
        (
            json!({"type": "PReLU", "negative_slope": 0.3}),
            Activation::PReLU {
                negative_slopes: vec![0.3],
            },
        ),
        (
            json!({"type": "PReLU", "negative_slopes": [0.1, 0.2]}),
            Activation::PReLU {
                negative_slopes: vec![0.1, 0.2],
            },
        ),
        (
            json!({"type": "LeakyHardtanh"}),
            Activation::LeakyHardtanh {
                min_val: -1.0,
                max_val: 1.0,
                min_slope: 0.01,
                max_slope: 0.01,
            },
        ),
        // Both casings map to the same type, as the reference's duplicate
        // type_map entry does.
        (
            json!({"type": "LeakyHardTanh", "max_val": 0.5}),
            Activation::LeakyHardtanh {
                min_val: -1.0,
                max_val: 0.5,
                min_slope: 0.01,
                max_slope: 0.01,
            },
        ),
    ];
    for (j, expected) in cases {
        assert_eq!(activation_from_json(&j).unwrap(), expected, "{j}");
    }
}

#[test]
fn activation_rejects_what_the_reference_rejects() {
    assert!(
        activation_from_json(&json!(null))
            .unwrap_err()
            .to_string()
            .contains("expected string or object")
    );
    assert!(
        activation_from_json(&json!([1, 2]))
            .unwrap_err()
            .to_string()
            .contains("expected string or object")
    );
    assert!(
        activation_from_json(&json!("Bogus"))
            .unwrap_err()
            .to_string()
            .contains("Unknown activation type: Bogus")
    );
    assert!(
        activation_from_json(&json!({"type": "Bogus"}))
            .unwrap_err()
            .to_string()
            .contains("Unknown activation type")
    );
}

// ---------------------------------------------------------------------------
// Negative tests
// ---------------------------------------------------------------------------

#[test]
fn root_must_be_an_object_with_the_four_required_keys() {
    assert!(
        parse_json("[]")
            .unwrap_err()
            .to_string()
            .contains("must be an object")
    );
    assert!(
        parse_json("3")
            .unwrap_err()
            .to_string()
            .contains("must be an object")
    );
    assert!(matches!(parse_json("{"), Err(Error::Json(_))));
    for key in ["version", "architecture", "config", "weights"] {
        let mut v = legacy_wavenet();
        v.as_object_mut().unwrap().remove(key);
        let msg = err_of(&v);
        assert!(
            msg.contains(&format!("missing required key \"{key}\"")),
            "{key}: {msg}"
        );
    }
}

#[test]
fn version_range_is_enforced_at_the_file_level() {
    for bad in ["0.4.9", "0.8.0", "1.0.0", "0.5", "nope"] {
        let mut v = legacy_wavenet();
        v["version"] = json!(bad);
        assert!(
            matches!(parse_value(&v), Err(Error::UnsupportedVersion { .. })),
            "version {bad} should be rejected"
        );
    }
    // A patch beyond the latest is partial support, and still loads.
    let mut v = legacy_wavenet();
    v["version"] = json!("0.7.9");
    assert!(parse_value(&v).is_ok());
}

#[test]
fn unknown_architecture_is_reported_by_name() {
    let mut v = legacy_wavenet();
    v["architecture"] = json!("Fnord");
    assert!(matches!(parse_value(&v), Err(Error::UnsupportedArchitecture(a)) if a == "Fnord"));
}

#[test]
fn weights_must_be_an_array_of_numbers() {
    let mut v = legacy_wavenet();
    v["weights"] = json!("nope");
    assert!(err_of(&v).contains("weights must be an array"));
    v["weights"] = json!([1.0, null]);
    assert!(err_of(&v).contains("weights[1]"));
}

#[test]
fn kernel_size_and_kernel_sizes_are_exclusive() {
    let mut layer = legacy_layer(3);
    layer["kernel_sizes"] = json!([3, 3]);
    assert!(
        err_of(&wavenet_with_layer(layer.clone()))
            .contains("only one of kernel_size (int) or kernel_sizes (array) may be provided")
    );

    layer.as_object_mut().unwrap().remove("kernel_size");
    layer.as_object_mut().unwrap().remove("kernel_sizes");
    assert!(
        err_of(&wavenet_with_layer(layer.clone()))
            .contains("either kernel_size (int) or kernel_sizes (array) must be provided")
    );

    layer["kernel_sizes"] = json!([3]);
    assert!(
        err_of(&wavenet_with_layer(layer))
            .contains("kernel_sizes array size (1) must match dilations size (2)")
    );
}

#[test]
fn per_layer_array_lengths_must_match_the_dilations() {
    let mut layer = legacy_layer(3);
    layer["activation"] = json!(["Tanh"]);
    assert!(
        err_of(&wavenet_with_layer(layer))
            .contains("activation array size (1) must match dilations size (2)")
    );

    let mut layer = legacy_layer(3);
    layer.as_object_mut().unwrap().remove("gated");
    layer["gating_mode"] = json!(["none"]);
    assert!(
        err_of(&wavenet_with_layer(layer))
            .contains("gating_mode array size (1) must match dilations size (2)")
    );

    let mut layer = legacy_layer(3);
    layer.as_object_mut().unwrap().remove("gated");
    layer["gating_mode"] = json!(["none", "none"]);
    layer["secondary_activation"] = json!(["Sigmoid"]);
    assert!(
        err_of(&wavenet_with_layer(layer))
            .contains("secondary_activation array size (1) must match dilations size (2)")
    );

    // The "at least" message fires earlier, while walking the gating array,
    // when a gated layer would index past the secondary array.
    let mut layer = legacy_layer(3);
    layer.as_object_mut().unwrap().remove("gated");
    layer["gating_mode"] = json!(["none", "gated"]);
    layer["secondary_activation"] = json!(["Sigmoid"]);
    assert!(
        err_of(&wavenet_with_layer(layer))
            .contains("secondary_activation array size must be at least 2")
    );
}

#[test]
fn gating_mode_names_are_checked() {
    let mut layer = legacy_layer(3);
    layer.as_object_mut().unwrap().remove("gated");
    layer["gating_mode"] = json!("sideways");
    assert!(err_of(&wavenet_with_layer(layer)).contains("Invalid gating_mode: sideways"));
}

#[test]
fn a_head_is_required_in_one_of_its_two_spellings() {
    let mut layer = legacy_layer(3);
    layer.as_object_mut().unwrap().remove("head_size");
    let msg = err_of(&wavenet_with_layer(layer.clone()));
    assert!(
        msg.contains("expected 'head' object with out_channels, kernel_size, and bias"),
        "{msg}"
    );

    layer["head"] = json!(7);
    assert!(err_of(&wavenet_with_layer(layer.clone())).contains("'head' must be a JSON object"));

    layer["head"] = json!({"out_channels": 2, "bias": false});
    assert!(err_of(&wavenet_with_layer(layer.clone())).contains("kernel_size"));

    layer["head"] = json!({"out_channels": 2, "kernel_size": 0, "bias": false});
    assert!(err_of(&wavenet_with_layer(layer)).contains("head.kernel_size must be >= 1"));
}

#[test]
fn layer1x1_post_film_requires_layer1x1() {
    let mut layer = legacy_layer(3);
    layer["layer1x1"] = json!({"active": false, "groups": 1});
    layer["layer1x1_post_film"] = json!({"active": true, "shift": true, "groups": 1});
    assert!(
        err_of(&wavenet_with_layer(layer))
            .contains("layer1x1_post_film cannot be active when layer1x1.active is false")
    );
}

#[test]
fn wavenet_needs_a_head_scale_and_at_least_one_layer_array() {
    let mut v = legacy_wavenet();
    v["config"].as_object_mut().unwrap().remove("head_scale");
    assert!(err_of(&v).contains("head_scale"));

    let mut v = legacy_wavenet();
    v["config"]["layers"] = json!([]);
    assert!(err_of(&v).contains("WaveNet config requires at least one layer array"));

    // A null "layers" has size 0 in nlohmann, so it lands on the same error.
    let mut v = legacy_wavenet();
    v["config"]["layers"] = json!(null);
    assert!(err_of(&v).contains("WaveNet config requires at least one layer array"));
}

#[test]
fn post_stack_head_is_checked_against_the_last_layer_arrays_head_size() {
    let mut v = legacy_wavenet();
    v["config"]["head"] = json!({
        "in_channels": 5, "channels": 4, "out_channels": 1,
        "kernel_sizes": [3], "activation": "Tanh"
    });
    assert!(err_of(&v).contains("head.in_channels (5) must equal last layer's head_size (2)"));

    // in_channels matching, or absent, both work; it is always taken from the
    // last layer array.
    v["config"]["head"]["in_channels"] = json!(2);
    let f = parse_value(&v).unwrap();
    let ArchConfig::WaveNet(w) = &f.config else {
        panic!()
    };
    assert_eq!(w.head.as_ref().unwrap().in_channels, 2);

    v["config"]["head"] = json!({
        "channels": 4, "out_channels": 1, "kernel_sizes": [], "activation": "Tanh"
    });
    assert!(err_of(&v).contains("head.kernel_sizes must be non-empty"));
}

#[test]
fn slimmable_method_must_be_the_one_the_reference_implements() {
    let mut layer = legacy_layer(3);
    layer["slimmable"] = json!({"method": "magic"});
    assert!(
        err_of(&wavenet_with_layer(layer))
            .contains("SlimmableWavenet: unsupported slimmable method 'magic'")
    );

    // A null or non-object slimmable field is ignored, which is how A2.nam
    // ships "slimmable": null on a non-slimmable model.
    let mut layer = legacy_layer(3);
    layer["slimmable"] = json!(null);
    assert!(parse_value(&wavenet_with_layer(layer)).is_ok());
}

#[test]
fn condition_dsp_sample_rate_must_match() {
    let mut v = legacy_wavenet();
    let mut cond = legacy_wavenet();
    cond["sample_rate"] = json!(44100);
    v["config"]["condition_dsp"] = cond;
    assert!(err_of(&v).contains("Condition DSP expected sample rate"));

    let mut v = legacy_wavenet();
    v["config"]["condition_dsp"] = legacy_wavenet();
    assert!(parse_value(&v).is_ok());
}

#[test]
fn lstm_linear_and_convnet_defaults_and_requirements() {
    let lstm = json!({
        "version": "0.5.4", "architecture": "LSTM", "weights": [], "sample_rate": 48000,
        "config": {"num_layers": 1, "input_size": 1, "hidden_size": 3}
    });
    let f = parse_value(&lstm).unwrap();
    assert_eq!(
        f.config,
        ArchConfig::Lstm(LstmConfig {
            num_layers: 1,
            input_size: 1,
            hidden_size: 3,
            in_channels: 1,
            out_channels: 1
        })
    );
    let mut bad = lstm.clone();
    bad["config"].as_object_mut().unwrap().remove("hidden_size");
    assert!(err_of(&bad).contains("hidden_size"));

    // A dropped architecture is recognised by name; its config block is not
    // parsed, so even a nonsense one gets past the format layer and is
    // refused by the engine factory instead, which names the real reason.
    for arch in ["Linear", "ConvNet"] {
        let f = parse_value(&json!({
            "version": "0.5.4", "architecture": arch, "weights": [], "sample_rate": 48000,
            "config": {"nonsense": true}
        }))
        .unwrap();
        assert_eq!(f.config, ArchConfig::Dropped(arch.to_string()));
    }
}

#[test]
fn sequential_requires_empty_weights_and_complete_children() {
    let child = legacy_wavenet();
    let mut v = json!({
        "version": "0.7.0", "architecture": "Sequential", "weights": [], "sample_rate": 48000,
        "config": {"models": [child.clone(), child.clone()]}
    });
    let f = parse_value(&v).unwrap();
    let ArchConfig::Sequential(s) = &f.config else {
        panic!()
    };
    assert_eq!(s.models.len(), 2);

    let mut bad = v.clone();
    bad["weights"] = json!([1.0]);
    assert!(err_of(&bad).contains("top-level weights must be empty"));

    let mut bad = v.clone();
    bad["config"] = json!({});
    assert!(err_of(&bad).contains("config must contain a 'models' array"));

    let mut bad = v.clone();
    bad["config"]["models"] = json!([]);
    assert!(err_of(&bad).contains("'models' must be a non-empty array"));

    v["config"]["models"] = json!([{"version": "0.5.4", "architecture": "LSTM"}]);
    assert!(err_of(&v).contains("each child must be a complete NAM model"));
}

#[test]
fn container_submodels_must_be_ordered_and_cover_one() {
    let sub = |max: f64, sr: f64| {
        json!({"max_value": max, "model": {
            "version": "0.5.4", "architecture": "LSTM", "weights": [], "sample_rate": sr,
            "config": {"num_layers": 1, "input_size": 1, "hidden_size": 3}
        }})
    };
    let container = |subs: Value| {
        json!({
            "version": "0.7.0", "architecture": "SlimmableContainer", "weights": [],
            "sample_rate": 48000, "config": {"submodels": subs}
        })
    };

    let ok = container(json!([sub(0.5, 48000.0), sub(1.0, 48000.0)]));
    let f = parse_value(&ok).unwrap();
    let ArchConfig::SlimmableContainer(c) = &f.config else {
        panic!()
    };
    assert_eq!(c.submodels.len(), 2);
    assert_eq!(c.submodels[0].max_value, 0.5);

    assert!(err_of(&container(json!([]))).contains("'submodels' must be a non-empty array"));
    assert!(
        err_of(&container(json!([sub(1.0, 48000.0), sub(0.5, 48000.0)])))
            .contains("submodels must be sorted by ascending max_value")
    );
    assert!(
        err_of(&container(json!([sub(0.5, 48000.0), sub(0.5, 48000.0)])))
            .contains("submodels must be sorted by ascending max_value")
    );
    assert!(
        err_of(&container(json!([sub(0.5, 48000.0), sub(0.9, 48000.0)])))
            .contains("last submodel max_value must be >= 1.0")
    );
    assert!(
        err_of(&container(json!([sub(0.5, 44100.0), sub(1.0, 48000.0)])))
            .contains("submodel sample rate mismatch")
    );
    // An unknown (-1.0) rate on either side is exempt.
    assert!(parse_value(&container(json!([sub(0.5, -1.0), sub(1.0, 48000.0)]))).is_ok());
}

#[test]
fn sequential_children_must_agree_on_a_sample_rate() {
    let child = |sr: f64| {
        json!({
            "version": "0.5.4", "architecture": "LSTM", "weights": [], "sample_rate": sr,
            "config": {"num_layers": 1, "input_size": 1, "hidden_size": 3}
        })
    };
    let seq = |children: Value, sr: f64| {
        json!({
            "version": "0.7.0", "architecture": "Sequential", "weights": [],
            "sample_rate": sr, "config": {"models": children}
        })
    };
    assert!(parse_value(&seq(json!([child(48000.0), child(48000.0)]), 48000.0)).is_ok());
    // An unknown rate anywhere is exempt; the first known one becomes the
    // expectation for the rest.
    assert!(parse_value(&seq(json!([child(-1.0), child(48000.0)]), -1.0)).is_ok());
    assert!(
        err_of(&seq(json!([child(44100.0), child(48000.0)]), -1.0))
            .contains("submodel sample rate mismatch")
    );
    assert!(
        err_of(&seq(json!([child(44100.0)]), 48000.0)).contains("submodel sample rate mismatch")
    );
}

#[test]
fn metadata_is_read_the_way_the_reference_reads_it() {
    let mut v = legacy_wavenet();
    v["metadata"] = json!({
        "loudness": -20.5, "gain": 0.19, "input_level_dbu": 18.3,
        "output_level_dbu": 12.3, "name": "Test Model", "gear_make": "Acme"
    });
    let f = parse_value(&v).unwrap();
    let m = &f.metadata;
    assert_eq!(m.loudness, Some(-20.5));
    assert_eq!(m.gain, Some(0.19));
    assert_eq!(m.input_level_dbu, Some(18.3));
    assert_eq!(m.output_level_dbu, Some(12.3));
    assert_eq!(m.name.as_deref(), Some("Test Model"));
    assert_eq!(m.raw["gear_make"], json!("Acme"));

    // A null value is "not present", exactly as the reference's extract()
    // treats it.
    v["metadata"] = json!({"loudness": null});
    assert_eq!(parse_value(&v).unwrap().metadata.loudness, None);

    // A non-numeric loudness is a type error there, so it is one here.
    v["metadata"] = json!({"loudness": "loud"});
    assert!(err_of(&v).contains("metadata.loudness"));

    // No metadata at all.
    let f = parse_value(&legacy_wavenet()).unwrap();
    assert_eq!(f.metadata, Metadata::default());
}

#[test]
fn an_absent_or_sentinel_sample_rate_is_unknown() {
    let mut v = legacy_wavenet();
    v.as_object_mut().unwrap().remove("sample_rate");
    assert_eq!(parse_value(&v).unwrap().sample_rate, None);
    v["sample_rate"] = json!(-1);
    assert_eq!(parse_value(&v).unwrap().sample_rate, None);
    v["sample_rate"] = json!(44100);
    assert_eq!(parse_value(&v).unwrap().sample_rate, Some(44100.0));
}

#[test]
fn a_sample_rate_must_be_one_a_host_could_run_at() {
    for bad in [json!(0), json!(-48000), json!(1e30), json!(f64::NAN)] {
        let mut v = legacy_wavenet();
        v["sample_rate"] = bad.clone();
        if bad.is_null() {
            continue; // serde_json has no NaN literal; the value above serialised to null
        }
        assert!(err_of(&v).contains("sample_rate must be"), "{bad}");
    }
}

#[test]
fn load_file_reports_a_missing_path() {
    let e = load_file("/nonexistent/nowhere.nam")
        .unwrap_err()
        .to_string();
    assert!(e.contains("file does not exist"), "{e}");
}

/// [`FILM_KEYS`] and [`film_site`] are two orderings of the same eight sites,
/// and nothing but this test makes them agree.
///
/// Getting it wrong is silent and severe: every FiLM site would still parse
/// and still build, with scale/shift parameters wired to the wrong point in
/// the layer. Cheap to pin, so pin it.
#[test]
fn film_keys_are_in_site_order() {
    assert_eq!(FILM_KEYS[film_site::CONV_PRE], "conv_pre_film");
    assert_eq!(FILM_KEYS[film_site::CONV_POST], "conv_post_film");
    assert_eq!(
        FILM_KEYS[film_site::INPUT_MIXIN_PRE],
        "input_mixin_pre_film"
    );
    assert_eq!(
        FILM_KEYS[film_site::INPUT_MIXIN_POST],
        "input_mixin_post_film"
    );
    assert_eq!(FILM_KEYS[film_site::ACTIVATION_PRE], "activation_pre_film");
    assert_eq!(
        FILM_KEYS[film_site::ACTIVATION_POST],
        "activation_post_film"
    );
    assert_eq!(FILM_KEYS[film_site::LAYER1X1_POST], "layer1x1_post_film");
    assert_eq!(FILM_KEYS[film_site::HEAD1X1_POST], "head1x1_post_film");
}
