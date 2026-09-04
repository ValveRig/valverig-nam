//! nlohmann's conversion rules, expressed as serde adapters.
//!
//! The reference reads its configuration through `nlohmann::json`, and the
//! exact conversion rules are part of the file format. Serde's own rules are
//! close but not identical, so the config structs derive `Deserialize` and
//! reach the reference's semantics through the helpers here:
//!
//! | reference | here |
//! |---|---|
//! | `j.at(key)` / const `j[key]` on a missing key: throws, or is UB under `NDEBUG` | a required field: serde reports it by name |
//! | `j.value(key, default)`: `default` only when the key is *absent* | `#[serde(default = "…")]`, which never sees a present value |
//! | a present-but-`null` where a value is required: a type error | reached by [`present`], because serde would otherwise fold `null` into `None` |
//! | a present-but-`null` the reference tests with `!is_null()` | plain `Option<T>`, whose `null` *is* `None` |
//! | `get<int>()`: accepts JSON floats and truncates toward zero | [`count`], [`Count`] |
//! | `get<float>()`: one `static_cast` from the parsed double | [`float`] |
//! | `get<bool>()`: accepts nothing but a JSON boolean | serde's own `bool`, which is already this strict |
//!
//! Sizes and counts go through [`count`] rather than serde's `usize`, which
//! would reject `3.0` where the reference accepts it. Negative values are
//! refused: the reference stores these in `int` and would carry one into a
//! `resize`, and naming the field beats an out-of-bounds later on. So are
//! values above [`super::MAX_COUNT`], which no real capture approaches and
//! which would otherwise size an allocation the process cannot survive.
//!
//! Failures funnel into [`Error::Config`] through [`from_value`], which tags
//! them with the position in the config tree, which serde tracks no more
//! than it tracks, for a mistyped value, the field it came from. So a missing key is
//! named ("Layer array 0: missing field `channels`") and a mistyped one is
//! located but not named ("Layer array 0: invalid type: string \"x\",
//! expected a non-negative integer"). Nesting the objects that have their
//! own struct keeps that second case narrow, and the reference's
//! `type_error` messages carry neither piece of information.

use crate::error::{Error, Result};
use serde::Deserialize;
use serde::de::{self, DeserializeOwned, Deserializer, Unexpected, Visitor};
use serde_json::Value;
use std::fmt;

/// Deserialize `v`, reporting any failure as `ctx` plus serde's message.
///
/// `ctx` is the path to `v` in the config tree ("Layer array 0.head1x1").
/// Serde names a *missing* field on its own but not a mistyped one, so this
/// is what keeps an error pointing at the object it came from.
pub fn from_value<T: DeserializeOwned>(v: &Value, ctx: &str) -> Result<T> {
    T::deserialize(v).map_err(|e| Error::Config(format!("{ctx}: {e}")))
}

/// `get<int>()` narrowed to a count or size in `0..=MAX_COUNT`.
///
/// Use as `#[serde(deserialize_with = "de::count")]` on a `usize` field.
pub fn count<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<usize, D::Error> {
    d.deserialize_any(CountVisitor)
}

/// [`count`] for a `usize` that appears inside another type: `Option`, `Vec`.
#[derive(Debug, Clone, Copy)]
pub struct Count(pub usize);

impl<'de> Deserialize<'de> for Count {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        count(d).map(Count)
    }
}

struct CountVisitor;

impl CountVisitor {
    fn bounded<E: de::Error>(
        v: usize,
        unexpected: Unexpected<'_>,
    ) -> std::result::Result<usize, E> {
        if v <= super::MAX_COUNT {
            Ok(v)
        } else {
            Err(E::invalid_value(unexpected, &CountVisitor))
        }
    }
}

impl Visitor<'_> for CountVisitor {
    type Value = usize;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "a non-negative integer no larger than {}",
            super::MAX_COUNT
        )
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<usize, E> {
        let n = usize::try_from(v).map_err(|_| E::invalid_value(Unexpected::Unsigned(v), &self))?;
        Self::bounded(n, Unexpected::Unsigned(v))
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<usize, E> {
        let n = usize::try_from(v).map_err(|_| E::invalid_value(Unexpected::Signed(v), &self))?;
        Self::bounded(n, Unexpected::Signed(v))
    }

    /// nlohmann's `get<int>()` on a float is a plain `static_cast`: truncate
    /// toward zero, then the same range check.
    fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<usize, E> {
        let t = v.trunc();
        // `usize::MAX as f64` rounds up to 2^64, so `<` is the exact bound.
        // NaN and the infinities fail both comparisons.
        if t >= 0.0 && t < usize::MAX as f64 {
            Self::bounded(t as usize, Unexpected::Float(v))
        } else {
            Err(E::invalid_value(Unexpected::Float(v), &self))
        }
    }
}

/// `get<std::vector<int>>()` restricted to non-negative values.
pub fn count_vec<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Vec<usize>, D::Error> {
    Ok(Vec::<Count>::deserialize(d)?
        .into_iter()
        .map(|c| c.0)
        .collect())
}

/// `get<float>()`: the double is narrowed exactly as the reference's
/// `static_cast<float>` narrows it.
pub fn float<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<f32, D::Error> {
    f64::deserialize(d).map(|x| x as f32)
}

/// [`float`] for an `f32` that appears inside another type.
#[derive(Debug, Clone, Copy)]
pub struct F32(pub f32);

impl<'de> Deserialize<'de> for F32 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        float(d).map(F32)
    }
}

/// An optional field for which a present `null` is a type error, not a `None`.
///
/// Serde maps `null` onto `None` for any `Option`, which is the reference's
/// rule only where it tests `!is_null()`. Where it instead reads the key and
/// converts, through `j.value(key, default)` or a `find() != end()` followed
/// by a `get<T>()`, a present `null` throws, and this is how that is
/// spelled:
///
/// ```text
/// #[serde(default, deserialize_with = "de::present")]
/// bottleneck: Option<de::Count>,
/// ```
///
/// The `default` covers absence; anything present, `null` included, is handed
/// to `T`.
pub fn present<'de, D, T>(d: D) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(d).map(Some)
}

/// `#[serde(default = "de::one")]`: the reference's usual absent-key count.
pub const fn one() -> usize {
    1
}

/// `#[serde(default = "de::yes")]`: an absent flag the reference reads as on.
pub const fn yes() -> bool {
    true
}

/// The reference's `find(key) != end() && !j[key].is_null()`, as a borrow.
///
/// A subtree big enough that copying it would be felt, a nested `.nam`
/// document and its weights, is read through this rather than through an
/// `Option<Value>` field, which would clone it into the struct.
pub fn non_null<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.get(key).filter(|x| !x.is_null())
}

/// `get<double>()`.
pub fn as_f64(v: &Value, ctx: &str) -> Result<f64> {
    v.as_f64()
        .ok_or_else(|| Error::Config(format!("{ctx}: expected a number, found {}", kind(v))))
}

/// `get<std::string>()`.
pub fn as_str<'a>(v: &'a Value, ctx: &str) -> Result<&'a str> {
    v.as_str()
        .ok_or_else(|| Error::Config(format!("{ctx}: expected a string, found {}", kind(v))))
}

/// The JSON type name, for error messages.
pub fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Debug, Deserialize)]
    struct Probe {
        #[serde(deserialize_with = "count")]
        required: usize,
        #[serde(default = "seven", deserialize_with = "count")]
        a: usize,
        #[serde(default, deserialize_with = "present")]
        b: Option<Count>,
        #[serde(default)]
        c: Option<Count>,
    }

    fn seven() -> usize {
        7
    }

    /// `required` is supplied by every case that is not about it.
    fn probe(mut v: Value) -> Result<Probe> {
        let o = v.as_object_mut().unwrap();
        o.entry("required").or_insert(json!(0));
        from_value(&v, "x")
    }

    #[test]
    fn a_default_covers_absence_but_never_a_present_value() {
        assert_eq!(probe(json!({"required": 1})).unwrap().required, 1);
        assert_eq!(probe(json!({})).unwrap().a, 7);
        assert_eq!(probe(json!({"a": 5})).unwrap().a, 5);
        // Present-but-null is a type error in nlohmann, not a fallback.
        assert!(probe(json!({"a": null})).is_err());
    }

    #[test]
    fn present_is_what_separates_a_missing_key_from_a_null_one() {
        assert!(probe(json!({})).unwrap().b.is_none());
        assert!(probe(json!({"b": null})).is_err());
        // Without it, serde folds null into None, the rule the reference
        // uses only where it tests `!is_null()`.
        assert!(probe(json!({"c": null})).unwrap().c.is_none());
    }

    #[test]
    fn int_conversion_truncates_floats_like_static_cast() {
        assert_eq!(probe(json!({"a": 3.9})).unwrap().a, 3);
        assert!(probe(json!({"a": -1})).is_err());
        assert!(probe(json!({"a": -3.9})).is_err());
        assert!(probe(json!({"a": "3"})).is_err());
        assert!(probe(json!({"a": true})).is_err());
    }

    #[test]
    fn counts_are_capped() {
        let max = super::super::MAX_COUNT;
        assert_eq!(probe(json!({"a": max})).unwrap().a, max);
        let e = probe(json!({"a": max + 1})).unwrap_err().to_string();
        assert!(e.contains("no larger than"), "{e}");
        assert!(probe(json!({"a": 1e30})).is_err());
        assert!(probe(json!({"a": u64::MAX})).is_err());
    }

    #[test]
    fn bool_conversion_is_strict() {
        #[derive(Deserialize)]
        struct Flag {
            active: bool,
        }
        // The table at the top of this module leans on serde's own `bool`
        // being exactly as strict as `get<bool>()`.
        assert!(from_value::<Flag>(&json!({"active": 1}), "x").is_err());
        assert!(from_value::<Flag>(&json!({"active": "true"}), "x").is_err());
        assert!(
            from_value::<Flag>(&json!({"active": true}), "x")
                .unwrap()
                .active
        );
    }

    #[test]
    fn errors_say_where_in_the_tree_they_happened() {
        let e = probe(json!({"a": "3"})).unwrap_err().to_string();
        assert!(e.contains("x: "), "{e}");
        // Serde names a missing field on its own, so the two compose.
        let e = from_value::<Probe>(&json!({}), "x")
            .unwrap_err()
            .to_string();
        assert!(e.contains("x: ") && e.contains("required"), "{e}");
    }
}
