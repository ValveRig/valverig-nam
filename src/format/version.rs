//! `.nam` version parsing and the supported-range check.
//!
//! Transliterated from `NAM/get_dsp.cpp` (`ParseVersion`,
//! `CoreVersionSupportChecker::support`, `verify_config_version`).

use crate::error::{Error, Result};

/// Oldest `.nam` file version this crate accepts -
/// `EARLIEST_SUPPORTED_NAM_FILE_VERSION` in `NAM/get_dsp.h`.
pub const EARLIEST_SUPPORTED_NAM_FILE_VERSION: &str = "0.5.0";

/// Newest `.nam` file version this crate fully supports -
/// `LATEST_FULLY_SUPPORTED_NAM_FILE_VERSION` in `NAM/get_dsp.h`.
pub const LATEST_FULLY_SUPPORTED_NAM_FILE_VERSION: &str = "0.7.0";

const EARLIEST: (u32, u32, u32) = (0, 5, 0);
const LATEST: (u32, u32, u32) = (0, 7, 0);

/// How well this crate supports a given file version.
///
/// Mirrors `nam::Supported`. The ordering of the variants is the reference's
/// (`NO = 0 < PARTIAL = 1 < YES = 2`), which is how it merges the verdicts of
/// several registered checkers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Supported {
    /// Refuse to load.
    No,
    /// Load, but the file may use fields this crate does not know.
    Partial,
    /// Fully supported.
    Yes,
}

/// Parse a `major.minor.patch` version string.
///
/// Faithful to `nam::ParseVersion`, which splits on `.` with `std::getline`
/// and runs each piece through `std::stoi`. Two consequences are load-bearing
/// and reproduced here: the *third* piece is the whole remainder, so
/// `"1.2.3.4"` parses as `(1, 2, 3)`; and `stoi` stops at the first non-digit,
/// so `"0.5.0-rc1"` parses as `(0, 5, 0)`. Neither string reaches this
/// function from [`is_version_supported`], which screens with the reference's
/// `^\d+\.\d+\.\d+$` regex first.
///
/// Errors where the reference throws: an empty or non-numeric component
/// (`std::invalid_argument`), a component outside `int` (`std::out_of_range`),
/// or a negative component.
pub fn parse_version(s: &str) -> Result<(u32, u32, u32)> {
    let (major_str, rest) = match s.split_once('.') {
        Some((a, b)) => (a, b),
        None => (s, ""),
    };
    let (minor_str, patch_str) = match rest.split_once('.') {
        Some((a, b)) => (a, b),
        None => (rest, ""),
    };

    let mut out = [0i64; 3];
    for (slot, piece) in out.iter_mut().zip([major_str, minor_str, patch_str]) {
        *slot = match stoi(piece) {
            Stoi::Ok(v) => v,
            Stoi::Invalid => {
                return Err(Error::Config(format!("Invalid version string: {s}")));
            }
            Stoi::OutOfRange => {
                return Err(Error::Config(format!("Version string out of range: {s}")));
            }
        };
    }

    if out.iter().any(|v| *v < 0) {
        return Err(Error::Config(format!("Negative version component: {s}")));
    }
    Ok((out[0] as u32, out[1] as u32, out[2] as u32))
}

/// The outcome of `std::stoi`, whose three cases the reference distinguishes.
enum Stoi {
    Ok(i64),
    Invalid,
    OutOfRange,
}

/// `std::stoi` for base 10: leading whitespace, an optional sign, then digits
/// up to the first character that is not one.
fn stoi(s: &str) -> Stoi {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() {
        i += 1;
    }
    let negative = match bytes.get(i) {
        Some(b'-') => {
            i += 1;
            true
        }
        Some(b'+') => {
            i += 1;
            false
        }
        _ => false,
    };
    let start = i;
    let mut acc: i64 = 0;
    let mut overflow = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        if !overflow {
            acc = acc * 10 + i64::from(bytes[i] - b'0');
            // `int` is 32-bit everywhere the reference is built.
            if acc > i64::from(i32::MAX) + 1 {
                overflow = true;
            }
        }
        i += 1;
    }
    if i == start {
        return Stoi::Invalid;
    }
    let value = if negative { -acc } else { acc };
    if overflow || value > i64::from(i32::MAX) || value < i64::from(i32::MIN) {
        return Stoi::OutOfRange;
    }
    Stoi::Ok(value)
}

/// `true` when `s` matches the reference's `^\d+\.\d+\.\d+$`.
fn is_semver(s: &str) -> bool {
    let mut parts = s.split('.');
    let ok = |p: Option<&str>| matches!(p, Some(p) if !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    ok(parts.next()) && ok(parts.next()) && ok(parts.next()) && parts.next().is_none()
}

/// How well this crate supports `version`.
///
/// `CoreVersionSupportChecker::support`, transcribed. The middle test is the
/// reference's and is deliberately *not* a lexicographic comparison:
///
/// ```text
/// if (parsed.major > latest.major || parsed.minor > latest.minor) return NO;
/// ```
///
/// so a bumped minor is rejected outright while a bumped patch degrades to
/// [`Supported::Partial`]. Keeping the disjunction as written is what makes
/// `0.8.0` unsupported but `0.7.1` partially supported.
///
/// Errors only where the reference's `std::stoi` throws `out_of_range`: a
/// well-shaped version whose components exceed `int`, e.g. `"99999999999.0.0"`.
pub fn is_version_supported(version: &str) -> Result<Supported> {
    if !is_semver(version) {
        return Ok(Supported::No);
    }
    let parsed = parse_version(version)?;
    if parsed < EARLIEST {
        return Ok(Supported::No);
    }
    if parsed.0 > LATEST.0 || parsed.1 > LATEST.1 {
        return Ok(Supported::No);
    }
    if LATEST < parsed {
        return Ok(Supported::Partial);
    }
    Ok(Supported::Yes)
}

/// `verify_config_version`: reject [`Supported::No`], pass everything else
/// through.
///
/// The reference logs a line to `stderr` for a partially-supported version and
/// continues; this returns the verdict instead of printing, so a host can
/// decide whether to surface it.
pub fn verify_version(version: &str) -> Result<Supported> {
    let support = is_version_supported(version)?;
    if support == Supported::No {
        return Err(Error::UnsupportedVersion {
            found: version.to_string(),
            earliest: EARLIEST_SUPPORTED_NAM_FILE_VERSION,
            latest: LATEST_FULLY_SUPPORTED_NAM_FILE_VERSION,
        });
    }
    Ok(support)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_version_splits_like_getline() {
        assert_eq!(parse_version("0.5.0").unwrap(), (0, 5, 0));
        assert_eq!(parse_version("1.20.300").unwrap(), (1, 20, 300));
        // The third getline has no delimiter, so it swallows the remainder and
        // stoi stops at the '.'.
        assert_eq!(parse_version("1.2.3.4").unwrap(), (1, 2, 3));
        assert_eq!(parse_version("0.5.0-rc1").unwrap(), (0, 5, 0));
    }

    #[test]
    fn parse_version_rejects_what_stoi_rejects() {
        assert!(parse_version("0.5").is_err()); // patch component empty
        assert!(parse_version("").is_err());
        assert!(parse_version("v0.5.0").is_err());
        assert!(parse_version("0.-1.0").is_err()); // negative component
        assert!(parse_version("99999999999.0.0").is_err()); // out of int range
    }

    #[test]
    fn support_matches_the_reference_table() {
        // Hand-evaluated against CoreVersionSupportChecker::support with
        // earliest = 0.5.0 and latest = 0.7.0.
        let cases: [(&str, Supported); 14] = [
            ("0.4.9", Supported::No),
            ("0.5.0", Supported::Yes),
            ("0.5.4", Supported::Yes),
            ("0.6.0", Supported::Yes),
            ("0.7.0", Supported::Yes),
            // Patch beyond latest: partial.
            ("0.7.1", Supported::Partial),
            ("0.7.99", Supported::Partial),
            // Minor beyond latest: rejected by the `minor >` disjunct.
            ("0.8.0", Supported::No),
            // Major beyond latest: rejected. Note the reference's disjunction
            // means 1.0.0 fails on `major >`, not on a version comparison.
            ("1.0.0", Supported::No),
            // Not semver-shaped.
            ("0.5", Supported::No),
            ("0.5.0.1", Supported::No),
            ("v0.5.0", Supported::No),
            ("", Supported::No),
            ("0.5.0-rc1", Supported::No),
        ];
        for (v, expected) in cases {
            assert_eq!(is_version_supported(v).unwrap(), expected, "version {v}");
        }
    }

    #[test]
    fn verify_rejects_only_unsupported() {
        assert_eq!(verify_version("0.5.4").unwrap(), Supported::Yes);
        assert_eq!(verify_version("0.7.5").unwrap(), Supported::Partial);
        assert!(matches!(
            verify_version("0.4.0"),
            Err(Error::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn one_point_zero_is_rejected_even_though_a_lexicographic_test_would_pass_minor() {
        // 1.0.0 has minor 0, which is <= latest.minor. Only the `major >`
        // disjunct rejects it, so a regression that dropped that term would
        // silently accept a future major version.
        assert_eq!(is_version_supported("1.0.0").unwrap(), Supported::No);
    }
}
