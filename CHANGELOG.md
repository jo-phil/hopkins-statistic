# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-01-02

### Added

- `hopkins_test()` to perform a Hopkins test by computing the statistic
  and the $p$-value under the $\mathrm{Beta}(m, m)$ null distribution

### Fixed

- `rng` parameter typing to use RNGLike and SeedLike aliases following
  [Scientific Python - SPEC 7](https://scientific-python.org/specs/spec-0007/),
  broadening the range of accepted seed and generator types

## [0.1.0] - 2025-12-28

### Added

- `hopkins()` to compute the Hopkins statistic for $d$-dimensional data
  following Cross & Jain (1982), using a (hyper-)rectangular sampling frame

[Unreleased]: https://github.com/jo-phil/hopkins-statistic/compare/0.2.0...HEAD
[0.2.0]: https://github.com/jo-phil/hopkins-statistic/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/jo-phil/hopkins-statistic/releases/tag/0.1.0
