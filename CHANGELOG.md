# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `'hull'` option to the `frame` parameter for sampling from the convex hull.

### Removed

- `typing_extensions` dependency for Python 3.11.

[//]: # (--8<-- [start:released])

## [0.5.0] - 2026-03-27

### Added

- `frame` parameter to `hopkins()` and `hopkins_test()` for specifying
  a custom rectangular sampling frame.

## [0.4.0] - 2026-02-16

### Fixed

- `ModuleNotFoundError` on Python 3.12 when `typing_extensions` was not
  installed.

### Removed

- `HopkinsUndefinedWarning` and the explicit check for degenerate data;
  degenerate cases now rely on NumPy to warn and still produce `NaN`.

## [0.3.1] - 2026-01-27

### Fixed

- `hopkins()` and `hopkins_test()` to not modify input data in-place
  when called with `toroidal=True`.

## [0.3.0] - 2026-01-19

### Added

- `toroidal` option to `hopkins()` and `hopkins_test()` to compute the
  statistic with toroidal topology (i.e., periodic boundary conditions).

## [0.2.0] - 2026-01-02

### Added

- `hopkins_test()` to perform a Hopkins test by computing the statistic
  and the $p$-value under the $\mathrm{Beta}(m, m)$ null distribution.

### Fixed

- `rng` parameter typing to use RNGLike and SeedLike aliases following
  [SPEC 7], broadening the range of accepted seed and generator types.

[SPEC 7]: https://scientific-python.org/specs/spec-0007/

## [0.1.0] - 2025-12-28

### Added

- `hopkins()` to compute the Hopkins statistic for $d$-dimensional data
  following Cross & Jain (1982), using a (hyper-)rectangular sampling frame.

[Unreleased]: https://github.com/jo-phil/hopkins-statistic/compare/0.5.0...HEAD
[0.5.0]: https://github.com/jo-phil/hopkins-statistic/releases/tag/0.5.0
[0.4.0]: https://github.com/jo-phil/hopkins-statistic/releases/tag/0.4.0
[0.3.1]: https://github.com/jo-phil/hopkins-statistic/releases/tag/0.3.1
[0.3.0]: https://github.com/jo-phil/hopkins-statistic/releases/tag/0.3.0
[0.2.0]: https://github.com/jo-phil/hopkins-statistic/releases/tag/0.2.0
[0.1.0]: https://github.com/jo-phil/hopkins-statistic/releases/tag/0.1.0
