# hopkins-statistic

A Python package for computing the Hopkins statistic to test for departure from
complete spatial randomness (CSR), often used to assess clustering tendency.

## Installation

_Not yet available._

## Quick Start

```python
import numpy as np
from hopkins_statistic import hopkins

rng = np.random.default_rng(42)
X = rng.uniform(size=(100, 2))
print(hopkins(X, rng=rng))  # ~0.5 for CSR-like data
```

## Definition

As noted by Wright (2022), the definition of the Hopkins statistic is a common
source of confusion in both literature and software implementations. To avoid
ambiguity, this package defaults to the formulation by Cross and Jain (1982),
which generalizes the original definition by Hopkins and Skellam (1954) to data
in any dimension.

Given a set $X$ of $n$ data points in a $d$-dimensional Euclidean space,
choose $m$ such that $m \ll n$ and let

- $`\{x_i\}_{i=1}^m`$ be a simple random sample from $X$ (without replacement),
  and
- $`\{y_i\}_{i=1}^m`$ be synthetic points drawn i.i.d. uniformly from the
  sampling frame.

For each $`i \in \{1,\dots,m\}`$, let

- $u_i$ be the distance from $y_i$ to its nearest neighbor in $X$, and
- $w_i$ be the distance from $x_i$ to its nearest neighbor in
  $`X \setminus \{x_i\}`$.

Then the Hopkins statistic is defined as

$$
    H = \frac{\sum_{i=1}^m u_i^d}{\sum_{i=1}^m u_i^d + \sum_{i=1}^m w_i^d}.
$$

Under the CSR null hypothesis, $H \sim \mathrm{Beta}(m,m)$.

## Interpretation

While critical values can be obtained from the $\mathrm{Beta}(m,m)$ null
distribution, the table below lists commonly used rule-of-thumb thresholds
for interpreting $H$.

|           $H$ | Pattern   | Interpretation                                          |
|--------------:|:----------|:--------------------------------------------------------|
|     $\ge 0.7$ | clustered | Suggests a departure from CSR toward clustering.        |
| $\approx 0.5$ | random    | Consistent with complete spatial randomness (CSR).      |
|     $\le 0.3$ | regular   | Suggests a departure from CSR toward more even spacing. |

## Practical notes

- The **sampling frame** is approximated as the axis-aligned bounding box of `X`
  in this implementation. If this is not a reasonable representation of the
  region the data come from, `X` may be transformed beforehand.

- **Euclidean distances** on non-spatial data often benefit from scaling the
  features in `X` to comparable ranges.

- The **sample size** `m` should typically be at least 10 to avoid small-sample
  problems and no more than about one tenth of $n$ to balance variance and the
  approximations used for the null distribution of the statistic.

- The **exponent** `power` applied to distances defaults to $d$, the number of
  columns in `X`. This yields the statistic as defined above. Other values alter
  the null distribution.


## References

- Cross, G. R., & Jain, A. K. (1982). Measurement of Clustering Tendency.
  In Theory and Application of Digital Control (pp. 315–320). Elsevier.
  https://doi.org/10.1016/b978-0-08-027618-2.50054-1
- Hopkins, B., & Skellam, J. G. (1954). A New Method for determining the Type
  of Distribution of Plant Individuals. Annals of Botany, 18(2), 213–227.
  https://doi.org/10.1093/oxfordjournals.aob.a083391
- Wright, K. (2022). Will the Real Hopkins Statistic Please Stand Up?
  The R Journal, 14(3), 282–292.  
  https://doi.org/10.32614/rj-2022-055

## License

MIT. See [LICENSE](LICENSE).
