# hopkins-statistic

A Python package for computing the Hopkins statistic to test for departure from
complete spatial randomness (CSR), commonly used to assess clustering tendency.

## Installation

_Not yet available._

## Quick Start

_Not yet available._

## Definition of the Hopkins Statistic

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
  sampling frame  
  (e.g., the axis-aligned bounding box of $X$).

For each $`i \in \{1,\dots,m\}`$, let

- $u_i$ be the distance from $y_i$ to its nearest neighbor in $X$, and
- $w_i$ be the distance from $x_i$ to its nearest neighbor in
  $`X \setminus \{x_i\}`$.

Then the Hopkins statistic is defined as

$$H = \frac{\sum_{i=1}^m u_i^d}{\sum_{i=1}^m u_i^d + \sum_{i=1}^m w_i^d}.$$

Under the CSR null hypothesis, $H \sim \mathrm{Beta}(m,m)$.

### Interpretation

The table below lists commonly used rule-of-thumb thresholds for interpreting $H$.

|           $H$ | Pattern   | Interpretation                                          |
|--------------:|:----------|:--------------------------------------------------------|
|     $\ge 0.7$ | clustered | Suggests a departure from CSR toward clustering.        |
| $\approx 0.5$ | random    | Consistent with complete spatial randomness (CSR).      |
|     $\le 0.3$ | regular   | Suggests a departure from CSR toward more even spacing. |

To assess statistical significance, the observed $H$ can be compared to its CSR
null distribution, $\mathrm{Beta}(m,m)$.

## References

- Cross, G. R., & Jain, A. K. (1982). Measurement of Clustering Tendency.
  In Theory and Application of Digital Control (pp. 315–320). Elsevier.
  https://doi.org/10.1016/b978-0-08-027618-2.50054-1
- Hopkins, B., & Skellam, J. G. (1954). A New Method for determining the Type
  of Distribution of Plant Individuals. Annals of Botany, 18(2), 213–227.
  https://doi.org/10.1093/oxfordjournals.aob.a083391
- Wright, K. (2022). Will the Real Hopkins Statistic Please Stand Up?
  The R Journal, 14(3), 282–292. https://doi.org/10.32614/rj-2022-055

## License

MIT. See [LICENSE](LICENSE).
