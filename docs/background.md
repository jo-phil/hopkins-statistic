## Background

The Hopkins statistic is a test statistic for the null hypothesis of
complete spatial randomness (CSR), i.e., that points are independently
and uniformly distributed within a region of space (the sampling frame).
Under this null, the expected value of the statistic is 0.5.
Larger values indicate more clustering than expected under CSR, while
smaller values indicate more regular spacing.

### Definition

As noted by [Wright (2022)](#4), the definition of the Hopkins statistic is
a common source of confusion in both literature and software implementations.
This library defaults to the formulation by [Cross and Jain (1982)](#1), which
generalizes the original definition by [Hopkins and Skellam (1954)](#2) to
data in any dimension:

> Given a set $X$ of $n$ data points in a $d$-dimensional Euclidean space,
> choose $m$ such that $m \ll n$ and let
> - $\lbrace x_i \rbrace_{i=1}^m$ be a simple random sample from $X$
>   (without replacement), and
> - $\lbrace y_i \rbrace_{i=1}^m$ be synthetic points drawn i.i.d.
>   uniformly from the sampling frame.
> 
> For each $i \in \lbrace 1,\dots,m \rbrace$, let
> - $u_i$ be the distance from $y_i$ to its nearest neighbor in $X$, and
> - $w_i$ be the distance from $x_i$ to its nearest neighbor in
>   $X \setminus \lbrace x_i \rbrace$.
> 
> Then the Hopkins statistic is defined as
> $$
>     H = \frac{\sum_{i=1}^m u_i^d}{\sum_{i=1}^m u_i^d + \sum_{i=1}^m w_i^d}.
> $$
>  
> Under the CSR null hypothesis, $H \sim \mathrm{Beta}(m,m)$.

> [!NOTE]
> Other implementations may not raise distances to the power of $d$
> following [Lawson and Jurs (1990)](#3), or return $1 - H$ instead of $H$.

### Interpretation

While critical values can be obtained from the $\mathrm{Beta}(m,m)$
null distribution, the table below lists commonly used rules of thumb
for interpreting $H$.

|           $H$ | Pattern   | Interpretation                                          |
|--------------:|:----------|:--------------------------------------------------------|
|     $\ge 0.7$ | clustered | Suggests a departure from CSR toward clustering.        |
| $\approx 0.5$ | random    | Consistent with complete spatial randomness (CSR).      |
|     $\le 0.3$ | regular   | Suggests a departure from CSR toward more even spacing. |

## Guidelines

- This implementation approximates the **sampling frame** as the
  axis-aligned bounding box of `X`. Results are therefore relative to
  that frame; if it is not a reasonable representation of the region the
  data come from, `X` may be transformed beforehand.

- **Euclidean distances** on non-spatial data often benefit from scaling
  the features in `X` to comparable ranges.

- The **sample size** `m` should typically be at least 10 to avoid
  small-sample problems and no more than about one tenth of $n$ to
  keep the null-distribution approximations accurate.

- The **exponent** `power` applied to distances defaults to $d$, the
  number of columns in `X`. This yields the statistic as defined above.
  Other values alter the null distribution.

## References

- <a id="1"></a>
  Cross, G. R., & Jain, A. K. (1982). Measurement of clustering tendency.
  In *Theory and Application of Digital Control* (pp. 315–320). Pergamon.
  https://doi.org/10.1016/S1474-6670(17)63365-2

- <a id="2"></a>
  Hopkins, B., & Skellam, J. G. (1954). A new method for determining the type
  of distribution of plant individuals. *Annals of Botany, 18*(2), 213–227.
  https://doi.org/10.1093/oxfordjournals.aob.a083391

- <a id="3"></a>
  Lawson, R. G., & Jurs, P. C. (1990). New index for clustering tendency
  and its application to chemical problems. *Journal of chemical information
  and computer sciences, 30*(1), 36–41. https://doi.org/10.1021/ci00065a010

- <a id="4"></a>
  Wright, K. (2022). Will the Real Hopkins Statistic Please Stand Up?
  *The R Journal, 14*(3), 282–292. https://doi.org/10.32614/rj-2022-055
