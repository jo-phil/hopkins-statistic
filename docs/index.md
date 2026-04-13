---
title: ""
icon: lucide/house
---

# hopkins-statistic

--8<-- "README.md:opener"

## Installation

=== "pip"

    ```bash
    pip install hopkins-statistic
    ```

=== "uv (project)"

    ```bash
    uv add hopkins-statistic
    ```

=== "uv (environment)"

    ```bash
    uv pip install hopkins-statistic
    ```

## Usage

=== "Hopkins statistic"

    ```python exec="1" source="material-block" result="python { .no-copy }"
    import numpy as np
    from hopkins_statistic import hopkins

    rng = np.random.default_rng(42)
    X = rng.uniform(size=(100, 2))
    
    print(hopkins(X, rng=rng))
    ```

=== "Hopkins test"

    ```python exec="1" source="material-block" result="python { .no-copy }"
    import numpy as np
    from hopkins_statistic import hopkins_test
    
    rng = np.random.default_rng(42)
    X = rng.uniform(size=(100, 2))
    
    print(hopkins_test(X, rng=rng))
    ```
