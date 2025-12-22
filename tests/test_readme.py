import pytest
from pytest_examples import find_examples

from hopkins_statistic import hopkins


@pytest.mark.parametrize("example", find_examples("README.md"), ids=str)
def test_examples(example, eval_example):
    eval_example.run_print_check(example)
