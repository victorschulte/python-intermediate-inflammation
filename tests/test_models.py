"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ]
)
def test_daily_mean(test, expected):
    """Test that mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2, -2], [3, 4, 5], [5, -4, 7]], [5, 4, 7]),
    ]
)
def test_daily_max(test, expected):
    """Test that max function works for an array of positive and negative integers."""
    from inflammation.models import daily_max

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))
    

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[-3, 2, 7], [3, 1, 4], [5, 6, -7]], [-3, 1, -7]),
        ([[1, 2, -2], [3, 4, 5], [5, -4, 7]], [1, -4, -2]),
    ]
)
def test_daily_min(test, expected):
    """Test that min function works for an array of positive and negative integers."""
    from inflammation.models import daily_min

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))
    
    
def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min
    
    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])
