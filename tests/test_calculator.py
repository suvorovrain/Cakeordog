"""Tests for calculator"""

from cakeordog.calculator import get_five


def test_get_five_returns_five() -> None:
    """Check if function returns 5"""
    result = get_five()
    assert result == 5, f"Expected 5, received {result}"
