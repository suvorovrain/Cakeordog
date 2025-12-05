"""Sample module"""


def get_five() -> int:
    """
    Returns 5.

    Returns:
        int: Always 5

    Examples:
        >>> get_five()
        5
    """
    return 5


def main() -> None:
    """Entry point"""
    result = get_five()
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
