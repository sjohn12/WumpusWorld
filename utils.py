# Python version: 3.11
#Built off sample assignment 2 solution posted
from typing import List, Tuple, Dict, Any, Optional, Union, Union, TypedDict, Set


def manhattan_distance(start: Tuple[int, int], end: Tuple[int, int]) -> int:
    """Calculate the Manhattan distance between two points.

    Args:
        start (Tuple[int, int]): The start location.
        end (Tuple[int, int]): The end location.

    Returns:
        int: The Manhattan distance.
    """
    return abs(start[0] - end[0]) + abs(start[1] - end[1])
