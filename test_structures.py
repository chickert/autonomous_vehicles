import pytest

import numpy as np

from structures import *


def test_position():

    # Check properties:
    pos0 = Position(x=0.0, L=350)
    
    # Check exception behavior:
    pos1 = Position(x=0.0, L=350)
    pos2 = Position(x=0.0, L=200)
    with pytest.raises(AssertionError):
        pos1 == pos2  # Lengths don't match.
    with pytest.raises(AssertionError):
        pos1.distance_to(pos2)  # Lenths don't match.
    with pytest.raises(AssertionError):
        Position(pos2, L=830)  # Lengths don't match.
    with pytest.raises(AssertionError):
        Position(123, L=0)  # Length must be strictly positive.
    with pytest.raises(AttributeError):
        pos1.L = 500

    # Check distance function:
    pos3 = Position(x=3, L=200)
    assert pos3.distance_to(0) == 197
    assert pos3.distance_to(0,reverse=True) == 3
    assert np.isclose( pos3.distance_to(3.14), 0.14 )
    assert np.isclose( pos3.distance_to(1e15), 200-3 )

    # Check __eq__:
    pos4 = Position(x=3.14, L=200)
    assert (3.14 - 3.00) != 0.14  # Because of floating point error.
    assert (pos4 - 3.00) == 0.14  # Because __eq__ has built-in tolerance.
    
    

