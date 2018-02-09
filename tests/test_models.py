import os
import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal)
import ganzoo as gz


class TestHello(object):

    @staticmethod
    def test_hello():
        assert 1 == 1
