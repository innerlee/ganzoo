import os
import pytest
import torch
import torchvision
import ganzoo as gz


class TestTorch(object):

    @staticmethod
    def test_torch():
        assert torch.__version__ == '0.3.0.post4'

    @staticmethod
    def test_torchvision():
        assert torchvision.__version__ == '0.2.0'
