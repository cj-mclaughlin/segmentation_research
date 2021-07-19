from segmentation_research.models.pspnet import pspnet
from segmentation_research.models.fpn import fpn


def test_pspnet_valid_shape():
    model = pspnet(input_shape=(720, 960, 3), num_classes=11)
    print(model.summary())
    assert model is not None, "pspnet should compile properly"

def test_pspnet_invalid_shape():
    try:
        model = pspnet(input_shape=(39, 39, 3), num_classes=11)
    except AssertionError:
        assert True, "should have failed input shape assertion"

def test_fpn_valid_shape():
    model = fpn(input_shape=(720, 960, 3), num_classes=11)
    print(model.summary())
    assert model is not None, "fpn should compile properly"

test_fpn_valid_shape()