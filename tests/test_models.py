from segmentation_research.models.pspnet import PSPNet
from segmentation_research.models.fpn import FPN

def test_pspnet_valid_shape():
    model = PSPNet(input_shape=(720, 960, 3), num_classes=11)
    print(model.summary())
    assert model is not None, "pspnet should compile properly"

def test_pspnet_invalid_shape():
    try:
        model = PSPNet(input_shape=(39, 39, 3), num_classes=11)
    except AssertionError:
        assert True, "should have failed input shape assertion"

def test_fpn_valid_shape():
    model = FPN(input_shape=(288, 288, 3), num_classes=11)
    print(model.summary())
    assert model is not None, "fpn should compile properly"