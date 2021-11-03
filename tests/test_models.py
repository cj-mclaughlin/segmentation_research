from segmentation_research.models.pspnet import PSPNet
from segmentation_research.models.fpn import FPN

def test_pspnet_valid_shape():
    model = PSPNet(input_shape=(720, 960, 3), num_classes=11)
    assert model is not None, "pspnet should compile properly"

def test_pspnet_invalid_shape():
    try:
        model = PSPNet(input_shape=(39, 39, 3), num_classes=11)
    except AssertionError:
        assert True, "should have failed input shape assertion"

def test_pspnet_loss_layers():
    model = PSPNet(input_shape=(473, 473, 3), num_classes=151)
    model.compile(optimizer="adam", 
                loss=["ce", "ce"], 
                loss_weights= {
                    "stage4_prediction": 0.4, 
                    "end_prediction": 1.0
                },
                metrics=["accuracy"])
    assert model is not None, "should compile loss based on stage4 and end prediction layers"


def test_pspnet_context_bias():
    model = PSPNet(input_shape=(720, 960, 3), context_bias=True, num_classes=11)
    assert model is not None, "pspnet should compile properly"

def test_fpn_valid_shape():
    model = FPN(input_shape=(288, 288, 3), num_classes=11)
    assert model is not None, "fpn should compile properly"

test_pspnet_context_bias()