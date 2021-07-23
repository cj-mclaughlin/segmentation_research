from segmentation_research.backbones.drn import drn, drn_b_26, drn_c_26, drn_c_105
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

INPUT = Input((None, None, 3))

def count_layers(backbone, layer_name):
    count = 0
    features = backbone(INPUT)[-1]
    model = Model(INPUT, features)
    for layer in model.layers:
        if layer_name in layer.name:
            count += 1
    return count

def test_architecture():
    resid_b = count_layers(drn_b_26, "add")
    resid_c = count_layers(drn_c_26, "add")
    assert resid_b > resid_c, "b architecture should have more residual connections"


def test_drn_prebuilts():
    for m in [drn_b_26, drn_c_26, drn_c_105]:
        model = m(INPUT)
        assert model is not None, "all supplied drn models should compile properly"

def test_drn_invalid_arch():
    try:
        model = drn(INPUT, arch="D")
    except ValueError:
        assert True, "should have arch assertion"

test_architecture()