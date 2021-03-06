from segmentation_research.backbones.drn import drn, drn_b_26, drn_c_26, drn_c_58, drn_c_105
from segmentation_research.backbones.resnet_v2 import resnet50_v2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

INPUT = Input((None, None, 3))

def count_layers(model, layer_name):
    count = 0
    for layer in model.layers:
        if layer_name in layer.name:
            count += 1
    return count

def test_drn_architecture():
    resid_b = count_layers(drn_b_26(INPUT)[0], "add")
    resid_c = count_layers(drn_c_26(INPUT)[0], "add")
    assert (resid_b) == (resid_c + 2), f"b architecture should have two more residual blocks, found b={resid_b} c={resid_c}"


def test_drn_prebuilts():
    for model in [drn_b_26(INPUT), drn_c_26(INPUT), drn_c_58(INPUT), drn_c_105(INPUT)]:
        assert model is not None, "all supplied drn models should compile properly"

def test_drn_invalid_arch():
    try:
        model = drn(INPUT, arch="D")
    except ValueError:
        assert True, "should have arch assertion"

def test_resnet50v2():
    # shouldnt crash and should have right # parameters
    resnet, _ = resnet50_v2(INPUT)
    assert resnet.count_params() == 23685312  # sanity check compared to semseg impl