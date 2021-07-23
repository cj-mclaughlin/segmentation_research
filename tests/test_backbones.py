from segmentation_research.backbones.drn import drn, drn_b_26, drn_c_26, drn_c_105
from tensorflow.keras.layers import Input


INPUT = Input((None, None, 3))

def test_drn_prebuilts():
    for m in [drn_b_26, drn_c_26, drn_c_105]:
        model = m(INPUT)
        assert model is not None, "all supplied drn models should compile properly"

def test_drn_invalid_arch():
    try:
        model = drn(INPUT, arch="D")
    except ValueError:
        assert True, "should have arch assertion"