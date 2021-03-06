from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.layers import BatchNormalization

def get_num_groups(filters):
    """
    Helper to select number of groups for GroupNorm.
    Select 16 by default (as in paper)
    Otherwise takes filters//4 so we at least have 4 groups
    """
    return min(filters // 4, 16)

def get_norm_layer(normalization, filters, name_base=""):
    if normalization == 'batchnorm':
        return BatchNormalization(name=f"{name_base}-batchnorm")
    elif normalization == 'groupnorm':
        return GroupNormalization(groups=get_num_groups(filters), name=f"{name_base}-groupnorm")
    else:
        raise ValueError('normalization must be either batchnorm or groupnorm.')