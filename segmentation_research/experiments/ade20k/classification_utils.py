# utility functions for extracting classification heads
from config import N_CLASSES
import numpy as np

def mask_to_subgrids(mask, cell_scale):
    """
    break WxH annotation array into a cell_scale x cell_scale vectors
    """
    num_elem_row, num_elem_col = int(mask.shape[0] / cell_scale), int(mask.shape[1] / cell_scale)
    res = []
    for h in range(cell_scale):
        for w in range(cell_scale):
            start_h = h * num_elem_row
            start_w = w * num_elem_col
            end_h = min((h+1)*num_elem_row, mask.shape[0])
            end_w = min((w+1)*num_elem_col, mask.shape[1])
            section = mask[start_h:end_h, start_w:end_w]
            res.append(section)
    return res

def unique_to_sparse(unique):
    """
    list of unique classes --> onehot sparse matrix
    note that we ignore background class, so each index is offset by -1
    """
    sparse = np.zeros((N_CLASSES-1))
    for num in unique:
        if num != 0:
            sparse[num-1] = 1
    return sparse

def vector_list_to_mat(vectors):
    """
    take list of vectors and stack them to a square matrix
    """
    n_rows = int(np.sqrt(len(vectors)))
    rows = []
    count = 0
    curr_row = []
    for i in range(len(vectors)):
        if count < n_rows:
            curr_row.append(vectors[i])
        if count == n_rows:
            count = 0
            rows.append(curr_row)
            curr_row = [vectors[i]]
        count += 1
    rows.append(curr_row)
    return np.asarray(rows)

def extract_mask_classes(mask, head_sizes=[1, 2, 3, 6]):
    """
    annotation mask --> set of head_sizes x head_sizes matrices with one-hot class labels
    encoding which classes are present in that region
    """
    assert len(head_sizes) == 4
    classification_head_labels = []
    for s in head_sizes:
        if s == 1: # special case
            uniq = np.unique(mask)
            mat = unique_to_sparse(uniq)
        else:
            quadrants = mask_to_subgrids(mask[:,:,0], s)
            uniq_vectors = [ unique_to_sparse(np.unique(m)) for m in quadrants ]
            mat = vector_list_to_mat(uniq_vectors)
        classification_head_labels.append(mat.astype(np.float32))

    return classification_head_labels
