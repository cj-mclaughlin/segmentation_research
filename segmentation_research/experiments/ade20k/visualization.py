import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from preprocessing import denormalize
import pandas as pd
import numpy as np

import tensorflow as tf

from dataset import get_dataset
from classification_utils import mask_to_subgrids
from config import CLASS_DICT_PATH, IMG_SIZE

def create_ade20k_label_colormap():
  """Creates a label colormap used in ADE20K segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])

def get_class_color_palette_df(csv_path):
    """
    read class CSV and append color palette 
    """
    df = pd.read_csv(csv_path)
    df["Idx"] = df["Idx"]
    df_min = df[["Idx", "Name"]]
    df_min = df_min.append({'Idx': 0, 'Name': "background"}, ignore_index=True)
    df_min = df_min.sort_values(by="Idx")
    cmap = create_ade20k_label_colormap()
    df_min["r"] = cmap[:,0]
    df_min["g"] = cmap[:,1]
    df_min["b"] = cmap[:,2]
    df_min = df_min.reset_index(drop=True)
    return df_min.to_dict()

def get_class_cbar(mask):
    """
    get colorbar for plotting purposes of a given annotation mask
    """
    CLASS_DICT = get_class_color_palette_df(CLASS_DICT_PATH)
    label = []
    for ldx in np.unique(mask):
        label.append(CLASS_DICT["Name"][ldx])

    left = np.array(range(len(np.unique(mask))))
    height = np.ones(len(np.unique(mask))) / len(np.unique(mask))

    col = []
    for i in np.unique(mask):
        col.append([CLASS_DICT["r"][i]/255, CLASS_DICT["g"][i]/255, CLASS_DICT["b"][i]/255])

    return left, height, label, col


def plot_semantic_legend(mask, ax):
    left, height, labels, colors = get_class_cbar(mask)
    handles, _ = ax.get_legend_handles_labels()
    for i in range(len(colors)):
        label = labels[i].split(";")[0] if ";" in labels[i] else labels[i]
        patch = Patch(color=colors[i], label=label)
        handles.append(patch)
    ax.legend(handles=handles)

def display_sample(X, y, mask_title="GT Mask", show_pool = False):
    """
    image, gt w/cmap
    pool2 with cmap
    pool3 with cmap
    pool4 with cmap
    """
    X = X[0]  # get one image sample

    # image & gt cmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 18))
    ax1.imshow(tf.keras.preprocessing.image.array_to_img(denormalize(X)))
    ax1.set_title("GT Image")
    ax1.axis("off")
    mask = np.squeeze(y["end_prediction"][0])
    ax2.imshow(create_ade20k_label_colormap()[mask])
    ax2.set_title(mask_title)
    ax2.axis("off")
    plot_semantic_legend(mask, ax2)
    plt.show()

    # pool2
    # pool2_quadrants = mask_to_subgrids(mask, int(IMG_SIZE/2))
    # fig, axes = plt.subplots(1, len(pool2_quadrants), figsize=(18, 18))
    # for i in range(len(pool2_quadrants)):
    #     plot_semantic_legend(pool2_quadrants[i], ax=axes[i])
    #     axes[i].imshow(create_ade20k_label_colormap()[pool2_quadrants[i]])
    #     axes[i].set_title(f"Quadrant {i+1} Mask")
    #     axes[i].axis("off")
    # plt.show()

def get_predicted_classes(quadrant):
    classes = {}
    CLASS_DICT = get_class_color_palette_df(CLASS_DICT_PATH)
    for i in range(len(quadrant)):
        if quadrant[i] >= 0.5:
            classes[CLASS_DICT["Name"][i+1]] = quadrant[i]
    return classes

def display_predictions(X, y, y_pred):
    display_sample(X, y)
    X = X[0]
    mask = np.squeeze(np.argmax(y_pred[-1][0], -1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 18))
    ax1.imshow(tf.keras.preprocessing.image.array_to_img(denormalize(X)))
    ax1.set_title("GT Image")
    ax1.axis("off")
    ax2.imshow(create_ade20k_label_colormap()[mask])
    ax2.set_title("Predicted Mask")
    ax2.axis("off")
    left, height, labels, colors = get_class_cbar(mask)
    handles, _ = ax2.get_legend_handles_labels()
    for i in range(len(colors)):
        label = labels[i].split(";")[0] if ";" in labels[i] else labels[i]
        patch = Patch(color=colors[i], label=label)
        handles.append(patch)
    ax2.legend(handles=handles)
    plt.show()

    # print belief for each quadrant
    pool2_predictions = y_pred[1][0]
    q1_classes = get_predicted_classes(pool2_predictions[0][0])
    print("Predictions for 1st quadrant:")
    print(q1_classes)
    q2_classes = get_predicted_classes(pool2_predictions[0][1])
    print("Predictions for 2nd quadrant:")
    print(q2_classes)
    q3_classes = get_predicted_classes(pool2_predictions[1][0])
    print("Predictions for 3rd quadrant:")
    print(q3_classes)
    q4_classes = get_predicted_classes(pool2_predictions[1][1])
    print("Predictions for 4th quadrant:")
    print(q4_classes)
    

if __name__ == "__main__":
    # display samples as a sanity check
    dataset = get_dataset(classification_heads=None, crop_aug=True)
    train = dataset["test"]
    for X, y in train.take(1):
        display_sample(X, y)
        break