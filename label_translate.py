import os
from cityscapes_tools.cityscapesscripts.helpers import labels
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def show_label_info():
    num_classes = 19
    for trainId in range(num_classes):
        print(trainId, labels.trainId2label[trainId].id, labels.trainId2label[trainId].name)

def read_label_img(label_img_path):
    label_img = cv2.imread(label_img_path)
    label_img = label_img[:, : , 0]
    return label_img

def set_palette():
    n = 256
    palette = []
    for i in range(0, 256):
        if i in labels.id2label:
            palette.append(list(labels.id2label[i].color))
        else:
            palette.append([0,0,0])
    palette = np.array(palette)
    return palette.flatten().tolist()

def label_to_color_img(label_img):
    palette = set_palette()
    color_img = Image.fromarray(label_img)
    color_img.putpalette(palette)
    return color_img

def convert_label_to_color_image(label_path, color_path):
    #gt_label_path = "cs230/data/cityscapes/gtFine/val/
    label_list = os.listdir(label_path)
    for label in label_list:
        img_name = os.path.basename(label)
        label_img_path = os.path.join(label_path, img_name)
        label_img = read_label_img(label_img_path)
        color_img = label_to_color_img(label_img)

        color_img_path = os.path.join(color_path, img_name)
        color_img.save(color_img_path)

if __name__ == "__main__":
    show_label_info()
    label_path = "test_results/cityscapes/labels/"
    color_path = "test_results/cityscapes/color_images/"
    convert_label_to_color_image(label_path, color_path)
