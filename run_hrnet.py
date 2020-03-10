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

def get_files(base_dir, ext):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(base_dir):
        for file_path in f:
            if file_path.endswith(ext):
                files.append(os.path.join(r, file_path))
    return files

def cityscapes_generate_color_images():
    label_path = "test_results/cityscapes/labels/"
    color_path = "test_results/cityscapes/color_images/"
    convert_label_to_color_image(label_path, color_path)

def rescope_image_path(img_path):
   #remove "HRNet-Semantic-Segmentation/data/cityscapes/ from the img_path
   word_list = img_path.split("/")
   new_img_path = "/".join(word_list[3:])
   return new_img_path

def bdd100k_generate_image_list(mode):
    # mode could be train / val / test
    # We link the bdd100 images to cityscapes/video_images
    base_dir = "HRNet-Semantic-Segmentation/data/cityscapes/video_images/"
    base_dir = os.path.join(base_dir, mode)
    image_list = get_files(base_dir, '.png')
    #remove "HRNet-Semantic-Segmentation/data/cityscapes/ from the img_path
    image_list = [rescope_image_path(img_path) for img_path in image_list]
    return image_list

def output_image_list(image_list, list_file):
    with open(list_file, "w") as fp:
        for img_path in image_list:
            fp.write(img_path + "\n")

if __name__ == "__main__":
    #show_label_info()
    train_image_list = bdd100k_generate_image_list("train")
    output_image_list(train_image_list, "test.lst")
