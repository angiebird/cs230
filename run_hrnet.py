import os
import labels
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

def pil_label_to_color_img(label_img):
    palette = set_palette()
    color_img = Image.fromarray(label_img)
    color_img.putpalette(palette)
    return color_img

label_mapping = {-1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18}

def label_to_color_img(label):
    shape = (label.shape[0], label.shape[1], 3)
    color_img = np.zeros(shape, dtype = int)
    for k, v in label_mapping.items():
        color_img[label == k] = np.array(labels.id2label[k].color)
    return color_img.astype("uint8")

def train_id_label_to_color_img(train_id_label):
    label = convert_label(train_id_label, inverse = True)
    return label_to_color_img(label)

def id20_label_to_color_img(label):
    shape = (label.shape[0], label.shape[1], 3)
    color_img = np.zeros(shape, dtype = int)
    for k in labels.id20_to_label.keys():
        color_img[label == k] = np.array(labels.id20_to_label[k].color)
    return color_img.astype("uint8")

#copy from HRNet-Semantic-Segmentation/lib/datasets/cityscapes.py
def convert_label(label, inverse = False):
    new_label = label.copy()
    if inverse:
        # convert train_id to id
        for v, k in label_mapping.items():
            new_label[label == k] = v
    else:
        # convert id to train_id
        for k, v in label_mapping.items():
            new_label[label == k] = v
    return new_label

def convert_train_id_to_id20(label, inverse = False):
    new_label = label.copy()
    if inverse:
        new_label[label == 19] = 255
    else:
        new_label[label == 255] = 19
    return new_label

def convert_label_to_color_image(label_path, color_path):
    #gt_label_path = "cs230/data/cityscapes/gtFine/val/
    label_list = os.listdir(label_path)
    for label in label_list:
        img_name = os.path.basename(label)
        label_img_path = os.path.join(label_path, img_name)
        label_img = read_label_img(label_img_path)
        color_img = pil_label_to_color_img(label_img)

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

def bdd100k_generate_resize_image_list():
    # We link the bdd100 images to cityscapes/video_images
    base_dir = "HRNet-Semantic-Segmentation/data/cityscapes/video_images/train/resize/"
    image_list = get_files(base_dir, '.png')
    #remove "HRNet-Semantic-Segmentation/data/cityscapes/ from the img_path
    image_list = [rescope_image_path(img_path) for img_path in image_list]
    for img in image_list:
        print(img)
    return image_list

def output_image_list(image_list, list_file):
    with open(list_file, "w") as fp:
        for img_path in image_list:
            fp.write(img_path + "\n")

def convert_hr_output_to_use_train_id():
    hr_dir = "data/bdd100k/hrnet_output"
    new_hr_dir = "data/bdd100k/hrnet_output_train_id"
    for label_file in os.listdir(hr_dir):
        label_file_path = os.path.join(hr_dir, label_file)
        label = read_label_img(label_file_path)
        train_id_label = convert_label(label)
        train_id_label = Image.fromarray(train_id_label)
        train_id_label_path = os.path.join(new_hr_dir, label_file)
        train_id_label.save(train_id_label_path)

def convert_labels(convert_func, src_dir, dst_dir):
    for label_file in os.listdir(src_dir):
        label_file_path = os.path.join(src_dir, label_file)
        label = read_label_img(label_file_path)
        new_label = convert_func(label)
        new_label = Image.fromarray(new_label)
        new_label_path = os.path.join(dst_dir, label_file)
        new_label.save(new_label_path)

def convert_hr_output_train_id_to_use_id20():
    src_dir = "data/bdd100k/hrnet_output_train_id"
    dst_dir = "data/bdd100k/hrnet_output_id20"
    convert_labels(convert_train_id_to_id20, src_dir, dst_dir)

def convert_seg_train_id_to_use_id20():
    src_dir = "data/bdd100k/seg/labels/train"
    dst_dir = "data/bdd100k/seg/labels/train_id20"
    convert_labels(convert_train_id_to_id20, src_dir, dst_dir)

def save_color_image(id20_label, file_path):
    color_image = id20_label_to_color_img(id20_label)
    color_image = Image.fromarray(color_image)
    color_image.save(file_path)

def save_label(id20_label, file_path):
    label = Image.fromarray(id20_label)
    label.save(file_path)

def get_gt_label(seg_hash):
    gt_dir = "data/bdd100k/seg/labels/train_id20/resize/"
    gt_label_file = os.path.join(gt_dir, seg_hash + "_train_id.png")
    label = hr.read_label_img(gt_label_file)
    return label

def get_hr_label(seg_hash, time_idx = 1000):
    hrnet_dir = "data/bdd100k/hrnet_output_id20/resize/"
    label_file = os.path.join(hrnet_dir, seg_hash + "_"+ str(1000) + ".png")
    label = hr.read_label_img(label_file)
    return label


if __name__ == "__main__":
    show_label_info()
    #bdd100k_generate_resize_image_list()
    #convert_hr_output_to_use_train_id()
    #convert_hr_output_train_id_to_use_id20()
    #convert_seg_train_id_to_use_id20()
    #train_image_list = bdd100k_generate_image_list("train")
    #output_image_list(train_image_list, "test.lst")
    pass
