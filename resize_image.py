import os
import cv2
from PIL import Image

def downsample_image(original_path):
    orignal_list = os.listdir(original_path)
    for original in orignal_list:
        img_name = os.path.basename(original)
        orignal_img_path = os.path.join(original_path, img_name)
        orignal_img = cv2.imread(orignal_img_path)
        downsample_img = resize_image(orignal_img)

        downsample_path = os.path.join(original_path, "resize", img_name)
        downsample_img.save(downsample_path)

def resize_image(image)
    width, height = image.size
    resize_ratio = 0.5
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

if __name__ == "__main__":
    downsample_image("cs230/data/bdd100k/seg/labels/train_id20")
    downsample_image("cs230/data/bdd100k/video_images/train/")
    downsample_image("data/bdd100k/hrnet_output_id20")
    pass