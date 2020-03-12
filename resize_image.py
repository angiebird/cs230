import os

from PIL import Image

def downsample_image(original_path):
    original_list = os.listdir(original_path)
    for original in original_list:
        name = os.path.basename(original)
        path = os.path.join(original_path, name)
        if name != "resize" and os.path.isdir(path):
            image_list = os.listdir(path)
            for image in image_list:
                img_name = os.path.basename(image)
                original_image_path = os.path.join(path, img_name)
                original_img = Image.open(original_image_path)
                downsample_img = resize_image(original_img)
                downsample_path = os.path.join(original_path, "resize", img_name)
                downsample_img.save(downsample_path)
                print(downsample_path)
        elif not os.path.isdir(path):
            original_img = Image.open(path)
            downsample_img = resize_image(original_img)
            downsample_path = os.path.join(original_path, "resize", name)
            downsample_img.save(downsample_path)
            print(downsample_path)

def resize_image(image):
    width, height = image.size
    resize_ratio = 0.5
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    return image.convert('RGB').resize(target_size, Image.NEAREST)

if __name__ == "__main__":
    downsample_image("data/bdd100k/seg/labels/train_id20")
    downsample_image("data/bdd100k/video_images/train")
    downsample_image("data/bdd100k/hrnet_output_id20")
    pass