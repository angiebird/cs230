import os
def get_bdd100k_download_list(list_file):
    train_list = []
    val_list = []
    test_list = []
    with open(list_file) as fp:
        for line in fp:
            line = line[:-1]  # remove \n
            word_ls = line.split(".")
            word_ls = word_ls[0].split("_")
            mode = word_ls[2]
            idx = int(word_ls[3])
            if mode == "train":
                train_list.append(line)
            elif mode == "val":
                val_list.append(line)
            elif mode == "test":
                test_list.append(line)
    return train_list, val_list, test_list

def download_file(filename):
    print("== Download " + filename)
    base_url = "http://dl.yf.io/bdd-data/bdd100k/video_parts/"
    url = base_url + filename
    base_dir = "tmp/"
    file_path = os.path.join(base_dir, filename)
    if  os.path.exists(file_path):
        print(file_path + " exists")
    else:
        cmd = "wget " + url + " -P " + base_dir
        print(cmd)
        os.system(cmd)
    return file_path

def get_base_filename(filename):
    return filename.split(".")[0]

def process_zip_file(zip_file, mode):
    cmd = "unzip " + zip_file
    print(cmd)
    os.system(cmd)
    os.remove(zip_file)
    video_path = os.path.join("bdd100k/videos/", mode)
    return video_path

def keep_video_with_seg(video_path, seg_hash):
    video_list = os.listdir(video_path)
    keep_video_list = []
    remove_video_list = []
    for video in video_list:
        video_base = get_base_filename(video)
        if video_base in seg_hash:
            keep_video_list.append(video)
        else:
            remove_video_list.append(video)
            remove_video_file = os.path.join(video_path, video)
            os.remove(remove_video_file)
    print("keep", len(keep_video_list), "remove", len(remove_video_list))

def get_seg_img_list(mode):
    path = "/home/ubuntu/cs230/data/bdd100k/seg/images"
    img_dir = os.path.join(path, mode)
    img_list = os.listdir(img_dir)
    return img_list

def get_seg_hash(mode):
    img_list = get_seg_img_list(mode)
    # remove the file prefix
    hash_list = [get_base_filename(img) for img in img_list]
    seg_hash = {}
    for h in hash_list:
        seg_hash[h] = h
    return seg_hash


if __name__ == "__main__":
    list_file = "bdd100k_download_list.txt"
    mode = "train"
    train_list, val_list, test_list = get_bdd100k_download_list(list_file)
    for train_file in train_list[:10]:
        file_path = download_file(train_file)
        video_path = process_zip_file(file_path, mode)
        seg_hash = get_seg_hash(mode)
        keep_video_with_seg(video_path, seg_hash)

