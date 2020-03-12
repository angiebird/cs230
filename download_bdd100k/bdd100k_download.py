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

def get_base_without_ext(filepath):
    return get_basename(filepath).split(".")[0]

def get_basename(filepath):
    return os.path.basename(filepath)

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
        video_base = get_base_without_ext(video)
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
    hash_list = [get_base_without_ext(img) for img in img_list]
    seg_hash = {}
    for h in hash_list:
        seg_hash[h] = h
    return seg_hash

def download():
    list_file = "bdd100k_download_list.txt"
    mode = "train"
    train_list, val_list, test_list = get_bdd100k_download_list(list_file)
    for train_file in train_list[:10]:
        file_path = download_file(train_file)
        video_path = process_zip_file(file_path, mode)
        seg_hash = get_seg_hash(mode)
        keep_video_with_seg(video_path, seg_hash)

def get_img_from_video(video_file, seg_hash, idx, out_dir):
    outfile = seg_hash + "_" + str(idx) + ".png"
    out_path = os.path.join(out_dir, outfile)
    #idx / 100. should be the time in seconds where this frame is at
    cmd = "ffmpeg -ss 00:00:" + str(idx / 100.) + " -i " + video_file + " -vframes 1 " + out_path
    print(cmd)
    os.system(cmd)

def get_video_images(video_dir, img_base_dir):
    index_list = range(900, 1100, 5) # dump image from 9th second to 11th second with frequency one image per 0.05 sec
    for video in os.listdir(video_dir):
        if video.endswith(".mov"):
            seg_hash = get_base_without_ext(video)
            video_path = os.path.join(video_dir, video)
            img_dir = os.path.join(img_base_dir, seg_hash)
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            for idx in index_list:
                get_img_from_video(video_path, seg_hash, idx, img_dir)

def get_seg_hash_list(video_dir):
    video_ls = os.listdir(video_dir)
    seg_hash_list = [it.split(".")[0] for it in video_ls]
    for seg_hash in seg_hash_list:
        print(seg_hash)
    return seg_hash_list

if __name__ == "__main__":
    video_dir = "/home/ubuntu/cs230/data/bdd100k/videos/train"
    img_dir = "/home/ubuntu/cs230/data/bdd100k/video_images/train"
    get_seg_hash_list(video_dir)
    #get_video_images(video_dir, img_dir)
