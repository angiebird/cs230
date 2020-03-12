import os
import matplotlib.image as mpimg
from PIL import Image
import cv2
import run_hrnet as hr
import numpy as np
import pickle
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from  keras.layers import CuDNNLSTM
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

def read_seg_hash_list(file_path = "seg_hash_list.txt"):
    seg_hash_list = []
    with open(file_path) as fp:
        for line in fp:
            seg_hash_list.append(line.split()[0])
    return seg_hash_list

def get_train_list():
    seg_hash_list = read_seg_hash_list()
    return seg_hash_list[:200]

def get_val_list():
    seg_hash_list = read_seg_hash_list()
    return seg_hash_list[200:250]

def get_test_list():
    seg_hash_list = read_seg_hash_list()
    return seg_hash_list[250:300]

def get_image_time_idx(image_name):
    # An example of image_name 19cd9c06-f7cb2ecd_965.png
    words = image_name.split(".")
    time_idx = int(words[0].split("_")[-1])
    return time_idx

def label_to_one_hot(label):
    # convert id20 to one hot
    label = label.flatten()
    one_hot = to_categorical(label, num_classes = 20)
    return one_hot

def load_video_data(seg_hash, test = False):
    index_list = range(975, 1005, 5) # time index before 10th second 

    # ground truth (gt) label
    gt_dir = "data/bdd100k/seg/labels/train_id20/resize"
    gt_label_file = os.path.join(gt_dir, seg_hash + "_train_id.png")
    #print(gt_label_file)
    gt_label = hr.read_label_img(gt_label_file)
    one_hot_gt_label = label_to_one_hot(gt_label)
    Y = one_hot_gt_label
    #print(one_hot_gt_label)
    #print(one_hot_gt_label.shape)

    # image list
    image_list = [seg_hash + "_" + str(idx) + ".png" for idx in index_list]

    # video image
    video_image_dir = "data/bdd100k/video_images/train/resize"
    video_image_list = [os.path.join(video_image_dir, imgfile) for imgfile in image_list]
    #print(video_image_list[0])

    # hrnet output
    hrnet_dir = "data/bdd100k/hrnet_output_id20/resize"
    hrnet_image_list = [os.path.join(hrnet_dir, imgfile) for imgfile in image_list]

    X = []
    image_size = mpimg.imread(video_image_list[0]).shape[0:2]
    for i in range(len(video_image_list)):
        img = mpimg.imread(video_image_list[i])
        rgb = img.reshape(-1, 3)
        hr_label = hr.read_label_img(hrnet_image_list[i])
        one_hot_hr_label = label_to_one_hot(hr_label)
        X.append(np.concatenate((one_hot_hr_label, rgb), axis=1))
        #print(one_hot_hr_label.shape)
        #print(rgb.shape)
        #print(out.shape)
    X = np.array(X)
    X = np.swapaxes(X,0,1)
    if test:
        X = X[0:10]
        Y = Y[0:10]
    return {"X": X, "Y": Y, "data_size": X.shape[0], "Tx": X.shape[1], "feature_dim": X.shape[2],
            "num_classes":20, "image_size": image_size, "num_hiden_states": 64}

def load_multiple_videos(seg_hash_list, test = False):
    total_data = load_video_data(seg_hash_list[0], test = test)
    X_list = []
    Y_list = []
    for seg_hash in seg_hash_list[1:]:
        video_data = load_video_data(seg_hash, test = test)
        X_list.append(video_data["X"])
        Y_list.append(video_data["Y"])

    total_data["X"] = np.concatenate(X_list, axis = 0)
    total_data["Y"] = np.concatenate(Y_list, axis = 0)
    total_data["data_size"] = total_data["X"].shape[0]
    return total_data

def build_lstm_model(Tx, num_hiden_states, feature_dim, num_classes):
    X = Input(shape=(Tx, feature_dim))

    # What exactly is return_state?
    LSTM_cell = CuDNNLSTM(num_hiden_states, return_state = True)

    a0 = Input(shape=(num_hiden_states,), name='a0')
    c0 = Input(shape=(num_hiden_states,), name='c0')
    a = a0
    c = c0

    outputs = []

    for t in range(Tx):
        x = Lambda(lambda z: z[:,t,:])(X)
        x = Reshape((1, feature_dim))(x)
        a, _, c = LSTM_cell(inputs = x, initial_state = [a, c])
        if t == Tx - 1:
            # we only have labels at the last frame
            out = Dense(num_classes, activation='softmax')(a)
            outputs.append(out)

    model = Model(inputs = [X, a0, c0], outputs = outputs)

    opt = Adam(lr=0.05, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def load_history(name):
    weight_dir = "model_weight"
    history_path = os.path.join(weight_dir, name + "_history.pkl")
    with open(history_path, 'rb') as fp:
      return pickle.load(fp)
    return None 

def save_history(name, history):
    weight_dir = "model_weight"
    history_path = os.path.join(weight_dir, name + "_history.pkl")
    with open(history_path, 'wb') as fp:
        pickle.dump(history, fp)

def load_weight(model, name):
    weight_dir = "model_weight"
    weight_path = os.path.join(weight_dir, name + ".h5")
    model.load_weights(weight_path)
    return model

def save_weight(model, name):
    weight_dir = "model_weight"
    weight_path = os.path.join(weight_dir, name + ".h5")
    model.save_weights(weight_path)
    return model

def test_load_multiple_videos():
    seg_hash_list = ["0555945c-a5a83e97", "064c84ab-5560b5a4"]
    video_data  = load_multiple_videos(seg_hash_list, test = True)

    X = video_data["X"]
    Y = video_data["Y"]
    Tx = video_data["Tx"]
    m = video_data["data_size"]
    feature_dim = video_data["feature_dim"]
    num_classes = video_data["num_classes"]

    print("X.shape: ", X.shape)
    print("Y.shape: ", Y.shape)
    print("Tx: ", Tx)
    print("m:  ", m)
    print("feature_dim:  ", feature_dim)
    print("num_classes:  ", num_classes)


def test_training():
    # load video data
    video_data  = load_video_data("0555945c-a5a83e97", test = False)
    X = video_data["X"]
    Y = video_data["Y"]
    Tx = video_data["Tx"]
    m = video_data["data_size"]
    feature_dim = video_data["feature_dim"]
    num_classes = video_data["num_classes"]

    print("X.shape: ", X.shape)
    print("Y.shape: ", Y.shape)
    print("Tx: ", Tx)
    print("m:  ", m)
    print("feature_dim:  ", feature_dim)
    print("num_classes:  ", num_classes)

    num_hiden_states = video_data["num_hiden_states"]

    #create model
    model = build_lstm_model(Tx, num_hiden_states, feature_dim, num_classes)
    #model.summary()

    #training
    a0 = np.zeros((m, num_hiden_states))
    c0 = np.zeros((m, num_hiden_states))
    history = model.fit([X, a0, c0], Y, epochs = 2)
    print(model.evaluate([X, a0, c0], Y))
    print(history.history)

    save_weight(model, "test")
    save_history("test", history)

    new_model = build_lstm_model(Tx, num_hiden_states, feature_dim, num_classes)

    load_weight(new_model, "test")
    new_history = load_history("test")
    #print(new_model.evaluate([X, a0, c0], Y))
    print(new_history.history)

def predict_label(model, X, num_hiden_states):
    m = X.shape[0]
    a0 = np.zeros((m, num_hiden_states))
    c0 = np.zeros((m, num_hiden_states))
    one_hot = model.predict([X, a0, c0])
    label = np.argmax(one_hot, axis = 1)
    label = label.astype("uint8")
    return label

def flatten_label_to_img(flatten_label, image_size):
    img_label = flatten_label.astype("uint8") #this step is important for saving image using PIL
    img_label = img_label.reshape(image_size)
    return img_label

def test_prediction():
    seg_hash = "0555945c-a5a83e97"
    video_data  = load_video_data(seg_hash, test = False)
    X = video_data["X"]
    Y = video_data["Y"]
    Tx = video_data["Tx"]
    feature_dim = video_data["feature_dim"]
    num_classes = video_data["num_classes"]
    num_hiden_states = video_data["num_hiden_states"]

    model = build_lstm_model(Tx, num_hiden_states, feature_dim, num_classes)

    load_weight(model, "test")

    label = predict_label(model, X, num_hiden_states)
    label = flatten_label_to_img(label, video_data["image_size"])

    print(label.shape, label.dtype)
    print(label.max(), label.min())
    hr.save_label(label, "pre_label.png")
    hr.save_color_image(label, "pre_color.png")

    hr_label = get_hr_label(seg_hash)
    hr.save_color_image(hr_label, "hr.png")

    gt_label = get_gt_label(seg_hash)
    hr.save_color_image(gt_label, "gt.png")

    #print(new_model.evaluate([X, a0, c0], Y))

def get_gt_label(seg_hash):
    gt_dir = "data/bdd100k/seg/labels/train/"
    gt_label_file = os.path.join(gt_dir, seg_hash + "_train_id.png")
    label = hr.read_label_img(gt_label_file)
    return label

def get_hr_label(seg_hash, time_idx = 1000):
    hrnet_dir = "data/bdd100k/hrnet_output_id20/resize/"
    label_file = os.path.join(hrnet_dir, seg_hash + "_"+ str(1000) + ".png")
    label = hr.read_label_img(label_file)
    return label

def train_model_v1():
    version = "v1"
    train_seg_hash_list = get_train_list()
    video_data  = load_multiple_videos(train_seg_hash_list, test = False)

    X = video_data["X"]
    Y = video_data["Y"]
    Tx = video_data["Tx"]
    m = video_data["data_size"]
    feature_dim = video_data["feature_dim"]
    num_classes = video_data["num_classes"]
    num_hiden_states = video_data["num_hiden_states"]

    print("X.shape: ", X.shape)
    print("Y.shape: ", Y.shape)
    print("Tx: ", Tx)
    print("m:  ", m)
    print("feature_dim:  ", feature_dim)
    print("num_classes:  ", num_classes)

    model = build_lstm_model(Tx, num_hiden_states, feature_dim, num_classes)

    a0 = np.zeros((m, num_hiden_states))
    c0 = np.zeros((m, num_hiden_states))

    name = version + "_" + str(0)
    save_weight(model, name)

    #training
    for idx in range(1, 10):
        print("=== training idx", idx)
        history = model.fit([X, a0, c0], Y, epochs = 1)
        print(history.history)
        name = version + "_" + str(idx)
        save_weight(model, name)
        save_history(name, history)

def evaluate_model_v1():
    version = "v1"
    train_seg_hash_list = get_val_list()
    video_data  = load_multiple_videos(train_seg_hash_list, test = False)

    X = video_data["X"]
    Y = video_data["Y"]
    Tx = video_data["Tx"]
    m = video_data["data_size"]
    feature_dim = video_data["feature_dim"]
    num_classes = video_data["num_classes"]
    num_hiden_states = video_data["num_hiden_states"]

    print("X.shape: ", X.shape)
    print("Y.shape: ", Y.shape)
    print("Tx: ", Tx)
    print("m:  ", m)
    print("feature_dim:  ", feature_dim)
    print("num_classes:  ", num_classes)

    model = build_lstm_model(Tx, num_hiden_states, feature_dim, num_classes)

    a0 = np.zeros((m, num_hiden_states))
    c0 = np.zeros((m, num_hiden_states))

    name = version + "_" + str(3)
    load_weight(model, name)
    print(model.evaluate(x = [X, a0, c0], y=Y))


if __name__ == "__main__":
    #evaluate_model_v1()
    #train_model_v1()
    #test_prediction()
    #test_training()
    #test_load_multiple_videos()
