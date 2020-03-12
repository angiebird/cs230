import os
import matplotlib.image as mpimg
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
    index_list = range(900, 1005, 5) # time index before 10th second 

    # ground truth (gt) label
    gt_dir = "data/bdd100k/seg/labels/train_id20"
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
    video_image_base_dir = "data/bdd100k/video_images/train/"
    video_image_dir = os.path.join(video_image_base_dir, seg_hash)
    video_image_list = [os.path.join(video_image_dir, imgfile) for imgfile in image_list]
    #print(video_image_list[0])

    # hrnet output
    hrnet_dir = "data/bdd100k/hrnet_output_id20"
    hrnet_image_list = [os.path.join(hrnet_dir, imgfile) for imgfile in image_list]

    X = []
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
        X = X[0:100]
        Y = Y[0:100]
    return {"X": X, "Y": Y, "data_size": X.shape[0], "Tx": X.shape[1], "feature_dim": X.shape[2], "num_classes":20}

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

    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
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

def test_training():
    # load video data
    video_data  = load_video_data("0555945c-a5a83e97", test = True)
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

    num_hiden_states = 64

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
if __name__ == "__main__":
    test_training()
