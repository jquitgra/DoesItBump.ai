import argparse
import os
import numpy as np
import tensorflow.keras
from PIL import Image
import h5py
import glob
import random
import cv2
import csv
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras import models
from keras.utils import to_categorical
import pickle
from sklearn.externals import joblib 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import pickle
from sklearn.externals import joblib 

def ShapeData(IMG_DIR:str, NEWDATA_DIR:str):->bool:
    #getting the directory of dataset
    img_row, img_cols = 200, 200

    #keep track of deleted images
    del_imgs = set()
    for file in os.listdir(IMG_DIR):
        im = Image.open(IMG_DIR+'\\'+file)
        if(im.size >= (200, 200)):  
            #resize images
            img = im.resize((img_row, img_cols))
            #convert to grayscale
            gray = im.convert('L')
            gray.save(NEWDATA_DIR+'\\'+file,"JPEG")
        else:
            del_imgs.add(file)
            continue
    return True

def CNN_Modelfy(x_train,y_train):
    x_train = np.array(x_train)
    x_train = x_train/255.0 #normalize
    
    
    y_train = np.array(y_train)
    
    
    #start
    model = Sequential()
    #insert a conv layer that accepts the image data x2
    
    ### READ THIS ####
    #You had an error in the input shape, where you se (50, 50 ,1) and it should be (50, 50, 3)
    # remember you need to be consisted with the size
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100,3)))
    
    #max pool layer for further processing x2
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
    model.add(keras.layers.MaxPooling2D((2,2)))
    
    #flatten data for full-connected layer
    model.add(keras.layers.Flatten())
    
    #regularization
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(512,activation='relu'))
    model.add(keras.layers.Dense(2,activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    #ntrain = len(x_train)
    #nval = len(x_test)
    
    
    model.fit(x_train,y_train,epochs=1,batch_size=32)

    model.save('CNN_testerSwamp.sav')

def h5pyPrepper(IMG_DIR, DATA_DIR, NEWDATA_DIR, h5pyFile):
    fileList = []

    for rot, dirs, files in os.walk(NEWDATA_DIR):
        for file in files:
            if(file.endswith("jpg") or file.endswith("JPG")):
                temp = str(NEWDATA_DIR+file)
                fileList.append(temp)
    #return fileList
    thisDict = dict()
    files = os.listdir(DATA_DIR)
    with open(DATA_DIR+'\\'+'500_picts_satz.csv', mode = 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        with open(DATA_DIR+'\\'+'new_data.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            thisDict = {IMG_DIR+'\\'+row[1]:row[2] for row in csv_reader if row[2] == 'happiness' or row[2] == 'neutral'}
    happiness = []
    boredom = []
    for key in thisDict:
        if(thisDict[key] == 'happiness'):
            happiness.append(key)
        else:
            boredom.append(key)
    hapAlt = random.sample(happiness, len(boredom))
    boreAlt = random.sample(boredom, len(boredom))

    label_bumps = np.ones(shape=(len(hapAlt),), dtype=np.uint8).tolist()
    label_trash = np.zeros(shape=(len(boreAlt),), dtype = np.uint8).tolist()

    img_addr = hapAlt
    img_label = label_bumps + label_trash

    matches = list(zip(img_addr, img_label))
    random.shuffle(matches)
    (addrs, labels) = zip(*matches)
    train_addr = list(addrs[0:int(len(addrs))])
    train_labels = list(labels[0:int(len(labels))])
    img_shape = (len(img_addr),100, 100, 3)
    f = h5py.File(test, mode="w")
    f.create_dataset("train_img", img_shape, np.float32)
    f.create_dataset("train_label", (len(train_labels), ), np.float32)
    f["train_label"][...] = train_labels
    #bad_indexes = []
    for i in range(len(img_addr)):
    
        addr_read = img_addr[i]
        img = cv2.imread(addr_read)
        img = cv2.resize(img, (100,100), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        f["train_img"][i, ...] = img[None]
    x,y = f.values()
    CNN_Modelfy(x,y)

def CNN_trainer(hyp5File):
    f = h5py.File(test, mode="r")
    x, y = f.values()
    print(x.shape)
    print(y.shape)
    y_binary = to_categorical(y)
    print(y_binary)
    print(x)
    CNN_training(x,y_binary)

def validator(imagePath:str):
    img = keras.preprocessing.image.load_img(imagePath)
    array = keras.preprocessing.image.img_to_array(img)
    resizedAr = cv2.resize(array(100,100))
    inputAr = resizedAr[np.newaxis,...]
    net = tf.keras.models.load_model(r'C:\Users\JohnQ\Documents\Hackathons\SwampHacks2020\CNN_testerSwamp.sav')

    prediction = net.predict_classes(inputAr)[0]

    return prediction

def main():
    IMG_DIR = "C:\\Users\\JohnQ\\Documents\\Hackathons\\SwampHacks2020\\images"
    DATA_DIR = "C:\\Users\\JohnQ\\Documents\\Hackathons\\SwampHacks2020\\data"
    NEWDATA_DIR = "C:\\Users\\JohnQ\\Documents\\Hackathons\\SwampHacks2020\\resized_data"
    hypy = r"./dsSwampers.hdf5"
    image = "insert path here"
    ShapeData(IMG_DIR, NEWDATA_DIR)
    h5pyPrepper(IMG_DIR, DATA_DIR, NEWDATA_DIR, hypy)
    CNN_trainer(hypy)
    print(validator(image))

if __name__ == '__main__':
    main()

    

