import os
import random
import cv2
import numpy as np
import pickle
ROWS = 64
COLS = 64
CHANNELS = 3

DIR = '/home/computer-vision-tutorial/data/PetImages'

train_dogs =   [DIR+'/Dog/'+i for i in os.listdir(DIR+'/Dog') if i[-3:]=='jpg'][:1000]
train_cats =   [DIR+'/Cat/'+i for i in os.listdir(DIR+'/Cat') if i[-3:]=='jpg'][:1000]
data = train_dogs + train_cats
random.shuffle(data)
test_images =  data[: int(0.25*len(data))]
train_images = data[int(0.25*len(data)):]


def read_image(file_path):
    img = cv2.imread(file_path) #cv2.IMREAD_GRAYSCALE
    img = cv2.resize(img, (ROWS, COLS))
    return img


def prep_data(images):
    count = len(images)
    train_y = []
    data = []
    for i, image_file in enumerate(images):
        try:
            k = [1,0]
            print(i, image_file)
            image = read_image(image_file)
            data.append(image)
            
            if 'Cat' in image_file:
                k = [0,1]
            train_y.append(np.array(k))
        except:
            pass
            
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
        
    data = np.array(data).astype(np.float32)
    train_y = np.array(train_y).astype(np.float32)
    return data, train_y

x_train, y_train = prep_data(train_images)
x_test, y_test = prep_data(test_images)

pkl_data = {
    'x_train': x_train,
    'y_train': y_train,
    'x_test': x_test,
    'y_test': y_test
}

with open('../data/data.pickle', 'wb') as handle:
    pickle.dump(pkl_data,handle)

print("Train shape: {}".format(x_train.shape))
print("Test shape: {}".format(x_test.shape))
print("train y shape: {}".format(y_train.shape))


