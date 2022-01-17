import keras.backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import concatenate, BatchNormalization, Flatten, Dropout
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import load_model

L2_RATE = 0.002
NUM_CLASS = 2
BATCH_SIZE = 128
EPOCH = 10

import os
from PIL import Image
import json
import ast
import numpy as np

class File(object):
    def __init__(self, filePath, eventType):
        self.filePath = filePath
        self.eventType  = eventType
        
    def fileRename(self):
        ##################status_group_no.jpg######################
        with open("data/amap_traffic_annotations_train.json") as f:
            result = f.read()
        dic= json.loads(result)
        data = ast.literal_eval(str(dic))
        dataAll = list(data['annotations'])
        data = ast.literal_eval(str(dataAll))
        status = []
        for i in data:
            status.append(i['status'])
        count = 0
        subFolder = os.listdir(self.filePath)
        for subClass in subFolder:
            print(subClass)
            files = os.listdir(self.filePath + "/" + subClass)
            for file in files:
                os.rename(self.filePath + "/" + subClass + "/" + file, self.filePath + "/" + subClass + "/" + str(status[count]) + "_" + subClass + '_' + file.split('.')[0] + ".jpg")
            count += 1
            
    def fileResize(self, width, height, subPath, no):
        ##################widthxheight######################
        with open("data/amap_traffic_annotations_train.json") as f:
            result = f.read()
            dic= json.loads(result)
            data = ast.literal_eval(str(dic))
            dataAll = list(data['annotations'])
            data = ast.literal_eval(str(dataAll))
            key_frame = []
            for i in data:
                key_frame.append(i['key_frame'])
            
            images = []
            
            files = os.listdir(self.filePath + "/" + subPath)
            
            for file in files:
                if file.split('_')[-1] == key_frame[no]:
                    print(file.split('_')[-1], key_frame[no])
                    print(no)
                    im = Image.open(self.filePath + "/" + subPath + '/' + file)
                    imRe = im.resize((320, 180))
                    images.append(np.asarray(imRe, np.float32))
                    break;
            no += 1
        label = file[0]
        if label == "2":
            label = "1"
        print(label)

        return np.asarray(images[0], np.float64), label
    
    def file2Train(self):
        train = []
        label = []
        subFolder = os.listdir(self.filePath)
        no = 0
        for folder in subFolder:
            t, l = self.fileResize(320, 180, folder, no)
            train.append(t)
            label.append(l)
            no += 1
        return train, label
    
def inception(x, filter_size, layer_number):
    layer_number = str(layer_number)
    with K.name_scope('Inception_' + layer_number):
        with K.name_scope("conv_1x1"):
            conv_1x1 = Conv2D(filters=filter_size[0], kernel_size=(1, 1),
                              strides=1, padding='same', activation='relu',
                              kernel_regularizer=l2(L2_RATE),
                              name='conv_1x1' + layer_number)(x)
        with K.name_scope('conv_3x3'):
            conv_3x3 = Conv2D(filters=filter_size[1], kernel_size=(1, 1),
                              strides=1, padding='same', activation='relu',
                              kernel_regularizer=l2(L2_RATE),
                              name='conv_3x3_bottleneck' + layer_number)(x)
            conv_3x3 = Conv2D(filters=filter_size[2], kernel_size=(3, 3),
                              strides=1, padding='same', activation='relu',
                              kernel_regularizer=l2(L2_RATE),
                              name='conv_3x3' + layer_number)(conv_3x3)
        with K.name_scope('conv_5x5'):
            conv_5x5 = Conv2D(filters=filter_size[3], kernel_size=(1, 1),
                              strides=1, padding='same', activation='relu',
                              kernel_regularizer=l2(L2_RATE),
                              name='conv_5x5_bottleneck' + layer_number)(x)
            conv_5x5 = Conv2D(filters=filter_size[4], kernel_size=(5, 5),
                              strides=1, padding='same', activation='relu',
                              kernel_regularizer=l2(L2_RATE),
                              name='conv_5x5' + layer_number)(conv_5x5)
        with K.name_scope('Max_Conv'):
            max_pool = MaxPooling2D(pool_size=3, strides=1, padding='same',
                                    name='maxpool'+layer_number)(x)
            max_pool = Conv2D(filters=filter_size[5], kernel_size=(1, 1),
                              strides=1, padding='same', activation='relu',
                              kernel_regularizer=l2(L2_RATE),
                              name='maxpool_conv1x1' + layer_number)(max_pool)
        with K.name_scope('concatenate'):
            x = concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=-1)  # high/width上相同,channels拼接在一起
    return x


def aux_classifier(x, filter_size, layer_number):
    layer_number = str(layer_number)
    with K.name_scope('aux_ckassifier'+layer_number):
        x = AveragePooling2D(pool_size=3, strides=2, padding='same',
                             name='AveragePooling2D'+layer_number)(x)
        x = Conv2D(filters=filter_size[0], kernel_size=1, strides=1,
                   padding='valid', activation='relu',
                   kernel_regularizer=l2(L2_RATE),
                   name='aux_conv' + layer_number)(x)
        x = Flatten()(x)
        x = Dense(units=filter_size[1], activation='relu',
                  kernel_regularizer=l2(L2_RATE),
                  name='aux_dense1_' + layer_number)(x)
        x = Dropout(0.7)(x)
        x = Dense(units=NUM_CLASS, activation='sigmoid',
                  kernel_regularizer=l2(L2_RATE),
                  name='aux_output' + layer_number)(x)
    return x

def front(x, filter_size):
    x = Conv2D(filters=filter_size[0], kernel_size=5, strides=1,
               padding='same', activation='relu',
               kernel_regularizer=l2(L2_RATE))(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=filter_size[1], kernel_size=1, strides=1,
               padding='same', activation='relu',
               kernel_regularizer=l2(L2_RATE))(x)
    x = Conv2D(filters=filter_size[2], kernel_size=3, strides=1,
               padding='same', activation='relu',
               kernel_regularizer=l2(L2_RATE))(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
    return x

def load():
    eventType = {"畅通" : 0, "缓行" : 1, "拥堵" : 2}
    filaPath = "data/train_jpg"
    file = File(filaPath, eventType)
    x, y = file.file2Train()
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.int32)
    num_example = x.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data=x[arr]
    data_label=y[arr]
    
    ratio = 0.8
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = data_label[:s]
    x_test = data[s:]
    y_test = data_label[s:]
    y_train = to_categorical(y_train, NUM_CLASS)
    y_test = to_categorical(y_test, NUM_CLASS)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train, y_train, x_test, y_test

def googlenet_model():
    X_Input = Input(shape=input_shape, name='Input')
    X = front(X_Input, [64, 64, 192])
    X = inception(X, filter_size=[64, 96, 128, 16, 32, 32], layer_number=0)
    X = inception(X, [128, 128, 192, 32, 96, 64], layer_number=1)
    X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)
    X = inception(X, [192, 96, 208, 16, 48, 64], layer_number=2)
    aux_output_1 = aux_classifier(X, [128, 1024], layer_number=1)
    X = inception(X, [160, 112, 225, 24, 64, 64], layer_number=3)
    X = inception(X, [128, 128, 256, 24, 64, 64], layer_number=4)
    X = inception(X, [112, 144, 288, 32, 64, 64], layer_number=5)
    aux_output_2 = aux_classifier(X, [128, 1024], layer_number=2)
    X = inception(X, [256, 160, 320, 32, 128, 128], layer_number=6)
    X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)
    X = inception(X, [256, 160, 320, 32, 128, 128], layer_number=7)
    X = inception(X, [386, 192, 384, 48, 128, 128], layer_number=8)
    X = AveragePooling2D(pool_size=4, strides=1, padding='valid')(X)
    X = Flatten()(X)
    X = Dropout(0.4)(X)
    main_output = Dense(NUM_CLASS, activation='sigmoid', kernel_regularizer=l2(L2_RATE))(X)
    model = Model(inputs=X_Input, outputs=[main_output, aux_output_1, aux_output_2])
    return model

def generator(x_data,y_data,size):
	while True:
		for i in range(size):
			x=x_data[i*size:(i+1)*size-1]
			y=y_data[i*size:(i+1)*size-1]
			yield x,[y,y,y]
            
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load()
    input_shape = x_train.shape[1:]
    GoogleNet = googlenet_model()
    optimizer = Adam(epsilon=0.01)
    GoogleNet.compile(optimizer=optimizer, 
                      loss='binary_crossentropy',
                      metrics=['accuracy'], 
                      loss_weights=[1, 0.3, 0.3])
    GoogleNet.summary()
    tfck = TensorBoard(log_dir='logs/GoogleNet')
    GoogleNet.fit_generator(generator(x_train, y_train, 16),
                        steps_per_epoch=60,
                        epochs=EPOCH,
                        callbacks=[tfck],
                        validation_data=generator(x_test, y_test, 16),
                        validation_steps=15,
                        verbose=1)
    # GoogleNet.fit(x=x_train,
    #               y=[y_train, y_train, y_train],
    #               validation_data=(x_test, [y_test, y_test, y_test]),
    #               epochs=EPOCH,
    #               callbacks=[tfck],
    #               batch_size=BATCH_SIZE
    #               )
    
    GoogleNet.save('model.h5')
    model = load_model('model.h5')
    
    prePath = "data/amap_traffic_test_0712"
    subFolder = os.listdir(prePath)
    pre = []
    no = 0
    for folder in subFolder:
        with open("data/amap_traffic_annotations_test.json") as f:
            result = f.read()
            dic= json.loads(result)
            data = ast.literal_eval(str(dic))
            dataAll = list(data['annotations'])
            data = ast.literal_eval(str(dataAll))
            key_frame = []
            for i in data:
                key_frame.append(i['key_frame'])
            
            images = []
            print(no)
            files = os.listdir("data/amap_traffic_test_0712" + "/" + folder)
            for file in files:
                if file.split('_')[-1] == key_frame[no]:
                    print(file.split('_')[-1], key_frame[no])
                    im = Image.open("data/amap_traffic_test_0712" + "/" + folder + '/' + file)
                    imRe = im.resize((320, 180))
                    images.append(np.asarray(imRe, np.float32))
                    break;
            no += 1
        xPre = []
        xPre=np.asarray(images, np.float32)
        yhat = model.predict(np.asarray(xPre, np.float64))
        print(yhat)
        pre.append(yhat)
    m = []
    with open("pre.txt", "a") as f: 
        for i in pre:
            print(i)
            i = list(i[0])
            ind = i.index(max(i))
            print(ind)
            m.append(ind)
    
    json_path = "data/amap_traffic_annotations_test.json"
    out_path = "data/amap_traffic_annotations_test_result.json"
    
    result = {}
    c = 0
    for folder in subFolder:
        result[folder] = m[c]
        c += 1
    
    # result 是你的结果, key是id, value是status
    with open(json_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as w:
        json_dict = json.load(f)
        data_arr = json_dict["annotations"]  
        new_data_arr = [] 
        for data in data_arr:
            id_ = data["id"]
            data["status"] = int(result[id_])
            new_data_arr.append(data)
        json_dict["annotations"] = new_data_arr
        json.dump(json_dict, w)