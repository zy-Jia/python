# -*- coding: utf-8 -*-
import os
from PIL import Image
import json
import ast
import numpy as np

np.random.seed(1337)

# from keras.models import Sequential
# from keras.layers import Dropout
# from keras.layers import BatchNormalization
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, MaxPooling2D ,Dense, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import load_model

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
                #print(i['key_frame'])
                key_frame.append(i['key_frame'])
            
            images = []
            
            files = os.listdir(self.filePath + "/" + subPath)
            
            for file in files:
                if file.split('_')[-1] == key_frame[no]:
                    print(file.split('_')[-1], key_frame[no])
                    print(no)
                    im = Image.open(self.filePath + "/" + subPath + '/' + file)
                    imRe = im.resize((320, 180))
                    #imRe.save(self.filePath + "/" + subClass + '/' + file)
                    images.append(np.asarray(imRe, np.float32))
                    #print(np.asarray(imRe, np.float32))
                    break;
            no += 1
        label = file[0]
        print(label)
        return np.asarray(images[0], np.float64), label
    
    def file2Train(self):
        train = []
        label = []
        subFolder = os.listdir(self.filePath)
        #####test
        #t, l = self.fileResize(320, 180, "000067")
        ##train.append(self.fileResize(320, 180, "000067"))
        #train.append(t)
        #label.append(l)
        #print(train)
        #print(label)
        no = 0
        for folder in subFolder:
            t, l = self.fileResize(320, 180, folder, no)
            train.append(t)
            label.append(l)
            no += 1
        return train, label
    
def main():
    eventType = {"畅通" : 0, "缓行" : 1, "拥堵" : 2}
    filaPath = "data/train_jpg"
    file = File(filaPath, eventType)

    # model = Sequential()
    # model.add(TimeDistributed(Conv2D(2, (4,4), activation= 'relu'), input_shape=(1,180,320,3)))
    # model.add(TimeDistributed(BatchNormalization()))
    # model.add(TimeDistributed(Dropout(0.5)))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    # model.add(TimeDistributed(BatchNormalization()))
    # model.add(TimeDistributed(Dropout(0.5)))

    # model.add(TimeDistributed(Flatten()))
    # model.add(LSTM(256))
    # model.add(Dense(3, activation='softmax'))
    
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # model.summary()

    # print(model.summary())
    
    x, y = file.file2Train()
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.int32)
    num_example = x.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data=x[arr]
    data_label=y[arr]
    data=x/255
    data_label=np_utils.to_categorical(y, num_classes=3)
    
    ratio = 0.8
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = data_label[:s]
    x_test = data[s:]
    y_test = data_label[s:]
    #y_train = np_utils.to_categorical(y_train, num_classes=3)
    #y_test = np_utils.to_categorical(y_test, num_classes=3)
    model = Sequential()
    #CNN Layer - 1
    model.add(Convolution2D(
        filters=32, 
        kernel_size= (2, 2), 
        padding= 'same', 
        input_shape=(180, 320, 3),
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2,2),
        padding='same',
    ))

    #CNN Layer - 2
    model.add(Convolution2D(
        filters=64,
        kernel_size=(2, 2),
        padding='same',
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2,2),
        padding='same',
    ))

    # Fully connected Layer -1
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    # Fully connected Layer -2
    model.add(Dense(512))
    model.add(Activation('relu'))
    # Fully connected Layer -3
    model.add(Dense(256))
    model.add(Activation('relu'))
    # Fully connected Layer -4
    model.add(Dense(3))
    model.add(Activation('softmax'))
    # Define Optimizer
    adam = Adam(lr = 0.001)
    #Compile the model
    model.compile(optimizer=adam,
                 loss="categorical_crossentropy",
                 metrics=['accuracy']
                 )
    print(model.summary())
    # Fire up the network
    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=1,
    )
    
    
    # model.fit(x_train, y_train, batch_size=32, epochs=16)
    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print('loss: %f, acc: %f' % (loss, acc))

    model.save('model.h5')

    model = load_model('model.h5')
    
    prePath = "data/amap_traffic_test_0712"
    subFolder = os.listdir(prePath)
    pre = []
    no = 0
    for folder in subFolder:
        
        #print(folder)
        with open("data/amap_traffic_annotations_test.json") as f:
            result = f.read()
            dic= json.loads(result)
            data = ast.literal_eval(str(dic))
            dataAll = list(data['annotations'])
            data = ast.literal_eval(str(dataAll))
            key_frame = []
            for i in data:
                #print(i['key_frame'])
                key_frame.append(i['key_frame'])
            
            images = []
            print(no)
            files = os.listdir("data/amap_traffic_test_0712" + "/" + folder)
            for file in files:
                
                if file.split('_')[-1] == key_frame[no]:
                    print(file.split('_')[-1], key_frame[no])
                    im = Image.open("data/amap_traffic_test_0712" + "/" + folder + '/' + file)
                    imRe = im.resize((320, 180))
                    #imRe.save(self.filePath + "/" + subClass + '/' + file)
                    images.append(np.asarray(imRe, np.float32))
                    #print(np.asarray(imRe, np.float32))
                    break;
            no += 1
        xPre = []
        xPre=np.asarray(images, np.float32)
        yhat = model.predict(np.asarray(xPre, np.float64))
        pre.append(yhat)
    m = []
    with open("pre.txt", "a") as f: 
        for i in pre:
            #print(i)
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
    
        
if __name__ == "__main__":
    main()