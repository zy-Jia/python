# -*- coding: utf-8 -*-
import os
from PIL import Image
import json
import ast
import numpy as np

np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
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
            
    def fileResize(self, width, height, subPath):
        ##################widthxheight######################
        images = []
        count = 0
        files = os.listdir(self.filePath + "/" + subPath)
        
        for file in files:
            if count == 3:
                break;
            count += 1
            im = Image.open(self.filePath + "/" + subPath + '/' + file)
            imRe = im.resize((width, height))
            #imRe.save(self.filePath + "/" + subClass + '/' + file)
            images.append(np.asarray(imRe, np.float32))
        label = file[0]
        #print(label)
        return np.asarray(images, np.float32), label
    
    
    
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
        for folder in subFolder:
            t, l = self.fileResize(320, 180, folder)
            train.append(t)
            label.append(l)
        return train, label
    
def generator(x_data,y_data,size):
	while True:
		for i in range(size):
			x=x_data[i*size:(i+1)*size-1]
			y=y_data[i*size:(i+1)*size-1]
			yield x,y
        
def main():
    # eventType = {"??????" : 0, "??????" : 1, "??????" : 2}
    # filaPath = "data/train_jpg"
    # file = File(filaPath, eventType)
    
    

    # model = Sequential()
    # model.add(TimeDistributed(Conv2D(2, (4,4), activation= 'relu'), input_shape=(3,180,320,3)))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    # model.add(TimeDistributed(Flatten()))
    # model.add(LSTM(256))
    # model.add(Dense(3, activation='softmax'))
    
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    
    # # epochs = 16

    # # lr = 0.005
    # # sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    # # model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # # model.summary()

    
    # print(model.summary())
    
    # x, y = file.file2Train()
    # x = np.asarray(x, np.float64)
    # y = np.asarray(y, np.int32)
    # num_example = x.shape[0]
    # arr = np.arange(num_example)
    # np.random.shuffle(arr)
    # data=x[arr]
    # data_label=y[arr]
    
    # ratio = 0.8
    # s = np.int(num_example * ratio)
    # x_train = data[:s]
    # y_train = data_label[:s]
    # x_test = data[s:]
    # y_test = data_label[s:]
    
    # model.fit(x_train, y_train, batch_size=32, epochs=16)
    # loss, acc = model.evaluate(x_test, y_test, verbose=1)
    # print('loss: %f, acc: %f' % (loss, acc))
    
    # model.save('model.h5')

    model = load_model('0.29model.h5')
    
    prePath = "data/amap_traffic_test_0712"
    subFolder = os.listdir(prePath)
    pre = []
    for folder in subFolder:
        #print(folder)
        images = []
        count = 0
        files = os.listdir(prePath + "/" + folder)
        
        for file in files:
            if count == 3:
                break;
            count += 1
            im = Image.open(prePath + "/" + folder + '/' + file)
            imRe = im.resize((320, 180))
            #imRe.save(self.filePath + "/" + subClass + '/' + file)
            images.append(np.asarray(imRe, np.float32))
        xPre = []
        xPre.append(np.asarray(images, np.float32))
        yhat = model.predict(np.asarray(xPre, np.float64))
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
    
    # result ???????????????, key???id, value???status
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