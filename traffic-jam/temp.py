# -*- coding: utf-8 -*-
# import json
# import ast
# import os
# from PIL import Image
# import numpy as np
# from keras.models import load_model

# with open("data\\amap_traffic_annotations_train.json") as f:
#     result = f.read()
    
#     #print(type(result))
#     dic= json.loads(result)
#     data = ast.literal_eval(str(dic))
#     #print(dic, type(dic))
    
# dataAll = list(data['annotations'])


# data = ast.literal_eval(str(dataAll))
    
# for i in data:
#     print(i['id'])
    
# filePath = "data/train_jpg"
# subFolder = os.listdir(filePath)
# no = 0
# for folder in subFolder:
#     with open("data/amap_traffic_annotations_train.json") as f:
#         result = f.read()
#         dic= json.loads(result)
#         data = ast.literal_eval(str(dic))
#         dataAll = list(data['annotations'])
#         data = ast.literal_eval(str(dataAll))
#         key_frame = []
#         for i in data:
#             #print(i['key_frame'])
#             key_frame.append(i['key_frame'])
        
#         images = []
        
#         files = os.listdir(filePath + "/" + folder)
        
#         for file in files:
            
#             count = 0
#             if file.split('_')[-1] == key_frame[no]:
#                 print(file.split('_')[-1])
#                 print(key_frame[no])
#                 im = Image.open(filePath + "/" + folder + '/' + file)
#                 imRe = im.resize((320, 180))
#                 #imRe.save(self.filePath + "/" + subClass + '/' + file)
#                 images.append(np.asarray(imRe, np.float32))
#                 #print(np.asarray(imRe, np.float32))
#                 break;
#             else:
#                 count += 1
#         no += 1

# with open("data\\amap_traffic_annotations_train.json") as f:
#     result = f.read()
    
#     #print(type(result))
#     dic= json.loads(result)
#     data = ast.literal_eval(str(dic))
#     #print(dic, type(dic))
    
# dataAll = list(data['annotations'])

# data = ast.literal_eval(str(dataAll))
    
# for i in data:
#     print(i['frames'][0]["gps_time"])
# model = load_model('0.29model.h5')
    
# prePath = "data/amap_traffic_test_0712"
# subFolder = os.listdir(prePath)
# pre = []
# for folder in subFolder:
#     #print(folder)
#     images = []
#     count = 0
#     files = os.listdir(prePath + "/" + folder)
        
#     for file in files:
#         if count == 3:
#             break;
#         count += 1
#         im = Image.open(prePath + "/" + folder + '/' + file)
#         imRe = im.resize((320, 180))
#         #imRe.save(self.filePath + "/" + subClass + '/' + file)
#         images.append(np.asarray(imRe, np.float32))
#     xPre = []
#     xPre.append(np.asarray(images, np.float32))
#     yhat = model.predict(np.asarray(xPre, np.float64))
#     pre.append(yhat)
# m = []
# with open("pre.txt", "a") as f: 
#     for i in pre:
#         #print(i)
#         i = list(i[0])
#         ind = i.index(max(i))
        
#         if ind == 1:
#             ind = 2
#         print(ind)
#         m.append(ind)
    
# json_path = "data/amap_traffic_annotations_test.json"
# out_path = "data/amap_traffic_annotations_test_result.json"
    
# result = {}
# c = 0
# for folder in subFolder:
#     result[folder] = m[c]
#     c += 1
    
# # result 是你的结果, key是id, value是status
# with open(json_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as w:
#     json_dict = json.load(f)
#     data_arr = json_dict["annotations"]  
#     new_data_arr = [] 
#     for data in data_arr:
#         id_ = data["id"]
#         data["status"] = int(result[id_])
#         new_data_arr.append(data)
#     json_dict["annotations"] = new_data_arr
#     json.dump(json_dict, w)

code = "123"

pwd = input()

a = 3

while a > 0:
    if "*" in pwd:
        print("2")
        pwd = input()
    else:
        if pwd == code:
            print("1")
        else:
            a -= 1
            print("0")
            pwd = input()
            
 



















