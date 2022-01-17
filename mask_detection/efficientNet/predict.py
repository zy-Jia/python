import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import efficientnet_b0 as create_model


def main(img_name):
    # 选择训练设备，基于CUDA或基于CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # efn每个模型输入图片不同，配置如下，本次使用b0
    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"
    # 数据预处理，对数据进行224*224的resize并进行均衡缩放
    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 载入图片
    img_path = "../photo_to_pred/" + img_name
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # 读取类别json
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 载入模型
    model = create_model(num_classes=3).to(device)
    # 加载权重
    model_weight_path = "../model-18.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    plt.savefig('../pred_to_save/' + print_res + '_' + num_model + '_' + img_name)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    rootdir = '../photo_to_pred'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        if list[i] != '.DS_Store':
            print(list[i])
            main(list[i])
