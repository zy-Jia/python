import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from alexnet.model import AlexNet
from vgg.model import vgg
from NN_base.NN import NN as NN_base


def main(model_name, img_name):
    # 选择训练设备，基于CUDA或基于CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 数据预处理，设置宽高并归一化
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 加载图片
    img_path = "photo_to_pred/" + img_name
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')

    plt.imshow(img)
    # 通道转换
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # 读取类别json
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    
    # 导入模型
    if  model_name == 'AlexNet':
        model = AlexNet(num_classes=3).to(device)
        # load model weights
        weights_path = "./AlexNet.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    elif model_name == 'vgg16':
        model = vgg(model_name="vgg16", num_classes=3).to(device)
        # load model weights
        weights_path = "./vgg16Net.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    elif model_name == 'NN_base':
        model = NN_base()
        # load model weights
        weights_path = "./NN_base.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)

    # 导入权重
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    plt.savefig('pred_to_save/' + print_res + '_' + model_name + '_' + img_name)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    rootdir = 'photo_to_pred'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for model_name in ['AlexNet', 'vgg16', 'NN_base']:
        for i in range(0, len(list)):
            if list[i] != '.DS_Store':
                main(model_name, list[i])
        