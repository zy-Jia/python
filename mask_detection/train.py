import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from alexnet.model import AlexNet
from vgg.model import vgg
from NN_base.NN import NN as NN_base


def main(model_name):
    # 选择训练设备，基于CUDA或基于CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    
    # 数据预处理，对数据进行224*224的随机裁剪和随机水平翻转并进行归一化，测试集则不需要进行随机裁剪和随机水平翻转
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    # 数据根目录
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    # 图片所在根目录
    image_path = os.path.join(data_root, "mask_detection", "mask_data")  # mask data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 数据加载器加载训练数据
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    # 获取分类标签
    # {'mask_weared_incorrect':0, 'with_mask':1, 'without_mask':2}
    mask_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in mask_list.items())
    # 将分类标签写入json文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    # 配置训练送入网络中训练的数据量为32
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    
    # 读取训练数据并进行随机打乱
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    
    # 加载测试数据
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    # 读取测试数据
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    epochs = 1
    # 加载alexnet模型，初始化权重，设置分类类别数为3
    if model_name == 'AlexNet':
        net = AlexNet(num_classes=3, init_weights=True)
        # 配置训练环境为cuda或cpu
        net.to(device)
        # 损失函数使用CrossEntropyLoss
        loss_function = nn.CrossEntropyLoss()
        # pata = list(net.parameters())
        # 优化器使用adam
        optimizer = optim.Adam(net.parameters(), lr=0.0002)
        save_path = './AlexNet.pth'
    # 加载vgg16模型
    elif model_name == 'vgg16':
        net = vgg(model_name=model_name, num_classes=3, init_weights=True)
        net.to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        save_path = './{}Net.pth'.format(model_name)
    # 加载基本模型
    # elif model_name == 'NN_base':
    else:
        net = NN_base()
        net.to(device)
        loss_function = nn.CrossEntropyLoss()
        # pata = list(net.parameters())
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        
        save_path = './NN_base.pth'
    # 记录最高准确率
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            # 反向传播
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # 验证集进行验证
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        # 得出该epoch的准确率和损失
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    # for model_name in ['alexnet', 'vgg16', 'NN_base']:
        # main(model_name)
    main('alexnet')
