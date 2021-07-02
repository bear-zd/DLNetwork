import cv2
import time
import torch
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import math
import random
import torch.nn as nn

SPPDict1 = r'xxx.pkl'
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
epochs = 70

class SPPLayer(torch.nn.Module): #SPP层，会受到bin_num的限制，可以使用另一个SPPLayer
    def __init__(self, num_levels= 2, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size() # num:样本数量 c:通道数 h:高 w:宽
        #print(x.size())
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))
            #pooling2 = (math.floor(kernel_size[0]/2),math.floor(kernel_size[1]/2))
            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = torch.nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = torch.nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten


# 该SPPLayer层先进行pading再池化，因而不会受到bin的限制。
class PFSPPLayer(torch.nn.Module):
    def __init__(self, num_levels=6, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size()  # num:样本数量 c:通道数 h:高 w:宽
        # print(x.size())
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (
            math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

            h_new = 2 * pooling[0] + h
            w_new = 2 * pooling[1] + w
            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))
            zero_pad = torch.nn.ZeroPad2d((pooling[1], pooling[1], pooling[0], pooling[0]))
            x_new = zero_pad(x)
            # pooling2 = (math.floor(kernel_size[0]/2),math.floor(kernel_size[1]/2))
            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = torch.nn.functional.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
            else:
                tensor = torch.nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=stride,
                                                        padding=pooling).view(num, -1)
            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten


class SPPNet(nn.Module):  # 由于相较于Alexnet只有三个输出，因而直接搭,相较于AlexNet删除了两个全连接层，并将池化层改为SPP层
    def __init__(self, num_classes=3):  # 这里的类别数
        super(SPPNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # (224+2*2-11)/4+1=55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (55-3)/2+1=27
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),  # (27+2*2-5)/1+1=27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (27-3)/2+1=13
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # (13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # (13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # 13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
        )  # 6*6*256=9126
        # 这里改为SPP层
        self.avgpool = SPPLayer(2)
        # nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(1280, num_classes), #这个全连接层的输入特征数由SPP层决定，计算公式 (1^2+...+bin^2)*卷积层输出特征数,如该1280=(1^2+2^2)*256
        )
    def forward(self, x):  # 定义前向传播方法
        x = self.features(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Model():
    def __init__(self):
        self.model = SPPNet(2)
    def loadmodelDict(self,Dict):
        if self.Device() == "cuda":
            self.model.load_state_dict(torch.load(Dict)) #由于自己写了SPP层，所以保存整个模型是没有用的。
        else:
            self.model.load_state_dict(torch.load(Dict, map_location=torch.device('cpu')))
    def Device(self): #检测设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return device
    def predict(self,ImgPath):
        Tensor = self.__img2tensor(ImgPath)
        pred = self.model(Tensor)
        result = self.__arg2label(pred.argmax())
        print(result)
        return result
    def __img2tensor(self,ImgPath):
        Img = cv2.imread(ImgPath)
        Transfm = Compose([transforms.ToTensor()])
        Img = Transfm(Img)
        Img = Img.unsqueeze(0)
        return Img
    def __arg2label(self,arg):
        return LabelDict[str(arg.item())]


class KZDataset(Dataset):#K折交叉检验
    def __init__(self, txt_path=None, ki=0, K=10, typ='train', transform=None, rand=False):
        '''
        txt_path: 所有数据的路径，我的形式为(单张图片路径 类别\n)
        	img1.png 0
        	...
     	    img100.png 1
     	ki：当前是第几折,从0开始，范围为[0, K)
     	K：总的折数
     	typ：用于区分训练集与验证集
     	transform：对图片的数据增强
     	rand：是否随机
        '''
        self.all_data_info = self.get_img_info(txt_path)
        if rand:
            random.seed(1)
            random.shuffle(self.all_data_info)
        leng = len(self.all_data_info)
        every_z_len = leng // K
        if typ == 'val':
            self.data_info = self.all_data_info[every_z_len * ki: every_z_len * (ki + 1)]
        elif typ == 'train':
            self.data_info = self.all_data_info[: every_z_len * ki] + self.all_data_info[every_z_len * (ki + 1):]
        self.transform = transform
    def __getitem__(self, index):
        # Dataset读取图片的函数
        img_pth, label = self.data_info[index]
        img = cv2.imread(img_pth)
        if self.transform is not None:
            img = self.transform(img)
        return img, labeldict[label.replace('\n', '')]
    def __len__(self):
        return len(self.data_info)
    def get_img_info(self, txtpath):
        txt = open(txtpath, 'r', encoding='utf-8')
        txt = txt.readlines()
        all_data_info = []
        for i in txt:
            all_data_info.append(list(i.split(',')))
        # print(all_data_info)
        return all_data_info

def my_collate(batch): #由于SPP的特殊性，需要重写loader的collate_fn
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    label = torch.Tensor(target)
    return data[0].unsqueeze(0), label

def Dateloader(filepath,ki,K,transfm,rand,shuffle):
    train_data = KZDataset(txt_path=filepath, ki=Ki, K=K, typ='train', transform=transfm, rand=rand)
    val_data = KZDataset(txt_path=filepath, ki=Ki, K=K, typ='val', transform=transfm, rand=rand)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=shuffle, num_workers=0, collate_fn=my_collate)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=shuffle, num_workers=0, collate_fn=my_collate)
    return train_loader,val_loader


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def Train(epochs):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer) #这里的model是模型，而非封装的类。
        test(val_loader, model, loss_fn)
    print("Done!")




