import os
import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision import transforms
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device)) #进行计算设备的检测和选择



transfm = Compose([transforms.ToTensor()])  # 数据增广，这里只是一个转换
labeldict = {'圆形': 0, '椭圆形': 1, '正方形': 2}
class KZDataset(Dataset):
    def __init__(self, txt_path=None, ki=0, K=10, typ='train', transform=None, rand=False):
        '''
        txt_path: 所有数据的路径，我的形式为(单张图片路径,类别\n)
        	img1.png,0
        	...
     	    img100.png,1
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
        img_path, label = self.data_info[index]
        img = cv2.imread(img_path)
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
        return all_data_info

txtpath = 'data.txt'
train_data = KZDataset(txt_path=txtpath, ki=0, K=10, typ='train', transform=transfm, rand=True)
val_data = KZDataset(txt_path=txtpath, ki=0, K=10, typ='val', transform=transfm, rand=True)
#dataloader装载
train_loader = DataLoader(train_data,batch_size=4,shuffle=False,num_workers=0)
val_loader = DataLoader(val_data,batch_size=4,shuffle=False,num_workers=0)


class AlexNet(nn.Module): #由于相较于Alexnet只有三个输出，因而直接搭
    def __init__(self,num_classes=3): #这里的类别数
        super(AlexNet,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,96,kernel_size=11,stride=4,padding=2),   #(224+2*2-11)/4+1=55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),   #(55-3)/2+1=27
            nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2), #(27+2*2-5)/1+1=27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),   #(27-3)/2+1=13
            nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1),    #(13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1),    #(13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),    #13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )   #6*6*256=9126

        self.avgpool=nn.AdaptiveAvgPool2d((6,6))
        self.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_classes),
        )

    def forward(self,x): #定义前向传播方法
        x=self.features(x)
        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x

model=AlexNet().to(device) #模型装载到设备上

loss_fn = nn.CrossEntropyLoss() #损失函数用交叉熵
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) #优化函数用随机梯度下降

#训练
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
            # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

            # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(batch)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#测试
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#设置epoch和定义流程
epochs = 500
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(val_loader, model, loss_fn)
print("Done!")
