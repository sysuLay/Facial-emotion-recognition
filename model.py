import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch import nn
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


data = pd.read_csv('train.csv')
target = [int(data.loc[i,'label']) for i in range(20000)]
origin = [list(map(int, data.loc[i, 'feature'].split())) for i in range(20000)]

Xtrain, Xval, Ytrain, Yval = train_test_split(origin, target, test_size=1e-3)
Xtrain, Xval, Ytrain, Yval = torch.Tensor(Xtrain), torch.Tensor(Xval), torch.Tensor(Ytrain), torch.Tensor(Yval)



class DataStore(Data.Dataset):
    def __init__(self, X, Y, training=True):
        self.X = X
        self.Y = Y
        self.Len=len(self.X)
    #预处理，随机翻转，归一化    
    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        x = self.X[index].reshape(1, 48, 48) / 255.0
        y = self.Y[index].long()
        return transform(x), y
    
    def __len__(self):
        return self.Len
    
dataset = DataStore(Xtrain, Ytrain)
testdataset = DataStore(Xval, Yval)
batch_size = 120
kwargs = {'num_workers': 4, 'pin_memory': True}
train_loader = DataLoader(dataset=dataset, shuffle=True,
                          batch_size=batch_size, **kwargs)
test_loader = DataLoader(dataset=testdataset, shuffle=True,
                          batch_size=1, **kwargs)
#模型初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

#注意力机制
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #对应Squeeze操作
        y = self.fc(y).view(b, c, 1, 1) #对应Excitation操作
        return x * y.expand_as(x)

#SE的VGG
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),  
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            SELayer(32),
            nn.MaxPool2d(2, 2, 0),      # [64, 12, 12]
            
            nn.Conv2d(32, 64, 3, 1, 1),  # [64, 24, 24]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            SELayer(64),
            nn.MaxPool2d(2, 2, 0),      # [64, 12, 12]
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            SELayer(128),
            nn.MaxPool2d(2, 2, 0),      # [128, 6, 6]

            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            SELayer(256),
            nn.MaxPool2d(2, 2, 0)       # [256, 3, 3]
        )

        self.fc = nn.Sequential(
            nn.Linear(256*3*3, 4096),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.4),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.4),
            nn.Linear(4096, 7)
        )

        self.cnn.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)




model = Classifier()
model = model.cuda()#GPU
from torch.optim import lr_scheduler
#交叉熵损失函数
def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if (batch_idx+1) % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))


def test(epoch):
    model.eval()
    
    idx = 0
    tot = 0
    for (data, target) in test_loader:
        data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            output = model(data).argmax(dim=1)
            tot += torch.sum(torch.eq(output, target))

        idx += 1
        
    tot = tot.float()
    print('Accuracy: {:.4f}'.format(tot / idx))

    
    torch.save(model, "model-cnn.pth")

best_acc = 0.0
criterion = nn.CrossEntropyLoss()#交叉熵
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=0.00001)#优化器
lr_scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=100, step_size_down=100, cycle_momentum=False)

for epoch in range(10000):
    train(epoch)
    test(epoch)
    
print('Best Accuracy: {:.4f}'.format(best_acc))

model = torch.load("model-cnn.pth")
    
testData = pd.read_csv('test.csv')

Xtest = [list(map(int, testData.loc[i, 'feature'].split())) for i in range(8500)]
Ytest = np.zeros((8500,))
#测试集统一格式
for i in range(8500):
    x = torch.Tensor(Xtest[i]).reshape(1, 48, 48) / 255.0
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    y = model(transform(x).unsqueeze(0).cuda()).argmax()
    Ytest[i] = y
testData.rename(columns={'feature':'label'},inplace=True)
testData['label'] = Ytest.astype(np.int)
testData.to_csv('submission.csv', index=False, header=True)
