import torch
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import scipy.io as sio
from time import time
from lib import ops
from torchvision import datasets, transforms
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from lib import metrics
from lib.datasets import MNIST
from lib import data_resize
import numpy as np
from lib import contrastive_loss
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='unables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

num_epochs = 100
batch_size = 512
z_dim = 50
n_cluster = 10
acc_log=[]#用于记录准确率的迭代变化
nmi_log=[]#用于记录nmi的迭代变化
# NUS dataset
train_loader = torch.utils.data.DataLoader(MNIST('./dataset/mnist', train=True, download=True),batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(MNIST('./dataset/mnist', train=False, download=True),batch_size=batch_size, shuffle=False, num_workers=0)
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class Autoencoder(nn.Module):
    def __init__(self, in_dim=100, h_dim=80):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(h_dim,z_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
            )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(h_dim,in_dim),
            #nn.Sigmoid()
            )

    def forward(self, x):
        """
        Note: image dimension conversion will be handled by external methods
        """
        h= self.encoder(x)
        out=self.decoder(h)

        return h,out
        #得到重建图像out和预测类别pre_label

class Autoencoder1(nn.Module):
    def __init__(self, in_dim=100, h_dim=6):
        super(Autoencoder1, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            #nn.Dropout(p=0.2)
            )
        self.decoder = nn.Sequential(
            nn.Linear(h_dim, in_dim),
            nn.Sigmoid()
            )

    def forward(self, x):
        """
        Note: image dimension conversion will be handled by external methods
        """
        h= self.encoder(x)
        out=self.decoder(h)

        return h,out
        #得到重建图像out和预测类别pre_label

ae = Autoencoder(in_dim=784, h_dim=256)

if torch.cuda.is_available():
    ae.cuda()

#定义损失函数
instance_temperature = 0.5  # 这个可以调，有的文章说0.1好，有的设置0.5
if torch.cuda.is_available():
    loss_device = torch.device("cuda")
else:
    loss_device = torch.device("cpu")
criterion_instance = contrastive_loss.InstanceLoss(instance_temperature, loss_device).to(
    loss_device)
def m_loss(input,target,alpha=0.1):
    loss1=torch.mean((input-target)**2)
    loss2=criterion_instance(input,target)
    return loss1+alpha*loss2

optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)

def train(epoch):
    ae.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = to_var(data.view(data.size(0), -1))
        target=Variable(target)
        #target=to_var(target.view(target.size(0),-1))
        optimizer.zero_grad()
        code,out= ae(data)
        loss = m_loss(data,out,alpha=0.05)
        loss.backward()
        train_loss += loss.item()*len(data)
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    ae.eval()
    global acc_log
    global nmi_log
    z_all = []
    label_true = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = to_var(data.view(data.size(0), -1))
        target = Variable(target)
        z, outputs = ae.forward(data)
        z1 = np.array(z.data.view(-1, z_dim).tolist())  # nus:31;
        z_all.append(z1)
        label_true.append(target.numpy())
    z_all = np.array(z_all)
    z_all = data_resize.data_trans(z_all)
    label_true = np.array(label_true)
    label_true = data_resize.target_trans(label_true)
    ##kmeans result
    estimator = KMeans(n_clusters=10)  # 聚类数量
    estimator.fit(z_all)
    centroids = estimator.cluster_centers_
    label_pred = estimator.labels_
    acc = metrics.acc(label_true, label_pred)
    nmi = metrics.nmi(label_true, label_pred)
    print("acc:", acc)
    print("nmi", nmi)
    if acc>0.63:
        path = './Mat/our_ae_subspace' + str(epoch)
        sio.savemat(path+'.mat', {'feature':z_all,'true':label_true,'pred':label_pred})
    acc_log.append(acc)
    nmi_log.append(nmi)

for epoch in range(1, num_epochs + 1):
    train(epoch)
    test(epoch)

print("max acc:",max(acc_log))
print("max acc in :",acc_log.index(max(acc_log)))
print("max nmi:",max(nmi_log))
x = torch.linspace(1, len(acc_log), steps=len(acc_log))
x = x.numpy()
y_acc = torch.FloatTensor(acc_log).numpy()
y_nmi = torch.FloatTensor(nmi_log).numpy()
#sio.savemat('./Mat/our_ae_acc.mat', {'ACC':y_acc})
#sio.savemat('./Mat/our_ae_nmi.mat', {'NMI':y_nmi})

