from sklearn.cluster import KMeans
import scipy.io as sio
import numpy as np
from lib import metrics

data_0 = sio.loadmat('./lib/data/mnist01.mat')
data_dict = dict(data_0)
target = data_dict['label']
data = data_dict['X2']
n_cluster=10
acc_all=[]

kmeans = KMeans(n_clusters=n_cluster).fit(data)  #
y_pre = kmeans.labels_
acc = metrics.acc(target.reshape(-1), y_pre)
nmi = metrics.nmi(target.reshape(-1), y_pre)
print('ACC', acc)
print('NMI', nmi)
