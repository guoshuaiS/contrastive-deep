import numpy as np
def data_trans(z_all,sample_nums=60000,z_dims=10):#训练集样本数为60000
    batchs=len(z_all)
    for i in range(batchs):
        if i==0:
            out=z_all[i].reshape(-1,10)
        else:
            out=np.concatenate((out,z_all[i].reshape(-1,10)),axis=0)
    return out
def target_trans(label_true):
    batchs=len(label_true)#235
    for i in range(batchs):
        if i == 0:
            out = label_true[i].reshape(-1)
        else:
            out = np.concatenate((out, label_true[i].reshape(-1)), axis=0)
    return out