#input:tensor matrix(two dims) ,sigma
#matrix(n*d):其中n是样本个数，d是特征维数
#output:out tensor
import torch
def norm_sigma(input,sigma):
    n=input.size(0)
    d=input.size(1)
    out=0.0
    for i in range(d):
        x_i=input[:,i:i+1]
        #print(x_i)
        x_i_norm=torch.norm(x_i)
        y=((1+sigma)*x_i_norm**2)/(sigma+x_i_norm)
        out+=y
    return out/n
'''input=torch.rand((10000,784))
out=norm_sigma(input,sigma=0)
out1=norm_sigma(input,sigma=0.1)
out2=norm_sigma(input,sigma=0.2)
print(out)
print(out1)
print(out2)'''
#out=norm_sigma(input,0)
#print(out)