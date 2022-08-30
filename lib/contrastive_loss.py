import torch
import torch.nn as nn


class InstanceLoss(nn.Module):
    def __init__(self, temperature, device):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature
        self.device = device

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * z_i.size(0)#
        z = torch.cat((z_i, z_j), dim=0)
        self.batch_size = z_i.size(0)
        self.mask = self.mask_correlated_samples(self.batch_size)
        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss = loss/N/784   #平均每个样本,平均每个像素
        return loss