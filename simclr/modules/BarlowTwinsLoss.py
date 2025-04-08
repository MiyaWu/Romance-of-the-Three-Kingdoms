import torch
import torch.nn as nn

def off_diagonal(x):
    # 返回方阵的非对角线元素的扁平视图
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLoss(nn.Module):
    def __init__(self, device, lambd=5e-3, feature_dim=64):
        super(BarlowTwinsLoss, self).__init__()
        self.lambd = lambd
        self.device = device
        self.bn = nn.BatchNorm1d(feature_dim, affine=False).to(self.device)

    def forward(self, z_a, z_b):
        # 归一化
        z_a = self.bn(z_a)
        z_b = self.bn(z_b)

        # 计算交叉协方差矩阵
        c = torch.mm(z_a.T, z_b)  # 计算交叉协方差矩阵

        # 对交叉协方差矩阵进行归一化
        c.div_(z_a.size(0))  # 使用当前批量大小进行归一化

        # 计算损失
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()  # 对角线元素减1
        off_diag = off_diagonal(c).pow_(2).sum()  # 非对角线元素
        loss = on_diag + self.lambd * off_diag  # 总损失
        return loss