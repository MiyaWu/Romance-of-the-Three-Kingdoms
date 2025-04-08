import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCE(nn.Module):
    def __init__(self, temperature):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.similarity_f = nn.CosineSimilarity(dim=1)

    def forward(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=0)
        N = z_i.size(0)  # 获取 batch_size

        # 计算相似度矩阵
        sim = torch.matmul(z, z.T) / self.temperature
        # print(f"Shape of similarity matrix: {sim.shape}")  # 应该是 [2*N, 2*N]

        # 提取正样本的相似度
        pos_sim = torch.diag(sim, N)  # z_i 与 z_j 的正样本
        pos_sim_j = torch.diag(sim, -N)  # z_j 与 z_i 的正样本
        pos_sim = torch.cat([pos_sim, pos_sim_j], dim=0).reshape(-1, 1)  # 合并正样本相似度

        # 计算负样本相似度
        neg_mask = ~torch.eye(2 * N, dtype=bool, device=sim.device)  # 生成 2N x 2N 的掩码矩阵，排除主对角线位置
        neg_sim = sim[neg_mask].view(2 * N, -1)  # 将负样本相似度重塑为 [2N, 2N-1] 形状

        # 组合 logits
        logits = torch.cat((pos_sim, neg_sim), dim=1)  # 这里修改为dim=1

        # 生成标签
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # 计算损失
        loss = F.cross_entropy(logits, labels)

        return loss
