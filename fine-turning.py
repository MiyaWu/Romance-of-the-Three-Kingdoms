import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cosine

# 加载预训练的SimCLR模型
class PretrainedSimCLR(nn.Module):
    def __init__(self, checkpoint_path, device):
        super().__init__()
        # 初始化编码器
        self.encoder = models.resnet50(pretrained=False)
        self.encoder.fc = nn.Identity()  # 移除原始全连接层

        # 加载SimCLR的 state dict 并包括投影头
        state_dict = torch.load(checkpoint_path, map_location=device)

        # 将state dict加载到模型中，包含编码器和投影头
        self.load_state_dict(state_dict, strict=False)
        self.to(device)

    def forward(self, x):
        return self.encoder(x)


# 特征提取函数
def extract_features(model, image_dir, transform, device):
    model.eval()
    features = {}
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)  # 添加批次维度
                with torch.no_grad():
                    feature = model(img_tensor).squeeze(0).cpu().numpy()
                features[file] = feature
    return features


# 数据集类，用于相似度检测
class SimilarityDataset(Dataset):
    def __init__(self, image_dir, score_lines, transform=None, features=None):
        self.image_dir = image_dir  # 所有图片所在目录
        self.transform = transform
        self.features = features
        self.image_pairs = self.load_image_pairs(score_lines)

    def load_image_pairs(self, score_lines):
        image_pairs = []
        for line in score_lines:
            parts = line.strip().split(', ')
            img1_name = parts[0]
            base_img2 = parts[1]
            score = float(parts[2])

            # 生成变体候选列表
            base_stem = os.path.splitext(base_img2)[0]  # 获取基准文件名主干
            candidates = []

            # 遍历目录查找变体
            for fname in os.listdir(self.image_dir):
                f_stem = os.path.splitext(fname)[0]
                # 严格匹配基准主干开头的文件
                if f_stem.startswith(base_stem) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    candidates.append(fname)

            if not candidates:
                candidates = [base_img2]

            image_pairs.append((img1_name, candidates, score))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_name, img2_candidates, score = self.image_pairs[idx]

        # 获取img1特征
        feat1 = self.features.get(img1_name, None)
        if feat1 is None:
            raise KeyError(f"特征缺失: {img1_name}")

        best_similarity = -float('inf')
        best_img2 = img2_candidates[0]  # 默认使用第一个候选

        for candidate in img2_candidates:
            feat2 = self.features.get(candidate, None)
            if feat2 is None:
                continue

            similarity = 1 - cosine(feat1, feat2)
            if similarity > best_similarity:
                best_similarity = similarity
                best_img2 = candidate

        # 加载图像（统一目录）
        img1_path = os.path.join(self.image_dir, img1_name)
        img2_path = os.path.join(self.image_dir, best_img2)

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(score, dtype=torch.float32)


# 微调模型
def fine_tune(model, train_loader, val_loader,test_loader, device, num_epochs, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_losses = []
    val_losses = []
    test_mses = []  # 测试指标存储
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for img1, img2, scores in train_loader:
            img1, img2, scores = img1.to(device), img2.to(device), scores.to(device)

            optimizer.zero_grad()

            feat1 = model(img1)
            feat2 = model(img2)

            similarities = torch.nn.functional.cosine_similarity(feat1, feat2, dim=1)
            loss = criterion(similarities, scores)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * img1.size(0)

        # 计算训练损失
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for img1, img2, scores in val_loader:
                img1, img2, scores = img1.to(device), img2.to(device), scores.to(device)

                feat1 = model(img1)
                feat2 = model(img2)

                similarities = torch.nn.functional.cosine_similarity(feat1, feat2, dim=1)
                loss = criterion(similarities, scores)

                epoch_val_loss += loss.item() * img1.size(0)

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        test_mse = evaluate(model, test_loader, device)
        test_mses.append(test_mse)  # 记录每个epoch的测试结果
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig('learning_curve.png')
    plt.show()


    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), test_mses, 'g-',
             linewidth=2.5,
             alpha=0.8,
             label='Test MSE')


    plt.ylim(0.08, 0.5)
    plt.grid(True, alpha=0.3)

    plt.title('Test Set Performance', fontsize=12, pad=15)
    plt.xlabel('Epochs', fontsize=10, labelpad=8)
    plt.ylabel('MSE', fontsize=10, labelpad=8)
    plt.legend(fontsize=9)

    plt.savefig('mse.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='#f8f9fa')

    return train_losses, val_losses


# evaluate
def evaluate(model, test_loader, device):
    model.eval()
    predictions = []
    truths = []
    with torch.no_grad():
        for img1, img2, scores in test_loader:
            img1, img2 = img1.to(device), img2.to(device)

            feat1 = model(img1)
            feat2 = model(img2)

            similarities = torch.nn.functional.cosine_similarity(feat1, feat2, dim=1)
            predictions.extend(similarities.cpu().numpy())
            truths.extend(scores.cpu().numpy())

    mse = mean_squared_error(truths, predictions)
    print(f"Test MSE: {mse:.4f}")
    return mse


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'D:\\google\\SimCLR-master\\SimCLR-master\\save\\checkpoint_100.tar'
    image_dir = 'D:\\google\\SimCLR-master\\SimCLR-master\\datasets\\壁画(微调)\\图'  # 图片目录路径
    score_file = 'D:\\google\\SimCLR-master\\SimCLR-master\\datasets\\壁画(微调)\\output.txt'  # 评分文件路径

    eopchs = 100

    with open(score_file, 'r',encoding="utf-8") as f:
        all_lines = f.readlines()
    random.shuffle(all_lines)

    # 划分数据集
    train_lines, test_lines = train_test_split(all_lines, test_size=0.2, random_state=42)
    train_lines, val_lines = train_test_split(train_lines, test_size=0.125, random_state=42)

    # 初始化模型
    model = PretrainedSimCLR(checkpoint_path, device)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 预提取特征
    features = extract_features(model, image_dir, transform, device)

    # 创建数据集
    train_dataset = SimilarityDataset(image_dir, train_lines, transform, features)
    val_dataset = SimilarityDataset(image_dir, val_lines, transform, features)
    test_dataset = SimilarityDataset(image_dir, test_lines, transform, features)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 微调模型
    train_loss, val_loss = fine_tune(model, train_loader, val_loader,test_loader, device,num_epochs=eopchs)

    # 评估模型
    mse = evaluate(model, test_loader, device)



if __name__ == "__main__":
    main()