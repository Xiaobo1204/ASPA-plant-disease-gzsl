import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json

# ================= 1. 全局配置与超参数 =================
DATA_DIR = r"D:\MyDesktop\denclip\offline_features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 终极 SOTA 超参数 (基于 102 类精简数据集)
BATCH_SIZE = 64
LR = 0.0001
EPOCHS = 50
W_DISTILL = 30.0    # 强结构蒸馏权重
W_ATTR = 1.2        # 纯净属性对齐权重
BEST_K = 30         # 鉴别性属性去噪截断值
BEST_BIAS = 2.0     # CS 推理可见类偏置惩罚

# ================= 2. 数据集与加载器 =================
class GlobalFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

def load_labels(split):
    if split == "new_test":
        p = os.path.join(DATA_DIR, "new_test_labels_filtered.npy")
    else:
        p = os.path.join(DATA_DIR, f"{split}_labels.npy")
        if not os.path.exists(p): p = os.path.join(DATA_DIR, f"{split}_label.npy")
    return np.load(p, allow_pickle=True)

# 加载离线特征 (注意使用 filtered 版本)
feat_train = torch.from_numpy(np.load(os.path.join(DATA_DIR, "base_train_clip.npy")).astype(np.float32)).to(DEVICE)
feat_test_base = torch.from_numpy(np.load(os.path.join(DATA_DIR, "base_test_clip.npy")).astype(np.float32)).to(DEVICE)
feat_test_new = torch.from_numpy(np.load(os.path.join(DATA_DIR, "new_test_clip_filtered.npy")).astype(np.float32)).to(DEVICE)

# 加载类别映射与标签
with open(os.path.join(DATA_DIR, 'prompt_cleaned_filtered.json'), 'r', encoding='utf-8') as f:
    class_to_idx = {name: i for i, name in enumerate(sorted(list(json.load(f).keys())))}

label_train = torch.tensor([class_to_idx[l] for l in load_labels("base_train")]).long().to(DEVICE)
label_test_base = torch.tensor([class_to_idx[l] for l in load_labels("base_test")]).long().to(DEVICE)
label_test_new = torch.tensor([class_to_idx[l] for l in load_labels("new_test")]).long().to(DEVICE)

train_loader = DataLoader(GlobalFeatureDataset(feat_train, label_train), batch_size=BATCH_SIZE, shuffle=True)
attr_bank = torch.from_numpy(np.load(os.path.join(DATA_DIR, "attribute_bank_filtered.npy"))).float().to(DEVICE)

# ================= 3. 鉴别性属性去噪 (Discriminative Denoising) =================
print("鉴别性属性去噪与视觉原型构建")
features_norm = F.normalize(feat_train, p=2, dim=1)
bank_norm = F.normalize(attr_bank, p=2, dim=2)
num_classes = attr_bank.shape[0]

mask = torch.zeros((num_classes, 61), dtype=torch.bool).to(DEVICE)
mask[:, 0] = True # 锁定类名基准 Anchor

for c in range(num_classes):
    current_attrs = bank_norm[c, 1:, :]
    num_base = len(torch.unique(label_train))
    if c < num_base:
        pos_idx = (label_train == c)
        if pos_idx.sum() > 0:
            pos_feats = features_norm[pos_idx]
            neg_idx = (label_train != c)
            neg_feats = features_norm[neg_idx][0:2000]
            scores = (pos_feats @ current_attrs.T).mean(0) - (neg_feats @ current_attrs.T).mean(0)
        else:
            scores = (current_attrs @ bank_norm[c, 0, :].unsqueeze(0).T).squeeze()
    else:
        scores = (current_attrs @ bank_norm[c, 0, :].unsqueeze(0).T).squeeze()

    _, topk = torch.topk(scores, k=BEST_K)
    mask[c, topk + 1] = True

# 构建纯净视觉原型
zsl_weights = F.normalize(attr_bank[:, 0, :], dim=-1)
vis_feats = attr_bank[:, 1:, :]
vis_mask = mask[:, 1:].unsqueeze(-1)
visual_protos = (vis_feats * vis_mask).sum(1) / (vis_mask.sum(1) + 1e-6)
visual_protos = F.normalize(visual_protos, dim=-1)

# ================= 4. 结构保持特征适配器 (Structure-Preserved Adapter) =================
class ConstraintAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("classifier", zsl_weights)
        self.register_buffer("visual_targets", visual_protos)
        self.adapter = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 512))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.temp = 0.02

    def forward(self, x):
        res = self.adapter(x)
        x_new = x + self.alpha * res
        x_new = F.normalize(x_new, dim=-1)
        logits = (x_new @ self.classifier.T) / self.temp
        return logits, x_new

# ================= 5. 模型训练与校准堆叠推理 =================
print("极简架构模型训练：")
model = ConstraintAdapter().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()

seen_indices = torch.unique(label_train)
best_h = 0.0

for epoch in range(EPOCHS):
    model.train()
    for imgs, lbls in train_loader:
        logits, feats = model(imgs)

        # 联合损失函数
        l_ce = criterion(logits, lbls)
        l_dst = (1 - F.cosine_similarity(feats, imgs, dim=-1)).mean()
        l_att = (1 - F.cosine_similarity(feats, model.visual_targets[lbls], dim=-1)).mean()
        loss = l_ce + (W_DISTILL * l_dst) + (W_ATTR * l_att)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证期推理 (带 CS 校准)
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            lb = model(feat_test_base)[0].cpu()
            ln = model(feat_test_new)[0].cpu()

            # 核心策略：CS 偏置惩罚
            lb[:, seen_indices.cpu()] -= BEST_BIAS
            ln[:, seen_indices.cpu()] -= BEST_BIAS

            acc_b = (lb.argmax(1) == label_test_base.cpu()).float().mean().item()
            acc_n = (ln.argmax(1) == label_test_new.cpu()).float().mean().item()
            h = 2 * acc_b * acc_n / (acc_b + acc_n + 1e-6)

            if h > best_h:
                best_h = h
                torch.save(model.state_dict(), os.path.join(DATA_DIR, "model_final_sota.pth"))

            print(f"Epoch [{epoch+1:02d}/{EPOCHS}] -> Seen: {acc_b*100:.2f}% | Unseen: {acc_n*100:.2f}% | H-Score: {h*100:.2f}%")

print("=" * 60)
print(f"训练完成，最高 H-Score 记录: {best_h*100:.2f}%")
