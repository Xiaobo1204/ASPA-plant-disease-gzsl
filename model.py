from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Config:
    data_root: str = "data/plantwild_v2"
    prompt_path: str = "data/prompts/prompt_cleaned_filtered.json"
    backbone: str = "ViT-B-32"
    pretrained: str = "openai"
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-3
    num_workers: int = 4
    best_k: int = 30
    lambda_attr: float = 1.2
    lambda_dist: float = 30.0
    seen_bias: float = 2.0
    eval_interval: int = 5
    max_neg_samples: int = 2000
    alpha_init: float = 0.2
    temperature: float = 0.02
    seed: int = 42
    output_dir: str = "outputs/aspa"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PlantDiseaseDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        class_to_idx: Dict[str, int],
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples: List[Tuple[Path, int, str]] = []

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.root}")

        for class_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                continue
            label = self.class_to_idx[class_name]
            for img_path in sorted(class_dir.rglob("*")):
                if img_path.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((img_path, label, class_name))

        if not self.samples:
            raise RuntimeError(f"No images found under: {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label, class_name = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, class_name, str(img_path)


class ASPA(nn.Module):
    def __init__(
        self,
        class_name_anchors: torch.Tensor,
        purified_prototypes: torch.Tensor,
        alpha_init: float = 0.2,
        temperature: float = 0.02,
    ) -> None:
        super().__init__()
        self.register_buffer("classifier", F.normalize(class_name_anchors, dim=-1))
        self.register_buffer("purified_prototypes", F.normalize(purified_prototypes, dim=-1))

        dim = class_name_anchors.shape[-1]
        hidden_dim = max(dim // 4, 128)

        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.temperature = temperature

    def forward(self, image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = self.adapter(image_features)
        adapted = F.normalize(image_features + self.alpha * residual, dim=-1)
        logits = adapted @ self.classifier.t() / self.temperature
        return logits, adapted


@torch.no_grad()
def encode_texts(
    clip_model,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start:start + batch_size])
        tokens = tokenizer(batch_texts).to(device)
        text_features = clip_model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        outputs.append(text_features)
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def collect_image_features(
    clip_model,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_features = []
    all_labels = []

    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        image_features = clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        all_features.append(image_features)
        all_labels.append(labels)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def load_prompt_map(
    prompt_path: str | Path,
    class_names: Sequence[str],
) -> Dict[str, List[str]]:
    with open(prompt_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    prompt_map: Dict[str, List[str]] = {}
    for class_name in class_names:
        if class_name not in raw:
            raise KeyError(f"Missing prompts for class: {class_name}")
        value = raw[class_name]
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or len(value) == 0:
            raise ValueError(f"Invalid prompts for class: {class_name}")
        prompt_map[class_name] = value

    return prompt_map


@torch.no_grad()
def build_text_bank(
    clip_model,
    tokenizer,
    class_names: Sequence[str],
    prompt_map: Dict[str, List[str]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    class_name_anchors = []
    raw_prompt_bank = []

    for class_name in class_names:
        anchor_text = class_name.replace("_", " ")
        anchor_feature = encode_texts(clip_model, tokenizer, [anchor_text], device)[0]
        prompt_features = encode_texts(clip_model, tokenizer, prompt_map[class_name], device)

        class_name_anchors.append(anchor_feature)
        raw_prompt_bank.append(prompt_features)

    class_name_anchors = torch.stack(class_name_anchors, dim=0)
    raw_prompt_bank = torch.stack(raw_prompt_bank, dim=0)
    return class_name_anchors, raw_prompt_bank


@torch.no_grad()
def build_dap_prototypes(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    raw_prompt_bank: torch.Tensor,
    class_name_anchors: torch.Tensor,
    seen_class_indices: Sequence[int],
    best_k: int,
    max_neg_samples: int,
) -> torch.Tensor:
    num_classes, num_prompts, dim = raw_prompt_bank.shape
    prototypes = torch.zeros((num_classes, dim), device=raw_prompt_bank.device)
    seen_set = set(seen_class_indices)

    for c in range(num_classes):
        prompt_feats = F.normalize(raw_prompt_bank[c], dim=-1)

        if c in seen_set:
            pos_mask = train_labels == c
            neg_mask = train_labels != c

            pos_feats = train_features[pos_mask]
            neg_feats = train_features[neg_mask]

            if pos_feats.numel() == 0:
                scores = prompt_feats @ class_name_anchors[c]
            else:
                if neg_feats.size(0) > max_neg_samples:
                    neg_feats = neg_feats[:max_neg_samples]
                pos_score = pos_feats @ prompt_feats.t()
                neg_score = neg_feats @ prompt_feats.t()
                scores = pos_score.mean(dim=0) - neg_score.mean(dim=0)
        else:
            scores = prompt_feats @ class_name_anchors[c]

        k = min(best_k, prompt_feats.size(0))
        keep_idx = torch.topk(scores, k=k, dim=0).indices
        prototype = prompt_feats[keep_idx].mean(dim=0)
        prototypes[c] = F.normalize(prototype, dim=-1)

    return prototypes


@torch.no_grad()
def evaluate(
    model: ASPA,
    clip_model,
    loader_seen: DataLoader,
    loader_unseen: DataLoader,
    seen_indices: torch.Tensor,
    seen_bias: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    def run_one_loader(loader: DataLoader) -> float:
        correct = 0
        total = 0

        for images, labels, _, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            image_features = clip_model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            logits, _ = model(image_features)
            logits[:, seen_indices] -= seen_bias

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

        return correct / max(total, 1)

    model.eval()
    clip_model.eval()

    acc_seen = run_one_loader(loader_seen)
    acc_unseen = run_one_loader(loader_unseen)
    h_score = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen + 1e-12)
    return acc_seen, acc_unseen, h_score


def make_global_class_map(data_root: str | Path) -> Tuple[List[str], List[str], List[str], Dict[str, int]]:
    data_root = Path(data_root)
    train_root = data_root / "base_train"
    seen_test_root = data_root / "base_test"
    unseen_test_root = data_root / "new_test"

    train_classes = sorted([p.name for p in train_root.iterdir() if p.is_dir()])
    seen_test_classes = sorted([p.name for p in seen_test_root.iterdir() if p.is_dir()])
    unseen_test_classes = sorted([p.name for p in unseen_test_root.iterdir() if p.is_dir()])

    all_classes = sorted(set(train_classes) | set(seen_test_classes) | set(unseen_test_classes))
    class_to_idx = {name: idx for idx, name in enumerate(all_classes)}
    return train_classes, seen_test_classes, unseen_test_classes, class_to_idx


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.backbone,
        pretrained=cfg.pretrained,
    )
    tokenizer = open_clip.get_tokenizer(cfg.backbone)
    clip_model = clip_model.to(device)
    clip_model.eval()

    for param in clip_model.parameters():
        param.requires_grad = False

    train_classes, _, unseen_test_classes, class_to_idx = make_global_class_map(cfg.data_root)
    all_class_names = sorted(class_to_idx.keys())

    train_set = PlantDiseaseDataset(Path(cfg.data_root) / "base_train", class_to_idx, preprocess)
    seen_test_set = PlantDiseaseDataset(Path(cfg.data_root) / "base_test", class_to_idx, preprocess)
    unseen_test_set = PlantDiseaseDataset(Path(cfg.data_root) / "new_test", class_to_idx, preprocess)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    train_loader_no_shuffle = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    seen_test_loader = DataLoader(
        seen_test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    unseen_test_loader = DataLoader(
        unseen_test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    prompt_map = load_prompt_map(cfg.prompt_path, all_class_names)

    with torch.no_grad():
        train_features, train_labels = collect_image_features(clip_model, train_loader_no_shuffle, device)
        class_name_anchors, raw_prompt_bank = build_text_bank(
            clip_model=clip_model,
            tokenizer=tokenizer,
            class_names=all_class_names,
            prompt_map=prompt_map,
            device=device,
        )

        seen_indices_list = [class_to_idx[name] for name in train_classes]
        unseen_indices_list = [class_to_idx[name] for name in unseen_test_classes]
        seen_indices = torch.tensor(seen_indices_list, device=device, dtype=torch.long)

        purified_prototypes = build_dap_prototypes(
            train_features=train_features,
            train_labels=train_labels,
            raw_prompt_bank=raw_prompt_bank,
            class_name_anchors=class_name_anchors,
            seen_class_indices=seen_indices_list,
            best_k=cfg.best_k,
            max_neg_samples=cfg.max_neg_samples,
        )

    model = ASPA(
        class_name_anchors=class_name_anchors,
        purified_prototypes=purified_prototypes,
        alpha_init=cfg.alpha_init,
        temperature=cfg.temperature,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_h = 0.0
    best_path = output_dir / "best_model.pth"

    for epoch in range(cfg.epochs):
        model.train()

        for images, labels, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)

            logits, adapted_features = model(image_features)

            loss_ce = criterion(logits, labels)
            loss_attr = (1 - F.cosine_similarity(adapted_features, model.purified_prototypes[labels], dim=-1)).mean()
            loss_dist = (1 - F.cosine_similarity(adapted_features, image_features, dim=-1)).mean()

            loss = loss_ce + cfg.lambda_attr * loss_attr + cfg.lambda_dist * loss_dist

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % cfg.eval_interval == 0 or epoch + 1 == cfg.epochs:
            acc_seen, acc_unseen, h_score = evaluate(
                model=model,
                clip_model=clip_model,
                loader_seen=seen_test_loader,
                loader_unseen=unseen_test_loader,
                seen_indices=seen_indices,
                seen_bias=cfg.seen_bias,
                device=device,
            )

            print(
                f"Epoch [{epoch + 1:03d}/{cfg.epochs}] | "
                f"Seen: {acc_seen * 100:.2f}% | "
                f"Unseen: {acc_unseen * 100:.2f}% | "
                f"H: {h_score * 100:.2f}%"
            )

            if h_score > best_h:
                best_h = h_score
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_to_idx": class_to_idx,
                        "all_class_names": all_class_names,
                        "seen_indices": seen_indices_list,
                        "unseen_indices": unseen_indices_list,
                        "config": vars(cfg),
                    },
                    best_path,
                )

    print(f"Training finished. Best H-Score: {best_h * 100:.2f}%")
    print(f"Best checkpoint saved to: {best_path}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/plantwild_v2")
    parser.add_argument("--prompt_path", type=str, default="data/prompts/prompt_cleaned_filtered.json")
    parser.add_argument("--backbone", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--best_k", type=int, default=30)
    parser.add_argument("--lambda_attr", type=float, default=1.2)
    parser.add_argument("--lambda_dist", type=float, default=30.0)
    parser.add_argument("--seen_bias", type=float, default=2.0)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--max_neg_samples", type=int, default=2000)
    parser.add_argument("--alpha_init", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/aspa")

    args = parser.parse_args()
    return Config(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
