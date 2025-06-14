import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from preprocessing import RadarPreprocessor
from DualEfficientNetB0 import DualEfficientNetB0
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import copy
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os


class FocalLoss(nn.Module):
    """Focal Loss解决类别不平衡和困难样本问题"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """标签平滑防止过拟合"""
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_target = torch.zeros_like(pred)
        smooth_target.fill_(self.smoothing / (self.num_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), confidence)

        log_prob = F.log_softmax(pred, dim=1)
        loss = -(smooth_target * log_prob).sum(dim=1).mean()
        return loss


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.early_stop = False

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_model = copy.deepcopy(model.state_dict())


class RadarDataset(Dataset):
    def __init__(self, data, labels, label2id, augment=False, augment_prob=0.5):
        self.data = data
        self.labels = [label2id[lbl] for lbl in labels]
        self.augment = augment
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.data)

    def add_gaussian_noise(self, tensor, noise_factor=0.02):
        """添加高斯噪声"""
        if random.random() < self.augment_prob:
            noise = torch.randn_like(tensor) * noise_factor
            return tensor + noise
        return tensor

    def temporal_shift(self, tensor, max_shift=5):
        """时间维度随机平移"""
        if random.random() < self.augment_prob:
            shift = random.randint(-max_shift, max_shift)
            if shift > 0:
                tensor = F.pad(tensor, (0, 0, shift, 0))[:, :tensor.shape[1], :]
            elif shift < 0:
                tensor = F.pad(tensor, (0, 0, 0, -shift))[:, -shift:, :]
        return tensor

    def amplitude_scaling(self, tensor, scale_range=(0.8, 1.2)):
        """幅度缩放"""
        if random.random() < self.augment_prob:
            scale = random.uniform(*scale_range)
            return tensor * scale
        return tensor

    def __getitem__(self, idx):
        rtm, dtm = self.data[idx]

        # 转换为张量并归一化
        rtm_tensor = torch.tensor(rtm).permute(2, 0, 1).float() / 255.
        dtm_tensor = torch.tensor(dtm).permute(2, 0, 1).float() / 255.

        # 数据增强
        if self.augment:
            rtm_tensor = self.add_gaussian_noise(rtm_tensor)
            dtm_tensor = self.add_gaussian_noise(dtm_tensor)
            rtm_tensor = self.temporal_shift(rtm_tensor)
            dtm_tensor = self.temporal_shift(dtm_tensor)
            rtm_tensor = self.amplitude_scaling(rtm_tensor)
            dtm_tensor = self.amplitude_scaling(dtm_tensor)

        label = torch.tensor(self.labels[idx])
        return rtm_tensor, dtm_tensor, label


def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, correct_cls = 0, 0

    for rtm, dtm, label in dataloader:
        rtm, dtm, label = rtm.to(device), dtm.to(device), label.to(device)

        # 前向传播
        logits = model(rtm, dtm)
        loss = criterion(logits, label)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * rtm.size(0)
        correct_cls += (logits.argmax(1) == label).sum().item()

    total = len(dataloader.dataset)
    return total_loss / total, correct_cls / total


def evaluate(model, dataloader, criterion, device, label_names, return_predictions=False):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_trues = [], []

    num_classes = len(label_names)
    correct_per_class = [0 for _ in range(num_classes)]
    total_per_class = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for rtm, dtm, label in dataloader:
            rtm, dtm, label = rtm.to(device), dtm.to(device), label.to(device)
            logits = model(rtm, dtm)
            loss = criterion(logits, label)

            preds = logits.argmax(1)
            total_loss += loss.item() * rtm.size(0)
            correct += (preds == label).sum().item()

            for i in range(len(label)):
                true_label = label[i].item()
                pred_label = preds[i].item()
                total_per_class[true_label] += 1
                if pred_label == true_label:
                    correct_per_class[true_label] += 1

            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(label.cpu().numpy())

    total = len(dataloader.dataset)
    avg_loss = total_loss / total
    avg_acc = correct / total

    print(f"\n[Per-Class Accuracy]")
    for i in range(num_classes):
        if total_per_class[i] > 0:
            acc_i = correct_per_class[i] / total_per_class[i]
            print(f"Class {i:02d}: Acc = {acc_i:.3f} ({correct_per_class[i]}/{total_per_class[i]})")
        else:
            print(f"Class {i:02d}: No samples.")

    if return_predictions:
        return avg_loss, avg_acc, all_preds, all_trues
    return avg_loss, avg_acc


def plot_training_history(results, config_name, save_dir='param_vis'):
    """绘制训练历史图"""
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(results['train_losses']) + 1)

    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training History - {config_name}', fontsize=16)

    # 损失图
    ax1.plot(epochs, results['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, results['val_losses'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Loss vs Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率图
    ax2.plot(epochs, results['train_accs'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, results['val_accs'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_title('Accuracy vs Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 损失详细图（放大）
    ax3.plot(epochs, results['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax3.plot(epochs, results['val_losses'], 'r-', label='Val Loss', linewidth=2)
    ax3.set_title('Loss (Zoomed)')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # 设置y轴范围为损失的合理范围
    loss_min = min(min(results['train_losses']), min(results['val_losses']))
    loss_max = max(max(results['train_losses']), max(results['val_losses']))
    ax3.set_ylim([loss_min * 0.9, loss_max * 1.1])

    # 准确率详细图（放大）
    ax4.plot(epochs, results['train_accs'], 'b-', label='Train Acc', linewidth=2)
    ax4.plot(epochs, results['val_accs'], 'r-', label='Val Acc', linewidth=2)
    ax4.set_title('Accuracy (Zoomed)')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    # 设置y轴范围为准确率的合理范围
    acc_min = min(min(results['train_accs']), min(results['val_accs']))
    acc_max = max(max(results['train_accs']), max(results['val_accs']))
    ax4.set_ylim([max(0, acc_min * 0.95), min(1, acc_max * 1.05)])

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(save_dir, f'training_history_{config_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history plot saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, label_names, config_name, save_dir='param_vis'):
    """绘制混淆矩阵"""
    os.makedirs(save_dir, exist_ok=True)

    # 创建Class+序号的标签格式
    class_labels = [f'Class {i:02d}' for i in range(len(label_names))]

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 创建图形
    plt.figure(figsize=(12, 10))

    # 绘制混淆矩阵热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'})

    plt.title(f'Confusion Matrix - {config_name}', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(save_dir, f'confusion_matrix_{config_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to: {save_path}")

    # 计算并显示分类报告
    print(f"\nClassification Report - {config_name}:")
    print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))


def get_hyperparameter_configs():
    """定义不同的超参数配置进行网格搜索"""
    configs = [
        {
            'name': 'baseline_improved',
            'lr': 0.001,
            'batch_size': 32,
            'optimizer': 'Adam',
            'scheduler': 'CosineAnnealingWarmRestarts',
            'loss_function': 'CrossEntropy',
            'weight_decay': 1e-4,
            'dropout': 0.3,
            'grad_clip': 1.0,
            'augment_prob': 0.5
        },
        {
            'name': 'focal_loss_config',
            'lr': 0.0005,
            'batch_size': 24,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau',
            'loss_function': 'FocalLoss',
            'weight_decay': 1e-3,
            'dropout': 0.4,
            'grad_clip': 0.5,
            'augment_prob': 0.7
        },
        {
            'name': 'label_smoothing_config',
            'lr': 0.0008,
            'batch_size': 28,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingWarmRestarts',
            'loss_function': 'LabelSmoothing',
            'weight_decay': 2e-4,
            'dropout': 0.35,
            'grad_clip': 1.5,
            'augment_prob': 0.6
        }
    ]
    return configs


def create_optimizer(model, config):
    """根据配置创建优化器"""
    if config['optimizer'] == 'Adam':
        return optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        return optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        return optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])


def create_scheduler(optimizer, config):
    """根据配置创建学习率调度器"""
    if config['scheduler'] == 'CosineAnnealingWarmRestarts':
        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    elif config['scheduler'] == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    elif config['scheduler'] == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)


def create_loss_function(config, num_classes, class_weights=None):
    """根据配置创建损失函数"""
    if config['loss_function'] == 'CrossEntropy':
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        return nn.CrossEntropyLoss()
    elif config['loss_function'] == 'FocalLoss':
        return FocalLoss(alpha=1.0, gamma=2.0)
    elif config['loss_function'] == 'LabelSmoothing':
        return LabelSmoothingLoss(num_classes, smoothing=0.1)


def train_with_config(config, data, labels, filepaths, le, device):
    """使用指定配置训练模型"""
    print(f"\n{'=' * 50}")
    print(f"Training with config: {config['name']}")
    print(f"{'=' * 50}")

    # 数据划分
    train_data, temp_data, train_labels, temp_labels, train_paths, temp_paths = train_test_split(
        data, labels, filepaths, test_size=0.2, stratify=labels, random_state=42)
    val_data, test_data, val_labels, test_labels, val_paths, test_paths = train_test_split(
        temp_data, temp_labels, temp_paths, test_size=0.5, stratify=temp_labels, random_state=42)

    # 计算类别权重
    label_ids = le.transform(labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(label_ids), y=label_ids)
    class_weights = torch.FloatTensor(class_weights).to(device)

    # 创建数据集
    label2id = {lbl: i for i, lbl in enumerate(le.classes_)}
    train_set = RadarDataset(train_data, train_labels, label2id, augment=True,
                             augment_prob=config['augment_prob'])
    val_set = RadarDataset(val_data, val_labels, label2id, augment=False)
    test_set = RadarDataset(test_data, test_labels, label2id, augment=False)

    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], num_workers=2, pin_memory=True)

    # 创建模型
    model = DualEfficientNetB0(num_classes=len(label2id)).to(device)

    # 创建优化器、调度器和损失函数
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # 选择是否使用类别权重
    use_class_weights = config['loss_function'] == 'CrossEntropy'
    criterion = create_loss_function(config, len(label2id),
                                     class_weights if use_class_weights else None)

    # 早停
    early_stopper = EarlyStopping(patience=20, min_delta=0.001)

    best_val_acc = 0
    results = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': []
    }

    for epoch in range(100):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                                grad_clip=config['grad_clip'])

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, le.classes_)

        # 更新学习率
        if config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_acc)
        else:
            scheduler.step()

        # 记录结果
        results['train_losses'].append(train_loss)
        results['train_accs'].append(train_acc)
        results['val_losses'].append(val_loss)
        results['val_accs'].append(val_acc)

        print(
            f"[Epoch {epoch + 1}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 早停检查
        early_stopper(val_acc, model)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # 加载最佳模型并测试
    if early_stopper.best_model is not None:
        model.load_state_dict(early_stopper.best_model)

    # 绘制训练历史
    plot_training_history(results, config['name'])

    # 测试并获取预测结果用于混淆矩阵
    test_loss, test_acc, test_preds, test_trues = evaluate(
        model, test_loader, criterion, device, le.classes_, return_predictions=True)
    print(f"[Final Test] 20-Class Accuracy: {test_acc:.4f}")

    # 绘制混淆矩阵
    plot_confusion_matrix(test_trues, test_preds, le.classes_, config['name'])

    # 保存模型
    model_name = f'best_model_{config["name"]}.pth'
    torch.save(early_stopper.best_model, model_name)
    print(f"Model saved as {model_name}")

    return {
        'config': config,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'results': results,
        'model_path': model_name,
        'test_predictions': {'y_true': test_trues, 'y_pred': test_preds}
    }


def plot_all_configs_comparison(all_results, save_dir='param_vis'):
    """绘制所有配置的对比图"""
    os.makedirs(save_dir, exist_ok=True)

    if not all_results:
        return

    # 创建对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('All Configurations Comparison', fontsize=18)

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for i, result in enumerate(all_results):
        color = colors[i % len(colors)]
        config_name = result['config']['name']
        results = result['results']
        epochs = range(1, len(results['train_losses']) + 1)

        # 训练损失对比
        ax1.plot(epochs, results['train_losses'], color=color, linestyle='-',
                 label=f'{config_name} (Train)', linewidth=2, alpha=0.8)

        # 验证损失对比
        ax2.plot(epochs, results['val_losses'], color=color, linestyle='-',
                 label=f'{config_name} (Val)', linewidth=2, alpha=0.8)

        # 训练准确率对比
        ax3.plot(epochs, results['train_accs'], color=color, linestyle='-',
                 label=f'{config_name} (Train)', linewidth=2, alpha=0.8)

        # 验证准确率对比
        ax4.plot(epochs, results['val_accs'], color=color, linestyle='-',
                 label=f'{config_name} (Val)', linewidth=2, alpha=0.8)

    # 设置子图
    ax1.set_title('Training Loss Comparison', fontsize=14)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Validation Loss Comparison', fontsize=14)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_title('Training Accuracy Comparison', fontsize=14)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.set_title('Validation Accuracy Comparison', fontsize=14)
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存对比图
    save_path = os.path.join(save_dir, 'all_configs_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison plot saved to: {save_path}")


if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 设置matplotlib参数，避免字体问题
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    pre = RadarPreprocessor()
    data, labels, filepaths = pre.preprocess_all_data()
    print(f"Loaded {len(data)} samples.")

    le = LabelEncoder()
    label_ids = le.fit_transform(labels)
    print("Label mapping:", {lbl: i for i, lbl in enumerate(le.classes_)})

    # 获取所有配置
    configs = get_hyperparameter_configs()

    # 存储所有结果
    all_results = []

    # 逐个配置训练
    for config in configs:
        try:
            result = train_with_config(config, data, labels, filepaths, le, device)
            all_results.append(result)
        except Exception as e:
            print(f"Error with config {config['name']}: {e}")
            continue

    # 绘制所有配置的对比图
    plot_all_configs_comparison(all_results)

    # 总结结果
    print(f"\n{'=' * 60}")
    print("HYPERPARAMETER TUNING RESULTS")
    print(f"{'=' * 60}")

    # 按测试准确率排序
    all_results.sort(key=lambda x: x['test_acc'], reverse=True)

    for i, result in enumerate(all_results):
        print(f"{i + 1}. Config: {result['config']['name']}")
        print(f"   Best Val Acc: {result['best_val_acc']:.4f}")
        print(f"   Test Acc: {result['test_acc']:.4f}")
        print(f"   Model: {result['model_path']}")
        print()

    if all_results:
        best_result = all_results[0]
        print(f"   BEST CONFIG: {best_result['config']['name']}")
        print(f"   Test Accuracy: {best_result['test_acc']:.4f}")
        print(f"   Hyperparameters:")
        for key, value in best_result['config'].items():
            if key != 'name':
                print(f"     {key}: {value}")

        print(f"\n All param_vis have been saved to the 'param_vis' directory")
        print(f"   - Training history param_vis for each configuration")
        print(f"   - Confusion matrices for each configuration")
        print(f"   - Overall comparison plot for all configurations")
