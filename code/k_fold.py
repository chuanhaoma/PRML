import torch
import numpy as np
from torch import nn
from torch import optim

import dataset
from train import train, save_obj
from model import ViTForPollenClassification, DenseNet201ForPollenClassification, ViTLForPollenClassification, ViTHForPollenClassification, ResNet50ForPollenClassification
from save import save_figure, save_report
from evaluate import evaluate_deep
from scheduler import CosineDecayWithLinearWarmup

def k_fold_val(
        *,
        model_class,
        k : int = 5,
        base_lr : float = 5e-6,
        weight_decay : float = 1e-3,
        min_lr : float = 1e-7,
        warmup_steps : int = 10,
        cosine_steps : int = 10,
        batch_size : int = 32,
        random_state : int = 42,
        save_model : bool = False,
        prefix : str = "model"
    ):

    """
    进行K折交叉验证
    @param model_class 模型类别
    @param k k-fold交叉验证折数
    @param base_lr 基础学习率
    @param weight_decay 正则化权重
    @param min_lr 学习率最小值
    @param warmup_steps 每折预热轮次
    @param cosine_steps 每折衰减轮次
    @param batch_size 每折batch大小
    @param random_state 随机数种子
    @param save_model 是否保存各折的最佳模型
    @param prefix 模型储存前缀
    """

    # 每折训练数据
    best_accs = [] # 最佳准确率
    histories = [] # 训练数据

    # 数据集变换
    train_transform, test_transform = dataset.get_basic_transform()
    
    # 加载数据集
    train_dataloaders, test_dataloaders = dataset.get_POLLEN73S_K_Fold(
        k = k,
        batch_size=batch_size,
        train_transform=train_transform,
        test_transform=test_transform,
        num_workers=8,
        pre_load=True
    )

    # 训练设备
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    for fold in range(k):
        # 本折数据集
        train_dataloader = train_dataloaders[fold] # 训练集
        test_dataloader = test_dataloaders[fold] # 测试集

        # 模型
        model = model_class(all_weight=True).to(DEVICE)

        # 优化器
        optimizer = optim.optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.98), lr=base_lr, weight_decay=weight_decay)

        # 学习率调度器
        scheduler = CosineDecayWithLinearWarmup(optimizer, warmup_steps=warmup_steps, cosine_steps=cosine_steps, min_lr=min_lr)

        # 进行训练
        obj = save_obj(model, optimizer, scheduler, loss_fn, DEVICE, batch_size, random_state) # 打包参数
        obj, history, best_acc = train(obj, train_dataloader, test_dataloader, (warmup_steps + cosine_steps), False, save_model, f"{prefix}_fold{fold}", max_length=1) # 训练

        # 储存结果
        histories.append(history)
        best_accs.append(best_acc)
    
    best_accs = np.array(best_accs)
    return model, best_accs, best_accs.mean(), best_accs.std(), histories # 计算均值和标准差

# ViT-B-16
if True:
    WARMUP = 50
    COSINE = 120
    model, best_accs, mean, std, histories = k_fold_val(model_class=ViTForPollenClassification, warmup_steps=WARMUP, cosine_steps=COSINE, save_model=True, prefix='vitb')
    save_report('./model/vit_report.txt', best_accs, mean, std, COSINE)
    evaluate_deep(model, model_name='ViT-B-16')
    for i in range(len(histories)):
        save_figure(histories[i], f"./figure/vit_{i:02d}.png", f"Fine-tuning on ViT-B-16 Model - Fold {i} Epoch {COSINE}")

# ViT-L-16
if True:
    WARMUP = 25
    COSINE = 25
    model, best_accs, mean, std, histories = k_fold_val(model_class=ViTLForPollenClassification, warmup_steps=WARMUP, cosine_steps=COSINE, save_model=True, prefix='vitl')
    save_report('./model/vitl_report.txt', best_accs, mean, std, COSINE)
    evaluate_deep(model, model_name='ViT-L-16')
    for i in range(len(histories)):
        save_figure(histories[i], f"./figure/vitl_{i:02d}.png", f"Fine-tuning on ViT-L-16 Model - Fold {i} Epoch {COSINE}")

# ViT-H-14
if False:
    WARMUP = 50
    COSINE = 120
    model, best_accs, mean, std, histories = k_fold_val(model_class=ViTHForPollenClassification, warmup_steps=WARMUP, cosine_steps=COSINE, save_model=True, prefix='vith')
    save_report('./model/vith_report.txt', best_accs, mean, std, COSINE)

    for i in range(len(histories)):
        save_figure(histories[i], f"./figure/vith_{i:02d}.png", f"Fine-tuning on ViT-H-14 Model - Fold {i} Epoch {COSINE}")

# DenseNet-201
if True:
    WARMUP = 50
    COSINE = 120
    model, best_accs, mean, std, histories = k_fold_val(model_class=DenseNet201ForPollenClassification, base_lr=5e-5, warmup_steps=WARMUP, cosine_steps=COSINE, save_model=True, prefix='dense')
    save_report('./model/dense_report.txt', best_accs, mean, std, COSINE)
    evaluate_deep(model, model_name='DenseNet-201')
    for i in range(len(histories)):
        save_figure(histories[i], f"./figure/dense_{i:02d}.png", f"Fine-tuning on DenseNet-201 Model - Fold {i} Epoch {len(histories[i]['train_loss'])}")

# ResNet-50
if True:
    WARMUP = 50
    COSINE = 120
    model, best_accs, mean, std, histories = k_fold_val(model_class=ResNet50ForPollenClassification, base_lr=1e-4, warmup_steps=WARMUP, cosine_steps=COSINE, save_model=True, prefix='resnet')
    save_report('./model/resnet_report.txt', best_accs, mean, std, COSINE)
    evaluate_deep(model, model_name='ResNet-50')
    for i in range(len(histories)):
        save_figure(histories[i], f"./figure/resnet_{i:02d}.png", f"Fine-tuning on ResNet-50 Model - Fold {i} Epoch {len(histories[i]['train_loss'])}")
