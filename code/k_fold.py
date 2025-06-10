import torch
import numpy as np
from torch import nn
from torch import optim

import dataset
from train import train, save_obj
from model import ViTForPollenClassification, DenseNet201ForPollenClassification
from figure import save_figure
from scheduler import warmup_cosine_scheduler

def k_fold_val_ViT_A(
        *,
        k : int = 5,
        epochs : int = 10,
        batch_size : int = 32,
        random_state : int = 42,
        save_model : bool = False
    ):

    """
    进行K折交叉验证
    @param k k-fold交叉验证折数
    @param epochs 每折训练轮次
    @param batch_size 每折batch大小
    @param random_state 随机数种子
    @param save_model 是否保存各折的最佳模型
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
        model = ViTForPollenClassification(all_weight=True).to(DEVICE)

        # 优化器 采用AdamW
        optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.98), lr=5e-5, weight_decay=1e-3)

        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        # 进行训练
        obj = save_obj(model, optimizer, scheduler, loss_fn, DEVICE, batch_size, random_state) # 打包参数
        obj, history, best_acc = train(obj, train_dataloader, test_dataloader, epochs, save_model, f"fold{fold}") # 训练

        # 储存结果
        histories.append(history)
        best_accs.append(best_acc)
    
    best_accs = np.array(best_accs)
    return best_accs, best_accs.mean(), best_accs.std(), histories # 计算均值和标准差

def k_fold_val_ViT_B(
        *,
        k : int = 5,
        warmup_steps : int = 10,
        cosine_steps : int = 10,
        batch_size : int = 32,
        random_state : int = 42,
        save_model : bool = False
    ):

    """
    进行K折交叉验证
    @param k k-fold交叉验证折数
    @param warmup_steps 每折预热轮次
    @param cosine_steps 每折衰减轮次
    @param batch_size 每折batch大小
    @param random_state 随机数种子
    @param save_model 是否保存各折的最佳模型
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
        model = ViTForPollenClassification(all_weight=True).to(DEVICE)

        # 优化器
        optimizer = optim.optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.98), lr=8e-5, weight_decay=5e-4)

        # 学习率调度器
        scheduler = warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, cosine_steps=cosine_steps, min_lr=1e-6)

        # 进行训练
        obj = save_obj(model, optimizer, scheduler, loss_fn, DEVICE, batch_size, random_state) # 打包参数
        obj, history, best_acc = train(obj, train_dataloader, test_dataloader, (warmup_steps + cosine_steps), False, save_model, f"fold{fold}") # 训练

        # 储存结果
        histories.append(history)
        best_accs.append(best_acc)
    
    best_accs = np.array(best_accs)
    return best_accs, best_accs.mean(), best_accs.std(), histories # 计算均值和标准差

def k_fold_val_DenseNet_A(
        *,
        k : int = 5,
        epochs : int = 10,
        batch_size : int = 32,
        random_state : int = 42,
        save_model : bool = False
    ):

    """
    进行K折交叉验证
    @param k k-fold交叉验证折数
    @param epochs 每折训练轮次
    @param batch_size 每折batch大小
    @param random_state 随机数种子
    @param save_model 是否保存各折的最佳模型
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
        model = DenseNet201ForPollenClassification(all_weight=True).to(DEVICE)

        # 优化器
        optimizer = optim.optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.98), lr=1e-4, weight_decay=1e-3)

        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        # 进行训练
        obj = save_obj(model, optimizer, scheduler, loss_fn, DEVICE, batch_size, random_state) # 打包参数
        obj, history, best_acc = train(obj, train_dataloader, test_dataloader, epochs, False, save_model, f"fold{fold}") # 训练

        # 储存结果
        histories.append(history)
        best_accs.append(best_acc)
    
    best_accs = np.array(best_accs)
    return best_accs, best_accs.mean(), best_accs.std(), histories # 计算均值和标准差

if False: # 0.9834+-0.0048
    WARMUP = 50
    COSINE = 120
    best_accs, mean, std, histories = k_fold_val_ViT_B(warmup_steps=WARMUP, cosine_steps=COSINE)
    print("\n------K-Fold Validation Report------")
    print(f"K: {len(best_accs)}, Epochs: {COSINE}")
    print(f"Best accs: {best_accs}")
    print(f"Acc mean: {mean}, Acc std: {std}")

    for i in range(len(histories)):
        save_figure(histories[i], f"./figure/{i:02d}.png", f"Fine-tuning on ViT-B-16 Model - Fold {i} Epoch {COSINE}")


if False: # 0.9885+-0.0049
    EPOCHS = 75
    best_accs, mean, std, histories = k_fold_val_DenseNet_A(epochs=EPOCHS)
    print("\n------K-Fold Validation Report------")
    print(f"K: {len(best_accs)}, Epochs: {EPOCHS}")
    print(f"Best accs: {best_accs}")
    print(f"Acc mean: {mean}, Acc std: {std}")

    for i in range(len(histories)):
        save_figure(histories[i], f"./figure/{i:02d}.png", f"Fine-tuning on DenseNet-201 Model - Fold {i} Epoch {len(histories[i]['train_loss'])}")
