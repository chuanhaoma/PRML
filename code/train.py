import os
import torch
from torch import nn
from torch import optim
from tqdm import tqdm

import dataset
from model import ViTForPollenClassification

MODEL_SAVE_PATH = './model'

def save_obj(model, optimizer, scheduler, loss_fn, DEVICE, batch_size = 16, random_state = 42):
    obj = {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_fn': loss_fn,
        'DEVICE': DEVICE,
        'batch_size': batch_size,
        'random_state': random_state,
    }
    return obj

def load_obj(obj):
    return obj['model'], obj['optimizer'], obj['scheduler'], obj['loss_fn'], obj['DEVICE'], obj['batch_size'], obj['random_state']

def train_step(model, loader, optimizer, loss_fn, DEVICE):
    """
    模型训练函数

    @param model 模型
    @param loader 训练集加载器
    @param optimizer 优化器
    @param loss_fn 损失函数
    @param DEVICE 训练设备

    @return 训练误差与训练精度
    """
    model.train() # 模型切换至训练模式
    train_loss = 0.0 # 训练误差
    train_acc = 0.0 # 训练准确率
    train_cnt = 0 # 训练数量

    for X, y in tqdm(loader, "Train Iteration"):
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad() # 清除梯度
        y_pred = model(X) # 前向传播
        loss = loss_fn(y_pred, y) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

        train_loss += loss.item() * X.size(0) # 统计损失
        _, predicted = y_pred.max(1) # 得到标签
        
        train_cnt += y.size(0)
        train_acc += predicted.eq(y).sum().item()
    
    train_loss = 0.0 if train_cnt == 0 else train_loss / train_cnt # 计算训练误差
    train_acc = 0.0 if train_cnt == 0 else train_acc / train_cnt # 计算训练准确率

    return train_loss, train_acc

def test_step(model, loader, loss_fn, DEVICE):
    """
    模型测试函数

    @param model 模型
    @param loader 测试集加载器
    @param loss_fn 损失函数
    @param DEVICE 训练设备

    @return 测试误差与测试精度
    """
    model.eval() # 模型切换至推理模式
    test_loss = 0.0
    test_acc = 0.0
    test_cnt = 0

    with torch.no_grad():
        for X, y in tqdm(loader, "Test Iteration"):
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            y_pred = model(X) # 前向传播
            loss = loss_fn(y_pred, y) # 计算损失
            test_loss += loss.item() * X.size(0) # 统计损失
            _, predicted = y_pred.max(1) # 得到标签
        
            test_cnt += y.size(0)
            test_acc += predicted.eq(y).sum().item()

        test_loss = 0.0 if test_cnt == 0 else test_loss / test_cnt # 计算测试误差
        test_acc = 0.0 if test_cnt == 0 else test_acc / test_cnt # 计算训练误差
    
    return test_loss, test_acc

def save_model(obj, history, max_length : int = 3, prefix : str = "", path : str = MODEL_SAVE_PATH):
    """
    存储模型与训练记录到文件
    @param obj 训练参数
    @param history 训练记录
    @param max_length 模型保存最大数量 
    @param prefix 模型文件名前缀
    @param path 保存路径
    """
    index = history['save_index'] + 1
    index = 0 if (index >= max_length) else index
    history['save_index'] = index

    file_name = f"best_model_{index:02d}.model"
    if len(prefix) > 0: # 如果有前缀
        file_name = prefix + "_" + file_name
    
    save_path = os.path.join(path, file_name)
    save_content = {'obj': obj, 'history': history}
    torch.save(save_content, save_path)

def load_model(path):
    """
    从文件加载模型与训练记录
    @param path 文件保存路径
    @return 模型参数与训练记录
    """
    load_content = torch.load(path)
    return load_content['obj'], load_content['history']

def train(obj, train_loader, test_loader, epochs : int = 10, save : bool = True, save_prefix : str = "", path : str = MODEL_SAVE_PATH):
    """
    根据参数与数据集训练模型
    """
    model, optimizer, scheduler, loss_fn, DEVICE, _, _ = load_obj(obj)
    model.to(DEVICE)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'save_index': -1} # 历史数据
    best_acc = 0.0 # 最佳acc

    tqdm_iter = tqdm(range(epochs), "Train Epoch")
    for epoch in tqdm_iter:
        # 首先进行训练
        train_loss, train_acc = train_step(model, train_loader, optimizer, loss_fn, DEVICE)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        LR = optimizer.param_groups[0]['lr']
        # 打印训练结果
        tqdm_iter.write(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Train Acc: {(train_acc * 100):.2f}%, lr: {LR:.6f}")

        # 随后进行测试
        test_loss, test_acc = test_step(model, test_loader, loss_fn, DEVICE)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # 打印测试结果
        tqdm_iter.write(f"Epoch {epoch}: Test Loss: {test_loss:.6f}, Test Acc: {(test_acc * 100):.2f}%")
        
        # 学习率调度
        scheduler.step()

        if test_acc > best_acc: # 如果当前测试acc大于最佳acc
            best_acc = test_acc
            obj = save_obj(model, optimizer, scheduler, loss_fn, DEVICE, obj['batch_size'], obj['random_state'])
            if save: # 如果选择保存模型文件
                save_model(obj, history, prefix=save_prefix, path=path)
                index = history['save_index']
                tqdm_iter.write(f"Best model saved with index {index}.")
    
    index = history['save_index']
    print(f'Training complete. Best test accuracy: {best_acc:.4f}. Model saved with index {index}.')

    return obj, history, best_acc

if False: # 分类头优化
    EPOCHS = 240
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ViTForPollenClassification() # 模型

    train_transform, test_transform = dataset.get_basic_transform() # 数据集变换
    train_dataloader, test_dataloader = dataset.get_POLLEN73S_dataloader(
        batch_size=16, 
        ratio=0.3, 
        train_transform=train_transform, 
        test_transform=test_transform
    ) # 产生数据集加载器

    loss_fn = nn.CrossEntropyLoss().to(DEVICE) # 交叉熵损失函数
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5) # 采用AdamW优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7) # 学习率调度器

    obj = save_obj(model, optimizer, scheduler, loss_fn, DEVICE)
    train(obj, train_dataloader, test_dataloader, EPOCHS, MODEL_SAVE_PATH)

if False: # 全量优化
    EPOCHS = 120
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ViTForPollenClassification(all_weight=True) # 模型

    train_transform, test_transform = dataset.get_basic_transform() # 数据集变换
    train_dataloader, test_dataloader = dataset.get_POLLEN73S_dataloader(
        batch_size=16, 
        ratio=0.3, 
        train_transform=train_transform, 
        test_transform=test_transform
    ) # 产生数据集加载器

    loss_fn = nn.CrossEntropyLoss().to(DEVICE) # 交叉熵损失函数
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3) # 采用AdamW优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6) # 学习率调度器

    obj = save_obj(model, optimizer, scheduler, loss_fn, DEVICE)
    train(obj, train_dataloader, test_dataloader, EPOCHS, './model2')
