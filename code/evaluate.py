import os
import save
import torch
import dataset
import numpy as np
from tqdm import tqdm
from torch.nn import functional
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from torchvision.utils import make_grid
from model import ViTForPollenClassification, ViTLForPollenClassification

def visualize_feature_maps(model, image, model_name, save_path, sample_idx):
    """
    可视化CNN模型的特征图
    """
    model.eval()
    DEVICE = next(model.parameters()).device
    
    # 注册钩子捕获特征图
    feature_maps = []
    def hook(module, input, output):
        feature_maps.append(output.detach().cpu())
    
    # 根据模型类型选择目标层
    if "ResNet" in model_name:
        target_layer = model.res.layer1[-1].conv2
    elif "DenseNet" in model_name:
        target_layer = model.dense.features.denseblock1.denselayer1.conv2
    else:
        return
    
    handle = target_layer.register_forward_hook(hook)
    
    # 前向传播获取特征图
    with torch.no_grad():
        model(image.unsqueeze(0).to(DEVICE))
    
    handle.remove()
    
    if not feature_maps:
        return
    
    # 可视化特征图
    feats = feature_maps[0][0]
    if "ResNet" in model_name:
        grid = make_grid(feats.unsqueeze(1), nrow=16, normalize=True, pad_value=0.5)
    elif "DenseNet" in model_name:
        grid = make_grid(feats.unsqueeze(1), nrow=8, normalize=True, pad_value=0.5)
    else:
        return
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"{model_name} Feature Maps - Sample {sample_idx}")
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f'feature_map_{sample_idx}.png'))
    plt.close()

def visualize_attention_maps(model, image, model_name, save_path, sample_idx, prefix='pos'):
    """
    可视化ViT模型的每一层注意力图
    """
    model.eval()
    DEVICE = next(model.parameters()).device
    
    # 创建存储所有层注意力的列表
    all_attentions = []
    
    # 为每一层注册钩子
    handles = []
    for i, layer in enumerate(model.vit.encoder.layers):
        def hook_factory(layer_idx):
            def hook(module, input, output):
                # 获取当前层的注意力权重
                attn = output[1].detach().cpu()
                all_attentions.append((layer_idx, attn))
            return hook
        
        handle = layer.self_attention.register_forward_hook(hook_factory(i))
        handles.append(handle)
    
    # 前向传播获取所有层的注意力
    with torch.no_grad():
        model(image.unsqueeze(0).to(DEVICE))
    
    # 移除所有钩子
    for handle in handles:
        handle.remove()
    
    if not all_attentions:
        return
    
    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_denorm = image * std + mean
    image_denorm = image_denorm.clamp(0, 1)
    
    # 为每个样本创建单独的文件夹
    sample_dir = os.path.join(save_path, prefix, f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)
    
    # 处理每一层的注意力
    for layer_idx, attn in all_attentions:
        # 取CLS token对其他token的注意力
        cls_attn = attn[0, 0, 1:]  # 形状为 (num_patches,)
        
        # 重塑为二维网格
        grid_size = 14
        cls_attn = cls_attn.reshape(grid_size, grid_size)
        
        # 将注意力图上采样到原始图像尺寸
        upsampled_attn = torch.nn.functional.interpolate(
            cls_attn.unsqueeze(0).unsqueeze(0),  # 添加batch和channel维度
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )[0, 0]  # 移除batch和channel维度
        
        # 将注意力图转换为热力图
        heatmap = upsampled_attn.numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # 归一化到0-1
        
        # 创建叠加图像
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Layer {layer_idx} Attention", fontsize=16)
        
        # 原始图像
        axes[0].imshow(image_denorm.permute(1, 2, 0).numpy())
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 注意力热力图
        im = axes[1].imshow(heatmap, cmap='hot', interpolation='bilinear')
        axes[1].set_title('Attention Heatmap')
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].axis('off')
        
        # 叠加图像
        alpha = 0.5  # 透明度
        axes[2].imshow(image_denorm.permute(1, 2, 0).numpy())
        axes[2].imshow(heatmap, cmap='hot', alpha=alpha, interpolation='bilinear')
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, f'layer_{layer_idx}_attention.png'))
        plt.close()
        
        # 单独保存叠加图像
        plt.figure(figsize=(8, 8))
        plt.imshow(image_denorm.permute(1, 2, 0).numpy())
        plt.imshow(heatmap, cmap='hot', alpha=alpha, interpolation='bilinear')
        plt.title(f'Layer {layer_idx} Attention Overlay')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, f'layer_{layer_idx}_overlay.png'))
        plt.close()
    
    # 创建所有层的概览图
    if len(all_attentions) > 12:
        fig, axes = plt.subplots(6, 4, figsize=(20, 15))
    else:
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f"Attention Maps Across Layers - Sample {sample_idx}", fontsize=20)
    
    for idx, (layer_idx, attn) in enumerate(all_attentions):
        row = idx // 4
        col = idx % 4
        
        # 处理当前层的注意力
        cls_attn = attn[0, 0, 1:]
        grid_size = 14
        cls_attn = cls_attn.reshape(grid_size, grid_size)
        
        # 创建热力图
        axes[row, col].imshow(cls_attn, cmap='hot', interpolation='nearest')
        axes[row, col].set_title(f'Layer {layer_idx}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{prefix}_sample_{sample_idx}_all_layers.png'))
    plt.close()

def evaluate(eval_acc, pred_prob, pred_labels, true_labels, model_name : str = "Model", save_path : str = './evaluate'):
    """
    模型性能评估测试
    @param eval_acc 评估精度
    @param pred_prob 模型在评估测试集上的概率输出 每行一个样本
    @param pred_labels 模型在评估测试集上的分类结果
    @param true_labels 评估测试集的真实分类结果
    @param model_name 模型名称
    @param save_path 模型存储路径
    """
    pred_prob = np.array(pred_prob) # 得到每个测试样本的多分类预测概率
    class_num = pred_prob.shape[1]

    mat = np.zeros((class_num, class_num)) # 混淆矩阵
    predict = np.array([pred_labels, true_labels]).T # 预测结果 每一行一个样本 分别为预测标签 真实标签

    # 生成混淆矩阵并保存图片
    for item in predict:
        true_label = int(item[1])
        pred_label = int(item[0])
        mat[true_label, pred_label] += 1
    s = np.sum(mat, axis=1, keepdims=True)
    mat = mat / s # 归一化 转为比例
    plt.matshow(mat, cmap='Purples')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    
    # 计算F1-score
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')

    # 计算AUROC
    auroc = roc_auc_score(true_labels, pred_prob, average='macro', multi_class='ovr')

    # 保存结果
    save.save_evaluate(os.path.join(save_path, 'report.txt'), model_name, eval_acc, macro_f1, auroc)


def evaluate_deep(model, k: int = 5, fold: int = 0, model_name: str = "Model", save_path: str = './evaluate'):
    # 评估结果存储路径
    save_path = os.path.join(save_path, model_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # 数据集变换
    _, test_transform = dataset.get_basic_transform()

    # 加载数据集
    _, test_dataloaders = dataset.get_POLLEN73S_K_Fold(
        k=k,
        batch_size=32,
        test_transform=test_transform,
    )

    # 选择fold对应的测试集加载器
    test_dataloader = test_dataloaders[0 if fold >= k else fold]

    pred_prob = []  # 分类预测概率
    pred_labels = np.array([])  # 分类标签
    true_labels = np.array([])  # 真实标签
    eval_acc = 0.0  # 评估准确率
    eval_cnt = 0  # 样本数量
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 收集样本用于可视化
    sample_images = []
    sample_indices = []
    
    # 得到测试集结果
    with torch.no_grad():
        model.eval()
        for idx, (X, y) in enumerate(tqdm(test_dataloader, "Evaluation Iteration")):
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_pred = model(X)  # 前向传播
            y_pred = functional.softmax(y_pred, dim=1)  # 获得分类概率

            _, predicted = y_pred.max(1)  # 得到标签

            pred_prob.extend(y_pred.cpu().numpy())
            pred_labels = np.hstack([pred_labels, predicted.cpu().numpy()])
            true_labels = np.hstack([true_labels, y.cpu().numpy()])

            eval_cnt += y.size(0)
            eval_acc += predicted.eq(y).sum().item()
            
            # 收集前5个样本用于可视化
            if len(sample_images) < 5:
                sample_images.append(X[0].cpu())
                sample_indices.append(idx)
    
    eval_acc = 0.0 if eval_cnt == 0 else eval_acc / eval_cnt  # 计算评估误差
    evaluate(eval_acc, pred_prob, pred_labels, true_labels, model_name, save_path)
    
    # 可视化中间结果
    for i, image in enumerate(sample_images):
        if "ViT" in model_name:
            # 展示已学习模型的注意力权重
            visualize_attention_maps(model, image, model_name, save_path, i, 'pos')

            # 引入未迁移学习的模型作对比
            if "B-16" in model_name:
                neg_model = ViTForPollenClassification(all_weight=True)
            else:
                neg_model = ViTLForPollenClassification(all_weight=True)
            visualize_attention_maps(neg_model, image, model_name, save_path, i, 'neg')
        else:
            visualize_feature_maps(model, image, model_name, save_path, i)

# Usage:
# path = r".\model\vitl\vitl_fold0_best_model.model"
# name = "ViT-L-16"
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# obj, history = save.load_model(path, None, map_location=DEVICE, full=True)
# model = obj['model'].to(DEVICE)
# evaluate_deep(model, model_name=name)
