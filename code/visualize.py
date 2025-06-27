import os
import save
import torch
import dataset
import numpy as np
from tqdm import tqdm
from torch.nn import functional
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

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

def visualize_attention_maps(model, image, model_name, save_path, sample_idx):
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
    sample_dir = os.path.join(save_path, f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)
    out_heatmap = None
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
        if layer_idx == 2:
            out_heatmap = heatmap
    
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
    plt.savefig(os.path.join(save_path, f'sample_{sample_idx}_all_layers.png'))
    plt.close()
    
    # 返回注意力图用于可视化
    return out_heatmap

def visualize_predictions(image, heatmap, pred_prob, true_label, class_names, model_name, sample_idx):
    """
    可视化预测结果
    @param image: 原始图像 (tensor)
    @param heatmap: 注意力热力图 (numpy array)
    @param pred_prob: 预测概率 (list)
    @param true_label: 真实标签 (int)
    @param class_names: 类别名称列表
    @param model_name: 模型名称
    @param sample_idx: 样本索引
    """
    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_denorm = image * std + mean
    image_denorm = image_denorm.clamp(0, 1).permute(1, 2, 0).numpy()
    
    # 创建4个子图
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(f"Prediction Visualization - Sample {sample_idx} (True: {class_names[true_label]})", fontsize=24)
    
    # 子图1: 原始图像
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(image_denorm)
    ax1.set_title('Original Image', fontsize=18)
    ax1.axis('off')
    
    # 子图2: 注意力热力图
    ax2 = fig.add_subplot(2, 2, 2)
    im = ax2.imshow(heatmap, cmap='hot', interpolation='bilinear')
    ax2.set_title('Attention Heatmap', fontsize=18)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis('off')
    
    # 子图3: 注意力叠加图
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(image_denorm)
    ax3.imshow(heatmap, cmap='hot', alpha=0.5, interpolation='bilinear')
    ax3.set_title('Attention Overlay', fontsize=18)
    ax3.axis('off')
    
    # 子图4: 预测概率Top5
    ax4 = fig.add_subplot(2, 2, 4)
    topk = 5
    top_indices = np.argsort(pred_prob)[::-1][:topk]
    top_probs = [pred_prob[i] for i in top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    # 为真实标签添加特殊颜色
    colors = ['skyblue' if i != true_label else 'lightgreen' for i in top_indices]
    
    bars = ax4.barh(np.arange(topk), top_probs, color=colors)
    ax4.set_yticks(np.arange(topk))
    ax4.set_yticklabels(top_classes, fontsize=14)
    ax4.invert_yaxis()  # 从上到下显示
    ax4.set_xlabel('Probability', fontsize=16)
    ax4.set_title(f'Top {topk} Predictions', fontsize=18)
    
    # 添加概率值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', 
                ha='left', va='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.close()

def visualize(model, k: int = 5, fold: int = 0, model_name: str = "Model", save_path: str = './evaluate'):
    # 评估结果存储路径
    save_path = os.path.join(save_path, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # 数据集变换
    _, test_transform = dataset.get_basic_transform()

    # 加载数据集
    _, test_dataloaders = dataset.get_POLLEN73S_K_Fold(
        k=k,
        batch_size=32,
        test_transform=test_transform,
    )
    class_names = dataset.POLLEN73S.get_label_mapping(None)

    # 选择fold对应的测试集加载器
    test_dataloader = test_dataloaders[0 if fold >= k else fold]
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 收集样本用于可视化
    sample_images = []
    sample_true_labels = []
    sample_pred_probs = []
    sample_indices = []
    
    # 得到测试集结果
    with torch.no_grad():
        model.eval()
        for idx, (X, y) in enumerate(tqdm(test_dataloader, "Evaluation Iteration")):
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_pred = model(X)  # 前向传播
            y_pred_softmax = functional.softmax(y_pred, dim=1)  # 获得分类概率
            
            # 收集前1个样本用于可视化
            if len(sample_images) < 1:
                sample_images.append(X[0].cpu())
                sample_true_labels.append(y[0].item())
                sample_pred_probs.append(y_pred_softmax[0].cpu().numpy())
                sample_indices.append(idx)
            else:
                break
    
    # 可视化中间结果并展示预测
    for i, (image, true_label, pred_prob_i) in enumerate(zip(sample_images, sample_true_labels, sample_pred_probs)):
        # 获取注意力热力图或特征图
        if "ViT" in model_name:
            # 对于ViT，获取最后一层的注意力热力图
            heatmap = visualize_attention_maps(model, image, model_name, save_path, i)
        else:
            # 对于CNN，生成特征图的平均热力图
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
                target_layer = None
            
            if target_layer:
                handle = target_layer.register_forward_hook(hook)
                with torch.no_grad():
                    model(image.unsqueeze(0).to(DEVICE))
                handle.remove()
                
                if feature_maps:
                    feats = feature_maps[0][0]
                    # 计算特征图的平均热力图
                    heatmap = torch.mean(feats, dim=0).numpy()
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                    
                    # 上采样到原始图像尺寸
                    heatmap = torch.nn.functional.interpolate(
                        torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    )[0, 0].numpy()
                else:
                    heatmap = np.zeros((224, 224))
            else:
                heatmap = np.zeros((224, 224))
        
        # 可视化预测结果
        visualize_predictions(image, heatmap, pred_prob_i, true_label, class_names, model_name, i)
        
        # 保存中间结果
        if "ViT" in model_name:
            visualize_attention_maps(model, image, model_name, save_path, i)
        else:
            visualize_feature_maps(model, image, model_name, save_path, i)

# Usage:
# path = r".\model\resnet\resnet_fold0_best_model.model"
# name = "ResNet-50"
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# obj, history = save.load_model(path, None, map_location=DEVICE, full=True)
# model = obj['model'].to(DEVICE)
# visualize(model, model_name=name)