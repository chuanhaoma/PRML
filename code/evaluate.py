import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
import torch
from torch.nn.functional import softmax
from tqdm import tqdm

def evaluate(model, test_dataloader, num_classes = 73):

    # 确保使用GPU如果可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # 设置为评估模式

    # 真实标签和预测概率
    all_labels, all_probs = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # 获取类别概率
            probs = softmax(outputs, dim=1)
            
            # 收集结果
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    # 合并所有批次的结果
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算每个类别的ROC曲线和AUC
    num_classes = 73
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))

    # 计算每个类别的ROC曲线
    for i in range(num_classes):
        # 获取当前类别的真实标签（二值化）
        class_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]
        
        # 计算ROC曲线
        fpr[i], tpr[i], _ = roc_curve(class_labels, class_probs)
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 绘制每个类别的ROC曲线（半透明）
        plt.plot(fpr[i], tpr[i], lw=1, alpha=0.3, 
                label=f'Class {i} (AUC = {roc_auc[i]:.2f})' if i % 10 == 0 else "")

    # 计算并绘制micro平均ROC曲线
    micro_labels = np.eye(num_classes)[all_labels].ravel()
    micro_probs = all_probs.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(micro_labels, micro_probs)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=4)

    # 计算并绘制macro平均ROC曲线
    # 首先聚合所有唯一的FPR点
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # 然后在这些点上插值所有TPR
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # 最终平均
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.plot(fpr["macro"], tpr["macro"],
            label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})',
            color='navy', linestyle=':', linewidth=4)

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curves')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('multiclass_roc.png', dpi=300)
    plt.show()

    # 计算F1分数
    pred_labels = np.argmax(all_probs, axis=1)

    # 每个类别的F1分数
    f1_per_class = f1_score(all_labels, pred_labels, average=None)

    # 整体F1分数（macro和weighted）
    f1_macro = f1_score(all_labels, pred_labels, average='macro')
    f1_weighted = f1_score(all_labels, pred_labels, average='weighted')

    print("\n" + "="*50)
    print(f"Macro F1 Score: {f1_macro:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print("="*50 + "\n")

    # 打印每个类别的性能
    print("Class-wise Performance:")
    print("-"*50)
    for i in range(num_classes):
        print(f"Class {i:2d}: AUC = {roc_auc[i]:.4f}, F1 = {f1_per_class[i]:.4f}")
    print("-"*50)