import os
import save
import torch
import dataset
import numpy as np
from tqdm import tqdm
from torch.nn import functional
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score

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

def evaluate_deep(model, k : int = 5, fold : int = 0, model_name : str = "Model", save_path : str = './evaluate'):
    # 评估结果存储路径
    save_path = os.path.join(save_path, model_name)
    if not os.path.exists(save_path): # 如果路径不存在
        os.mkdir(save_path) # 创建文件夹
    
    # 数据集变换
    _, test_transform = dataset.get_basic_transform()

    # 加载数据集
    _, test_dataloaders = dataset.get_POLLEN73S_K_Fold(
        k = k,
        batch_size=32,
        test_transform=test_transform,
    )

    # 选择fold对应的测试集加载器
    test_dataloader = test_dataloaders[0 if fold >= k else fold]

    pred_prob = [] # 分类预测概率
    pred_labels = np.array([]) # 分类标签
    true_labels = np.array([]) # 真实标签
    eval_acc = 0.0 # 评估准确率
    eval_cnt = 0 # 样本数量
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 得到测试集结果
    with torch.no_grad():
        model.eval()

        for X, y in tqdm(test_dataloader, "Evaluation Iteration"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_pred = model(X) # 前向传播
            y_pred = functional.softmax(y_pred, dim=1) # 获得分类概率

            _, predicted = y_pred.max(1) # 得到标签

            pred_prob.extend(y_pred.cpu().numpy())
            pred_labels = np.hstack([pred_labels, predicted.cpu().numpy()])
            true_labels = np.hstack([true_labels, y.cpu().numpy()])

            eval_cnt += y.size(0)
            eval_acc += predicted.eq(y).sum().item()
    eval_acc = 0.0 if eval_cnt == 0 else eval_acc / eval_cnt # 计算评估误差
    evaluate(eval_acc, pred_prob, pred_labels, true_labels, model_name, save_path)
