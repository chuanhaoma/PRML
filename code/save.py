import os
import torch
import threading
import seaborn as sns
import matplotlib.pyplot as plt

threadLock = threading.Lock()

MODEL_SAVE_PATH = './model'

def save_model(obj, history, prefix : str = "", path : str = MODEL_SAVE_PATH):
    """
    存储模型与训练记录到文件
    @param obj 训练参数
    @param history 训练记录
    @param prefix 模型文件名前缀
    @param path 保存路径
    """
    index = history['save_index']

    file_name = f"best_model_{index:02d}.model"
    if len(prefix) > 0: # 如果有前缀
        file_name = prefix + "_" + file_name
    
    save_path = os.path.join(path, file_name)
    model = obj['model'].state_dict() # 提取参数
    obj['model'] = model # 只保存参数
    save_content = {'obj': obj, 'history': history}
    torch.save(save_content, save_path)

def load_model(path, model_class, map_location=None):
    """
    从文件加载模型与训练记录
    @param path 文件保存路径
    @param model_class 模型类
    @param map_location 模型加载设备
    @return 模型参数与训练记录
    """
    model = model_class(all_weight=True) # 创建空模型
    load_content = torch.load(path, map_location=map_location, weights_only=False) # 加载模型内容
    obj, history = load_content['obj'], load_content['history']
    model.load_state_dict(obj['model'])
    return obj, history

def save_report(filename, best_accs, mean, std, epochs):
    """
    保存模型验证报告
    """
    with open(filename, 'w+') as file:
        file.write("------K-Fold Validation Report------\n")
        file.write(f"K: {len(best_accs)}, Epochs: {epochs}\n")
        file.write(f"Best accs: {best_accs}\n")
        file.write(f"Acc mean: {mean}, Acc std: {std}\n")

def save_evaluate(filename, model_name, eval_acc, macro_f1, auroc):
    """
    保存模型评估报告
    """
    with open(filename, 'w+') as file:
        file.write(f"------{model_name} Evaluate Report------\n")
        file.write(f"Eval Acc: {eval_acc}, Macro F1-score: {macro_f1}, AUROC: {auroc}\n")
        file.write(f"F1-score for each class:\n")

def save_figure(history, fig_path, fig_title):
    # 设置绘图样式
    sns.set_style("darkgrid")

    # 创建一个 2x2 的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # figsize 可以根据需要调整

    # 绘制 train_loss
    sns.lineplot(x=range(len(history['train_loss'])), y=history['train_loss'], ax=axes[0, 0])
    axes[0, 0].set_title('Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')

    # 绘制 train_acc
    sns.lineplot(x=range(len(history['train_acc'])), y=history['train_acc'], ax=axes[0, 1])
    axes[0, 1].set_title('Train Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')

    # 绘制 test_loss
    sns.lineplot(x=range(len(history['test_loss'])), y=history['test_loss'], ax=axes[1, 0])
    axes[1, 0].set_title('Test Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')

    # 绘制 test_acc
    sns.lineplot(x=range(len(history['test_acc'])), y=history['test_acc'], ax=axes[1, 1])
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')

    # 训练轮次
    epoch = len(history['test_acc'])
    # 添加总标题
    plt.suptitle(fig_title, fontsize=16)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(fig_path)

class Save(threading.Thread):
    def __init__(self, obj, history, prefix : str = "", path : str = MODEL_SAVE_PATH):
        threading.Thread.__init__(self)
        self.obj = obj
        self.history = history
        self.prefix = prefix
        self.path = path
    
    def run(self):
        threadLock.acquire()
        save_model(self.obj, self.history, self.prefix, self.path)
        threadLock.release()
