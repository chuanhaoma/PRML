import torch
import seaborn as sns
import matplotlib.pyplot as plt

content = torch.load("./model/best_model_02.model", map_location=torch.device('cpu'))
history = content['history']

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

# 添加总标题
plt.suptitle('Fine-tuning on ViT Baseline model - 60 epochs', fontsize=16)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
