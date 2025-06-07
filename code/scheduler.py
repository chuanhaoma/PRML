import math
from torch.optim import lr_scheduler

def warmup_cosine_scheduler(optimizer, warmup_steps, cosine_steps, min_lr):
    """带warmup的余弦退火调度器"""
    
    def lr_lambda(current_step):
        # warmup阶段 线性增加学习率
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # 余弦退火阶段
        progress = float(current_step - warmup_steps) / float(max(1, cosine_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        # 确保学习率不低于min_lr
        decay_factor = (1 - min_lr / optimizer.defaults["lr"]) * cosine_decay + min_lr / optimizer.defaults["lr"]
        
        return decay_factor

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)
