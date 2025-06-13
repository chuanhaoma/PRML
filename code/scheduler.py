import math
from torch.optim.lr_scheduler import LRScheduler

class CosineDecayWithLinearWarmup(LRScheduler):

    def __init__(self, optimizer, warmup_steps, cosine_steps, min_lr, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer')}
        state_dict['warmup_steps'] = self.warmup_steps
        state_dict['cosine_steps'] = self.cosine_steps
        state_dict['min_lr'] = self.min_lr

        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self.warmup_steps = state_dict['warmup_steps']
        self.cosine_steps = state_dict['cosine_steps']
        self.min_lr = state_dict['min_lr']
    
    def lr_lambda(self, current_step, base_lr):
        # warmup阶段 线性增加学习率
        if current_step < self.warmup_steps:
            return base_lr * float(current_step) / float(max(1, self.warmup_steps))
        
        # 余弦退火阶段
        progress = float(current_step - self.warmup_steps) / float(max(1, self.cosine_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        lr_factor = (1 - self.min_lr / base_lr) * cosine_decay + self.min_lr / base_lr
        # 确保学习率不低于min_lr
        lr = max(self.min_lr, lr_factor * base_lr)
        
        return lr

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            lrs.append(self.lr_lambda(self.last_epoch + 1, base_lr))
        return lrs
