import os
import torch
import threading

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
    save_content = {'obj': obj, 'history': history}
    torch.save(save_content, save_path)

def load_model(path, map_location=None):
    """
    从文件加载模型与训练记录
    @param path 文件保存路径
    @return 模型参数与训练记录
    """
    load_content = torch.load(path, map_location=map_location, weights_only=False)
    return load_content['obj'], load_content['history']

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
