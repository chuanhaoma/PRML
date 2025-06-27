import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DEFAULT_DATASET_DIR = './dataset/'

# POLLEN73S数据集
class POLLEN73S(Dataset):
    def read_image(self, path):
        image = cv2.imread(path)
        image = image[:, :, ::-1]
        image = image.transpose((2, 0, 1))
        return image
    
    def load_data(self, dir : str):
        "Load dataset from specific dir"
        class_names = self.get_label_mapping() # 得到数据集下所有分类
        for [idx, name] in enumerate(class_names): # 打开各个目录
            class_path = os.path.join(dir, name) # 获取类别文件夹路径
            file_names = os.listdir(class_path) # 获取类别下数据文件名
            file_names.sort() # 排序 防止顺序错乱打乱fold
            class_files = [os.path.join(class_path, item) for item in file_names] # 获取类别下数据文件名

            if self.pre_read: # 开启预读取
                images = [self.read_image(file) for file in class_files] # 一次读取一批图像
                self.data.extend(images) # 数据存入内存
            
            self.path.extend(class_files)
            self.labels.extend([idx] * len(class_files))

    def get_label_mapping(self):
        "Get the mapping array from idx to label string"
        class_names = ["qualea_multiflora","archontophoenix_cunninghamiana","caesalpinia_peltophoroides","bacopa_australis","chromolaena_laevigata","arrabidaea_florida","myracroduon_urundeuva","eucalyptus_sp","serjania_sp","aspilia_grazielae","schizolobium_parahyba","anadenanthera_colubrina","senegalia_plumosa","schinus_sp","ceiba_speciosa","cordia_trichotoma","ouratea_hexasperma","sida_cerradoensis","serjania_erecta","ochroma pyramidale","matayba_guianensis","mimosa_ditans","machaerium_aculeatum","trembleya_phlogiformis","hortia_oreadica","ricinus_communis","tapirira_guianensis","erythrina_mulungu","doliocarpus_dentatus","luehea_divaricata","zea_mays","cecropia_pachystachya","manihot_esculenta","tabebuia_rosealba","dipteryx_alata","solanum_sisymbriifolium","syagrus_romanzoffiana","magnolia_champaca","genipa_auniricana","caryocar_brasiliensis","mimosa_pigra","arachis_sp","mabea_fistulifera","hyptis_sp","tradescantia_Pallida","ligustrum_lucidum","dianella_tasmanica","myrcia_guianensis","cissus_spinosa","symplocos_nitens","poaceae_sp","tridax_procumbens","combretum_discolor","gomphrena_sp","paullinia_spicata","acrocomia_aculeta","serjania_hebecarpa","vochysia_divergens","cosmos_caudatus","mauritia_flexuosa","tabebuia_chysotricha","cissus_campestris","brugmansia_suaveolens","trema_micrantha","passiflora_gibertii","serjania_laruotteana","mitostemma_brevifilis","croton_urucurana","pachira_aquatica","faramea_sp","protium_heptaphyllum","piper_aduncum","guazuma_ulmifolia"]
        return class_names
    
    def get_class_distribution(self):
        """Get the dict of the number of items in each class"""
        class_counts = {}
        class_names = self.get_label_mapping()
        labels_np = np.array(self.labels)
        
        for idx, name in enumerate(class_names):
            class_counts[name] = np.sum(labels_np == idx)
        return class_counts

    def __init__(self, *, transform = None, dir : str = DEFAULT_DATASET_DIR, pre_read : bool = False):
        "Initialize the POLLEN73S dataset"
        # 数据初始化
        self.pre_read = pre_read # 预读取
        self.transform = transform # 变换

        self.path = [] # 路径
        self.data = [] # 数据
        self.labels = [] # 标签

        self.load_data(dir) # 载入数据

    def __len__(self):
        "Get the length of the POLLEN73S dataset"
        # 返回数据集大小
        return len(self.labels)

    def __getitem__(self, idx):
        # 按索引返回数据和标签
        label = self.labels[idx]
        if self.pre_read:
            sample = torch.from_numpy(self.data[idx].copy())
        else:
            sample = torch.from_numpy(self.read_image(self.path[idx]).copy())
        
        if self.transform: # 涉及到变换
            sample = self.transform(sample) # 执行变换
        
        return sample, label

# 支持变换操作的子集
class TransformSubset(Dataset):
    '''A custom subset class with transform function'''
    def __init__(self, dataset, indices, transform=None):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.transfrom = transform

    def __getitem__(self, idx): #同时支持索引访问操作
        X, y = self.dataset[self.indices[idx]]
        if self.transfrom: # 若变换不为空
            X = self.transfrom(X)
        return X, y

    def __len__(self): # 同时支持取长度操作
        return len(self.indices)

def get_basic_transform():
    "Get basic image transform function"
    # 定义变换操作
    train_transform = transforms.Compose([
        # Resize只支持PIL格式的图片，所以首先需要转成PIL
        transforms.ToPILImage(mode='RGB'),
        # 随机尺寸变换 (暂时不启用)
        # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        # 随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 随机旋转角度
        transforms.RandomRotation(15),
        # 设置resize的图片尺寸
        transforms.Resize(size=(224, 224)),
        # 转为Tensor
        transforms.ToTensor(),
        # 标准化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        # Resize只支持PIL格式的图片，所以首先需要转成PIL
        transforms.ToPILImage(mode='RGB'),
        # 设置resize的图片尺寸
        transforms.Resize(size=(224, 224)),
        # 将图片转为tensor
        transforms.ToTensor(),
        # 归一化处理：[0-255] -> [0-1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, test_transform

def get_POLLEN73S_dataloader(*, batch_size : int = 32, train_transform = None, test_transform = None, ratio : float = 0.2, random_state : int = 42, num_workers : int = 0, pre_load : bool = False, dir : str = DEFAULT_DATASET_DIR):
    "Initialize the POLLEN73S dataset and Generate train set & test set dataloader"
    dataset = POLLEN73S(dir=dir, pre_read=pre_load)

    # 生成索引并分层划分
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=ratio,              # 测试集比例
        stratify=dataset.labels,      # 分层依据的标签
        random_state=random_state     # 随机种子确保可重复
    )

    # 创建训练集和测试集的Subset
    train_dataset = TransformSubset(dataset, train_indices, train_transform)
    test_dataset = TransformSubset(dataset, test_indices, test_transform)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    return train_loader, test_loader

def get_POLLEN73S_K_Fold(*, k : int = 5, batch_size : int = 32, train_transform = None, test_transform = None, shuffle : bool = False, random_state : int = 42, num_workers : int = 0, pre_load : bool = False, dir : str = DEFAULT_DATASET_DIR):
    "Generate train & test dataloaders for k-fold cross-validation"
    # 初始化数据集
    dataset = POLLEN73S(dir=dir, pre_read=pre_load)

    # 分层K折交叉验证数据集划分
    skf = StratifiedKFold(n_splits=k, random_state=(random_state if shuffle else None), shuffle=shuffle)

    # 划分结果
    train_loaders = [] # 训练集加载器
    test_loaders = [] # 测试集加载器

    # 数据集样本索引
    indices = list(range(len(dataset)))

    # K折迭代
    for train_indices, test_indices in skf.split(indices, dataset.labels):
        # 创建训练集和测试集的Subset
        train_dataset = TransformSubset(dataset, train_indices, train_transform)
        test_dataset = TransformSubset(dataset, test_indices, test_transform)

        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        
        # 添加进列表
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders

def visualize_class_distribution(dataset_dir=DEFAULT_DATASET_DIR, figsize=(20, 8), max_categories=None):
    """
    可视化数据集中的类别分布（垂直条形图）
    Args:
        dataset_dir: 数据集路径
        figsize: 图表尺寸（宽度,高度）
        max_categories: 最大显示类别数（None表示显示全部）
    """
    # 加载数据集
    dataset = POLLEN73S(dir=dataset_dir, pre_read=False)
    class_counts = dataset.get_class_distribution()
    class_names = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # 按样本数量排序
    sorted_indices = np.argsort(counts)[::-1]
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    
    # 如果指定最大显示类别数，截取前n个
    if max_categories is not None and max_categories < len(sorted_class_names):
        sorted_class_names = sorted_class_names[:max_categories]
        sorted_counts = sorted_counts[:max_categories]
    
    # 创建图表
    fig = plt.figure(figsize=figsize)
    bars = plt.bar(sorted_class_names, sorted_counts, color='skyblue')
    
    # 设置轴标签和标题
    plt.xlabel('Sample Class', fontsize=12)
    plt.ylabel('Sample Size', fontsize=12)
    plt.title('Class Distribution of POLLEN73S Dataset', fontsize=16, pad=20)
    
    # 设置Y轴为整数刻度
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 旋转X轴标签以避免重叠
    plt.xticks(rotation=45, ha='right', fontsize=8)
    
    # 添加数值标签在条形顶部
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                 height + max(sorted_counts)*0.01, 
                 f'{int(height)}', 
                 ha='center', va='bottom', fontsize=8)
    
    # 调整布局
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 保存图表为PNG格式，设置透明背景
    fig.patch.set_alpha(0.0)
    plt.savefig(os.path.join(dataset_dir,'dataset_distribution.png'), transparent=True, dpi=300)
    
    # 显示图表
    # plt.show()
    
    # 打印统计信息
    total_samples = sum(counts)
    print(f"数据集总样本数: {total_samples}")
    print(f"类别数量: {len(class_names)}")
    print(f"平均每类样本数: {total_samples/len(class_names):.1f}")
    print(f"样本最多的类别: {sorted_class_names[0]} ({sorted_counts[0]}个)")
    print(f"样本最少的类别: {sorted_class_names[-1]} ({sorted_counts[-1]}个)")
