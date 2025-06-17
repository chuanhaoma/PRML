from torch import nn
import torchvision

# 模型构建

# 带预训练权重的Vision Transformer B-16
class ViTForPollenClassification(nn.Module):
    def __init__(self, all_weight=False, num_classes=73):
        super().__init__()
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        self.vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

        for parameter in self.vit.parameters(): # 冻结参数
            parameter.requires_grad = all_weight

        self.vit.heads = nn.Linear(in_features=self.vit.hidden_dim, out_features=num_classes)
    
    def forward(self, x):
        return self.vit(x)
    
# 带预训练权重的Vision Transformer L-16
class ViTLForPollenClassification(nn.Module):
    def __init__(self, all_weight=False, num_classes=73):
        super().__init__()
        pretrained_vit_weights = torchvision.models.ViT_L_16_Weights.DEFAULT
        self.vit = torchvision.models.vit_l_16(weights=pretrained_vit_weights)

        for parameter in self.vit.parameters(): # 冻结参数
            parameter.requires_grad = all_weight

        self.vit.heads = nn.Linear(in_features=self.vit.hidden_dim, out_features=num_classes)
    
    def forward(self, x):
        return self.vit(x)

# 带预训练权重的Vision Transformer H-14
class ViTHForPollenClassification(nn.Module):
    def __init__(self, all_weight=False, num_classes=73):
        super().__init__()
        pretrained_vit_weights = torchvision.models.ViT_H_14_Weights.DEFAULT
        self.vit = torchvision.models.vit_h_14(weights=pretrained_vit_weights)

        for parameter in self.vit.parameters(): # 冻结参数
            parameter.requires_grad = all_weight

        self.vit.heads = nn.Linear(in_features=self.vit.hidden_dim, out_features=num_classes)
    
    def forward(self, x):
        return self.vit(x)

# 带预训练权重的ResNet-50
class ResNet50ForPollenClassification(nn.Module):
    def __init__(self, all_weight=False, num_classes=73):
        super().__init__()
        pretrained_res_weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.res = torchvision.models.resnet50(weights=pretrained_res_weights)

        for parameter in self.res.parameters(): # 冻结参数
            parameter.requires_grad = all_weight
        self.res.fc = nn.Linear(in_features=self.res.fc.in_features, out_features=num_classes)
    
    def forward(self, x):
        return self.res(x)

# 带预训练权重的DenseNet-201
class DenseNet201ForPollenClassification(nn.Module):
    def __init__(self, all_weight=False, num_classes=73):
        super().__init__()
        pretrained_weights = torchvision.models.DenseNet201_Weights.DEFAULT
        self.dense = torchvision.models.densenet201(weights=pretrained_weights)

        for parameter in self.dense.parameters(): # 冻结参数
            parameter.requires_grad = all_weight

        self.dense.classifier = nn.Linear(in_features=self.dense.classifier.in_features, out_features=num_classes)
    
    def forward(self, x):
        return self.dense(x)
