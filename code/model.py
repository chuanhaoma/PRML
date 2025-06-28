import torch
from torch import nn
import torchvision

# 模型构建

# 带预训练权重的Vision Transformer B-16
class ViTForPollenClassification(nn.Module):
    def __init__(self, all_weight=False, num_classes=73):
        super().__init__()
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        self.vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

        for block in self.vit.encoder.layers: # 开启自注意力
            block.return_attention = True

        for parameter in self.vit.parameters(): # 冻结参数
            parameter.requires_grad = all_weight

        self.vit.heads = nn.Linear(in_features=self.vit.hidden_dim, out_features=num_classes)
    
    def forward(self, x):
        return self.vit(x)

# 获取融合注意力
class ViTWithCrossAttnForPollenClassification(nn.Module):
    def __init__(self, all_weight=False, num_classes=73):
        super().__init__()
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        self.vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

        for block in self.vit.encoder.layers: # 开启自注意力
            block.return_attention = True

        for parameter in self.vit.parameters(): # 冻结参数
            parameter.requires_grad = all_weight

        self.vit.heads = nn.Linear(in_features=self.vit.hidden_dim, out_features=num_classes)
    
    def forward(self, x):
        x = self.vit._process_input(x)

        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)

        x = torch.cat([batch_class_token, x], dim=1)

        attn_maps = []  # 存储各层注意力图
        for blk in self.vit.encoder.layers:
            blk.return_attention = True
            x, attn = blk(x)
            attn_maps.append(attn)
        
        # 融合最后3层注意力图（加权平均）
        fused_attn = 0.4*attn_maps[-1] + 0.3*attn_maps[-2] + 0.3*attn_maps[-3]
        return self.vit.heads(x), fused_attn  # 返回分类结果和融合注意力

# 带预训练权重的Vision Transformer L-16
class ViTLForPollenClassification(nn.Module):
    def __init__(self, all_weight=False, num_classes=73):
        super().__init__()
        pretrained_vit_weights = torchvision.models.ViT_L_16_Weights.DEFAULT
        self.vit = torchvision.models.vit_l_16(weights=pretrained_vit_weights)

        for block in self.vit.encoder.layers: # 开启自注意力
            block.return_attention = True
        
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

        for block in self.vit.encoder.layers: # 开启自注意力
            block.return_attention = True
        
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
