from torch import nn
import torchvision

# 模型构建
class ViTForPollenClassification(nn.Module):
    def __init__(self, all_weight=False, num_classes=73):
        super().__init__()
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        self.vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

        for parameter in self.vit.parameters(): # 冻结参数
            parameter.requires_grad = all_weight

        self.vit.heads = nn.Linear(in_features=768, out_features=num_classes)
    
    def forward(self, x):
        return self.vit(x)

if False:
    from torchinfo import summary
    vit = ViTForPollenClassification()
    summary(model=vit,
            input_size=(16, 3, 224, 224), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )
