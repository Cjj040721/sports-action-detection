import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [B, C]
        w = self.se(x)  # [B, C]
        return x * w  # 按通道缩放


class DualEfficientNetB0(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()

        # 加载两个EfficientNet-B0（带ImageNet预训练）
        base_rtm = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        base_dtm = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # 提取特征主干部分（不含分类头）
        self.branch_rtm = nn.Sequential(*list(base_rtm.features), nn.AdaptiveAvgPool2d(1))
        self.branch_dtm = nn.Sequential(*list(base_dtm.features), nn.AdaptiveAvgPool2d(1))

        # 注意力机制
        self.se_block = SEBlock(1280 * 2, reduction=16)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(1280 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes // 2)
        )

        # 每个类别一个二分类头
        self.binary_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1280 * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 2)  # 输出为 2 分类
            ) for _ in range(num_classes // 2)
        ])

    def forward(self, rtm, dtm):
        r_feat = self.branch_rtm(rtm)  # [B, 1280, 1, 1]
        d_feat = self.branch_dtm(dtm)  # [B, 1280, 1, 1]
        r_feat = r_feat.view(r_feat.size(0), -1)  # [B, 1280]
        d_feat = d_feat.view(d_feat.size(0), -1)  # [B, 1280]
        fused = torch.cat([r_feat, d_feat], dim=1)  # [B, 2560]
        fused = self.se_block(fused)

        main_logits = self.classifier(fused)  # [B, 10]
        binary_logits = [head(fused) for head in self.binary_heads]  # list of [B, 2]
        binary_logits = torch.stack(binary_logits, dim=1)  # [B, 10, 2]

        final_logits = []
        for i in range(main_logits.shape[1]):  # 10类
            cls_score = main_logits[:, i].unsqueeze(1)  # [B, 1]
            bin_score = binary_logits[:, i, :]  # [B, 2]
            combined = cls_score + bin_score  # [B, 2]
            final_logits.append(combined)  # list of [B, 2]

        final_logits = torch.cat(final_logits, dim=1)  # [B, 20]
        return final_logits
