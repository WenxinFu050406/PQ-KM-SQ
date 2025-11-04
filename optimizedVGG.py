import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg19, vgg11


# -------------------- 配置参数 --------------------
BATCH_SIZE = 256
NUM_EPOCHS = 100
NUM_CLASSES = 10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
FREEZE_LAYERS = 10  # 冻结前10层卷积层


# -------------------- 优化的VGG模型 --------------------
class OptimizedVGG(nn.Module):
    def __init__(self, model_type):
        super().__init__()
        if model_type == 'vgg11':
            vgg = vgg11(pretrained=True)
        elif model_type == 'vgg16':
            vgg = vgg16(pretrained=True)
        elif model_type == 'vgg19':
            vgg = vgg19(pretrained=True)
        else:
            raise ValueError("Unsupported model type. Choose from 'vgg11', 'vgg16', or 'vgg19'.")

        # 冻结指定层
        for idx, layer in enumerate(vgg.features.children()):
            if idx < FREEZE_LAYERS:
                for param in layer.parameters():
                    param.requires_grad = False

        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 优化分类头
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, NUM_CLASSES)
        )

        # 初始化分类头权重
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



