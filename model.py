from torch import nn
from torchvision import models


class FlowerModel(nn.Module):
    def __init__(self, num_classes):
        super(FlowerModel, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        self.densenet.classifier = nn.Linear(in_features=1024, out_features=512, bias=True)

        for param in self.densenet.parameters():
            param.requires_grad = False

        self.densenet.classifier.requires_grad = True

        self.dropout = nn.Dropout(p=.2)

        self.classifier = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.densenet(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
