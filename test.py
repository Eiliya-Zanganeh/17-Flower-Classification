from torchvision import models

densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

for name, layer in densenet.named_children():
    print(f'Layer name: {name}, Layer: {layer}')