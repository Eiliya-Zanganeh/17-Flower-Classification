from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms


def generate_dataset():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.ImageFolder('dataset/train', transform=transform)
    test_dataset = datasets.ImageFolder('dataset/test', transform=transform)

    train_size = round(.9 * len(train_dataset))
    validation_size = len(train_dataset) - train_size

    train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

    train_dataset = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = DataLoader(test_dataset, batch_size=32, shuffle=False)
    validation_dataset = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    return train_dataset, test_dataset, validation_dataset
