from ConvAutoencoder import ConvAutoencoderNoPool, ModifiedClassifier, AutoencoderClassifier, train_AutoClass

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = 0.5
    sd = 0.5
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[mean]*3, std=[sd]*3)])
    train_dataset = datasets.ImageFolder(root='D:\\MyPy\\MyPyTorchProj\\results\\Decoded\\Imitated_NAUSET', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    autoencoder = ConvAutoencoderNoPool()
    autoencoder.load_state_dict(torch.load('models_for_duplicates/conv_autoencoder_no_pool_with_cross_NAU_DATASET_last_all64.pth'))
    autoencoder.to(device)
    classifier = ModifiedClassifier(in_features=1024, num_classes=34)
    classifier.to(device)
    criterion_cls = nn.CrossEntropyLoss()
    optimizer_cls = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    criterion_ae = nn.MSELoss()
    optimizer_ae = optim.Adam(autoencoder.parameters(), lr=0.001)

    num_ep = 120
    model, train_losses = train_AutoClass(autoencoder, classifier, train_loader, criterion_cls, optimizer_cls,
                                          criterion_ae, optimizer_ae, device, num_ep)

    torch.save({
        'state_dict': model.state_dict(),
        'architecture': model
    }, 'models_for_duplicates/AutoencoderClassifier_NAU_DATASET_true.pt')

    plt.plot(train_losses, label='Train Loss')
    plt.legend()
    plt.show()
