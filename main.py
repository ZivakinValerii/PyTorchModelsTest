from cProfile import label

from ConvAutoencoder import ConvAutoencoder, ConvAutoencoderNoPool, train_coder, train_coder_with_cross

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = 0.5
    sd = 0.5

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[mean, mean, mean],
                                                         std=[sd, sd, sd])])
    #train_dataset = datasets.ImageFolder('D:/MyPy/MyFirstAutioncoderPro/TRAINN', transform=transform)
    train_dataset = datasets.ImageFolder('D:/MyPy/DataSets/NAU_DATASET', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = ConvAutoencoderNoPool()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_ep = 5
    train_losses, val_losses = train_coder_with_cross(model, train_loader, criterion, optimizer, num_ep, device)

    torch.save(model.state_dict(), 'models/conv_autoencoder_no_pool_with_cross_NAU_DATASET.pth')

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
