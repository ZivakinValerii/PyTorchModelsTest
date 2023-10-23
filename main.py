from cProfile import label
from CustomDataSet import CustomDataset
from ConvAutoencoder import ConvAutoencoder, ConvAutoencoderNoPool, train_coder, train_coder_with_cross, \
    ConvAutoencoder128, ConvAutoencoder256

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
    size_types = [(64, 64)]
    for size_type in size_types:
        transform = transforms.Compose([transforms.Resize(size_type), transforms.ToTensor(),
                                    transforms.Normalize(mean=[mean, mean, mean],
                                                         std=[sd, sd, sd])])
        #train_dataset = datasets.ImageFolder('D:/MyPy/MyFirstAutioncoderPro/TRAINN', transform=transform)
        train_dataset = datasets.ImageFolder('D:/MyPy/DataSets/NAU_DATASET_last', transform=transform)
        #train_dataset = CustomDataset("D:/MyPy/DataSets/NAU_DATASET_SORTED", size_type, transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        model = None
        if size_type == (64, 64):
            model = ConvAutoencoderNoPool()
        elif size_type == (128, 128):
            model = ConvAutoencoder128()
        else:
            model = ConvAutoencoder256()
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_ep = 5
        train_losses, val_losses = train_coder_with_cross(model, train_loader, criterion, optimizer, num_ep, device)

        torch.save(model.state_dict(), f'models_for_duplicates/conv_autoencoder_no_pool_with_cross_NAU_DATASET_last_all{size_type[0]}.pth')

    #plt.plot(train_losses, label='Train Loss')
    #plt.plot(val_losses, label='Val Loss')
    #plt.legend()
    #plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
