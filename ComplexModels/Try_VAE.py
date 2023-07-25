from cProfile import label

from VAE import VariationConvAutoencoderNoPool, train_vae_with_cross, simple_train
from VAE_FullGPT import ConvVAE, train_vae
#from CVAE import ConvConditionalVAE, train_conditional_vae
from CVAE_with_weighted_classes import ConvConditionalVAE, train_conditional_vae

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
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # mean = 0.5
    # sd = 0.5
    #
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[mean, mean, mean],
    #                                                                             std=[sd, sd, sd])])
    # train_dataset = datasets.ImageFolder('D:/MyPy/MyFirstAutioncoderPro/TRAINN', transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    #
    # model = VariationConvAutoencoderNoPool(1024)
    # model.to(device)
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    # num_ep = 30
    # #train_losses, val_losses = train_vae_with_cross(model, train_loader, criterion, optimizer, num_ep, device)
    # train_losses = simple_train(model, train_loader, criterion, optimizer, num_ep, device)
    # #torch.save(model.state_dict(), 'models/conv_autoencoder_no_pool_with_cross.pth')
####################################################################################################################
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # Пример использования
    # latent_size = 256
    # num_epochs = 50
    # learning_rate = 0.001
    # batch_size = 64
    #
    # # Создание экземпляра модели
    # model = ConvVAE(latent_size)
    #
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                                             std=[0.229, 0.224, 0.225])])
    # train_dataset = datasets.ImageFolder('D:/MyPy/MyFirstAutioncoderPro/TRAINN', transform=transform)
    #
    # dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # model.to(device)
    # # Обучение модели
    # train_vae(model, dataloader, num_epochs, learning_rate, device)
#####################################################################################################################
    # Пример использования
    latent_size = 256
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 64
    num_classes = 10

    # Создание экземпляра модели
    model = ConvConditionalVAE(latent_size, num_classes)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])
    train_dataset = datasets.ImageFolder('D:/MyPy/MyFirstAutioncoderPro/TRAINN', transform=transform)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_conditional_vae(model, dataloader, num_epochs, learning_rate)

    torch.save({
        'state_dict': model.state_dict(),
        'architecture': model
    }, '    ConvConditionalVAENoPool_with_weighed_classes_2.pt')

    # plt.plot(train_losses, label='Train Loss')
    # # plt.plot(val_losses, label='Val Loss')
    # plt.legend()
    # plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
