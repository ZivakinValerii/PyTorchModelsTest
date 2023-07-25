from ConvAutoencoder import ConvAutoencoder, ConvAutoencoderNoPool

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.to(device)
    mean = 0.5
    sd = 0.5
    autoencoder = ConvAutoencoderNoPool()
    autoencoder.load_state_dict(torch.load('models/conv_autoencoder_no_pool_with_cross.pth'))
    autoencoder.to(device)
    # Загрузка набора данных и преобразование его в формат, совместимый с моделью
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[mean]*3, std=[sd]*3)
    ])
    dataset = datasets.ImageFolder('D:/MyPy/MyFirstAutioncoderPro/TRAIN', transform=transform)
    base_folder = 'results/Decoded/Encoder old dataset no pool with normalize and cross'
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)

    # Прогон набора данных через модель, получение латентных представлений
    latent_vectors = []
    autoencoder.eval()
    with torch.no_grad():
        for batch, _ in dataloader:
            latent = autoencoder.forward(batch.to(device))
            latent_vectors.append(latent)
    latent_vectors = torch.cat(latent_vectors, dim=0)

    # Преобразование латентных представлений обратно в изображения и сохранение их
    transform = transforms.Compose([
        transforms.Normalize(mean=[-mean/sd] * 3, std=[1.0/sd] * 3),
        transforms.ToPILImage(),
    ])



    for i in range(len(dataset)):
        img_path, _ = dataset.samples[i]
        folder_path = os.path.join(base_folder, os.path.basename(os.path.dirname(img_path)))
        filename = os.path.basename(img_path)
        latent = latent_vectors[i].unsqueeze(0)
        #reconstructed = autoencoder.decode(latent)
        reconstructed = transform(latent[0])#reconstructed.squeeze(0))
        save_path = os.path.join(folder_path, filename)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        reconstructed.save(save_path)