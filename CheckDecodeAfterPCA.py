from ConvAutoencoder import ConvAutoencoder, ConvAutoencoderNoPool

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.decomposition import PCA


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.to(device)
    autoencoder = ConvAutoencoderNoPool()
    autoencoder.load_state_dict(torch.load('models/conv_autoencoder_no_pool.pth'))
    autoencoder.to(device)
    base_folder = 'Encoder without pool after PCA'
    # Загрузка набора данных и преобразование его в формат, совместимый с моделью
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(base_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)

    # Прогон набора данных через модель, получение латентных представлений
    latent_vectors = []
    autoencoder.eval()
    with torch.no_grad():
        for batch, _ in dataloader:
            latent = autoencoder.encode(batch.to(device))
            latent_vectors.append(latent)
    latent_vectors = torch.cat(latent_vectors, dim=0)

    sizes = latent_vectors.size()

    pca = PCA(n_components=500)
    latent_vectors = latent_vectors.reshape(latent_vectors.size(0), -1)
    PCA_latent_vectors = pca.fit_transform(latent_vectors.cpu().numpy())
    print(sum(pca.explained_variance_ratio_))
    latent_vectors = pca.inverse_transform(PCA_latent_vectors)
    latent_vectors = torch.from_numpy(latent_vectors).reshape(sizes[0], sizes[1], sizes[2], sizes[3]).to(device)
    #decoded_vectors = autoencoder.decode(latent_vectors.to(device))
    # Преобразование латентных представлений обратно в изображения и сохранение их
    transform = transforms.Compose([
        transforms.ToPILImage(),
    ])



    for i in range(len(dataset)):
        img_path, _ = dataset.samples[i]
        folder_path = os.path.join(base_folder, os.path.basename(os.path.dirname(img_path)))
        filename = os.path.basename(img_path)
        latent = latent_vectors[i].unsqueeze(0)
        reconstructed = autoencoder.decode(latent)
        reconstructed = transform(reconstructed.squeeze(0))
        save_path = os.path.join(folder_path, filename)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        reconstructed.save(save_path)