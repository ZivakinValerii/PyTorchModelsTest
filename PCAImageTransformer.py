import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torchsummary import summary
from sklearn.decomposition import PCA

from ConvAutoencoder import ConvAutoencoderNoPool
from PCAFilesGenerator import get_data_loader


def load_model(model_path: str, device):
    autoencoder = ConvAutoencoderNoPool()
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.to(device)

    autoencoder.eval()

    return autoencoder

def save_images_from_PCA(class_for_folder: str,save_dir: str, encoder_file: str, pca_file: str, model, device, mean=0.0, std=1.0):
    if os.path.exists(pca_file):
        x = torch.randn(1, 3, 64, 64)  # создаем случайный тензор размером 1x3x224x224

        output = model.encoder(x.to(device))  # прогоняем тензор через энкодер модели
        output = output.reshape(output.size(0), -1)
        cols = output.shape[1]  # получаем размерность выходного тензора

        encoder_list = np.loadtxt(encoder_file, usecols=range(cols))
        pca = PCA(n_components=1000)
        pca.fit(encoder_list)
        pca_list = np.loadtxt(pca_file, usecols=range(pca.n_components))
        invers_transform = pca.inverse_transform(pca_list)

        data_tensor = torch.from_numpy(invers_transform.reshape(invers_transform.shape[0], 256, 2, 2)).to(device).float()
        transform = transforms.Compose([
            transforms.Normalize(mean=[-mean/std]*3, std=[1.0/std]*3), transforms.ToPILImage(),
        ])
        save_dir = os.path.join(save_dir, class_for_folder) #os.path.basename(pca_file).split('-')[0]
        for i in range(len(data_tensor)):
            latent = data_tensor[i].unsqueeze(0)
            reconstructed = model.decode(latent)
            reconstructed = transform(reconstructed.squeeze(0))
            save_path = os.path.join(save_dir, f"{i+1}.jpg")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            reconstructed.save(save_path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Model loading")
    data_loader = get_data_loader('D:\\MyPy\\DataSets\\NAU_DATASET_last_PCA', 0.5, 0.5)
    classes = data_loader.dataset.classes
    model = load_model('models_for_duplicates/conv_autoencoder_no_pool_with_cross_NAU_DATASET_last_all64.pth', device)
    summary(model, input_size=(3, 64, 64))

    for class_name in classes:
        save_images_from_PCA(class_for_folder=class_name, save_dir="results/Decoded/Imitated_NAUSET_1000PCA",
                         encoder_file=f"results/Files/NAULASTSET_1000PCA/{class_name}-ENC.txt",
                         #encoder_file=f"results/Files/Encoder.txt",
                         pca_file=f"D:/VS/Modeling/Modeling/bin/Debug/was_modeled_{class_name}-PCA_by2DSpline.txt",
                         model=model, device=device, mean=0.5, std=0.5)
