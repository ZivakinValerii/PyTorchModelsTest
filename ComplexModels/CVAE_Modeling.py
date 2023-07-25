import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchsummary import summary
from sklearn.decomposition import PCA

from CVAE import ConvConditionalVAE
from PCAFilesGenerator import get_data_loader


def load_model(model_path: str, device):
    state = torch.load(model_path)
    model = state['architecture']
    model.load_state_dict(state['state_dict'])
    model.to(device)

    model.eval()

    return model


def generate_images_from_ENC(save_dir: str, encoder_file: str, model, device, components_count, class_index,
                             num_classes):
    # x = torch.randn(1, 3, 64, 64)  # создаем случайный тензор размером 1x3x224x224
    #
    # output = model.encoder(x.to(device))  # прогоняем тензор через энкодер модели
    # output = output.reshape(output.size(0), -1)
    # cols = output.shape[1]  # получаем размерность выходного тензора

    encoder_list = np.loadtxt(encoder_file, usecols=range(1024))
    pca = PCA(n_components=components_count)
    pca.fit(encoder_list)
    transformed = pca.transform(encoder_list)

    data_tensor = torch.from_numpy(transformed.reshape(transformed.shape[0], 256)).to(device).float()
    #data_tensor = torch.randn(transformed.shape[0], components_count).to(device)
    transform = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1.0 / 0.224, 1.0 / 0.229, 1.0 / 0.225]),
        transforms.ToPILImage(),
    ])

    labels_tensor = torch.from_numpy(np.array([class_index] * data_tensor.shape[0])).to(device).long()
    one_hot_labels = F.one_hot(labels_tensor, num_classes).float()

    data_tensor = torch.cat((data_tensor, one_hot_labels), dim=1)

    save_dir = os.path.join(save_dir, os.path.basename(encoder_file).split('-')[0])
    for i in range(len(data_tensor)):
        latent = data_tensor[i].unsqueeze(0)
        reconstructed = model.decode(latent)
        reconstructed = transform(reconstructed.squeeze(0))
        save_path = os.path.join(save_dir, f"{i + 1}.jpg")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        reconstructed.save(save_path)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("../models/ConvConditionalVAENoPool_with_weighed_classes.pt", device)
    dataset = datasets.ImageFolder('D:/MyPy/MyFirstAutioncoderPro/TRAINN')
    classes = dataset.classes
    for class_name in classes:
        generate_images_from_ENC(save_dir="../results/Decoded/Imitated_FromCVAE2",
                                 encoder_file=f"../results/Files/Satellite_Dataset/{class_name}-ENC.txt",
                                 model=model, device=device, components_count=256, num_classes=len(classes),
                                 class_index=classes.index(class_name))
