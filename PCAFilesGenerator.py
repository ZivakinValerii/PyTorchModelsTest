import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torchsummary import summary
from sklearn.decomposition import PCA

from ConvAutoencoder import ConvAutoencoderNoPool


def load_model(model_path: str, device):
    autoencoder = ConvAutoencoderNoPool()
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.to(device)

    autoencoder.eval()

    return autoencoder


def get_data_loader(data_path: str, mean=0.0, std=1.0):
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ToTensor(), transforms.Normalize(mean=[mean] * 3, std=[std] * 3)
                                    ])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    return dataloader


def get_generals(data_tensor, components_count):
    matrix = data_tensor.reshape(data_tensor.size(0), -1)
    np.savetxt('Encoder.txt', matrix.detach().cpu().numpy())
    pca = PCA(n_components=components_count)
    pca.fit(matrix.detach().cpu().numpy())
    pca_transform = pca.transform(matrix.detach().cpu().numpy())
    np.savetxt('PCA.txt', pca_transform)
    return pca_transform


def save_laltents(folder_path: str, classes_names, vectors_dict, pca_dict, components_count):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    for key in vectors_dict:
        # file = open(os.path.join(folder_path, f'{classes_names[key]}-ENC.txt'), 'w')
        filename = os.path.join(folder_path, f'{classes_names[key]}-ENC.txt')
        np.savetxt(filename, class_latent_dict[key])
        filename = os.path.join(folder_path, f'{classes_names[key]}-GeneralPCA.txt')
        np.savetxt(filename, pca_dict[key])

        pca = PCA(n_components=components_count)
        pca.fit(vectors_dict[key])
        transformed_vectors = pca.transform(vectors_dict[key])

        with open(os.path.join(folder_path, f'{classes_names[key]}-PCA.txt'), "w") as f:
            for vec in transformed_vectors:
                f.write(" ".join(map(str, vec.tolist())) + "\n")


if __name__ == '__main__':
    cc = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Model loading")
    #model = load_model('models/conv_autoencoder_no_pool_with_cross_NAU_DATASET.pth', device)
    model = load_model('models_for_duplicates/conv_autoencoder_no_pool_with_cross_NAU_DATASET_last_all64.pth', device)
    summary(model, input_size=(3, 64, 64))

    print("Data loading")
    data_loader = get_data_loader('D:\\MyPy\\DataSets\\NAU_DATASET_last_PCA', 0.5, 0.5)
    #data_loader = get_data_loader('D:/MyPy/DataSets/NAU_DATASET', 0.5, 0.5)
    print("Data loaded")

    latent_vectors = []
    labels = []
    classes = data_loader.dataset.classes
    with torch.no_grad():
        for i, (batch, label) in enumerate(data_loader):
            latent = model.encode(batch.to(device))
            latent_vectors.append(latent)
            labels.append(label)

    latent_vectors = torch.cat(latent_vectors, dim=0)
    latent_vectors = latent_vectors.reshape(latent_vectors.size(0), -1)
    labels = torch.cat(labels, dim=0)

    pca_view = get_generals(latent_vectors, cc)

    class_latent_dict = {}
    class_pca_dict = {}
    for i in range(len(labels)):
        label = labels[i].item()
        if label not in class_latent_dict:
            class_latent_dict[label] = [latent_vectors[i].detach().cpu().numpy()]
            class_pca_dict[label] = [pca_view[i]]
        else:
            class_latent_dict[label].append(latent_vectors[i].detach().cpu().numpy())
            class_pca_dict[label].append(pca_view[i])

    save_laltents('results/Files/NAULASTSET_1000PCA/', classes_names=classes, vectors_dict=class_latent_dict, pca_dict=class_pca_dict,
                  components_count=cc)
