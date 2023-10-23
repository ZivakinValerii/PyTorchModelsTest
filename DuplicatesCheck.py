import os
import shutil
import torch

from sklearn.metrics.pairwise import cosine_similarity

from ConvAutoencoder import ConvAutoencoder, ConvAutoencoderNoPool, train_coder, train_coder_with_cross, \
    ConvAutoencoder128, ConvAutoencoder256

from torchvision import datasets, transforms
from torchsummary import summary
from sklearn.decomposition import PCA


def get_data_loader(data_path: str, mean=0.0, std=1.0):
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ToTensor(), transforms.Normalize(mean=[mean] * 3, std=[std] * 3)
                                    ])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    return dataloader


def sort_images(source_folder: str, target_folder: str, size_type, device):
    # Получите список классов на основе подпапок в исходной папке
    class_names = os.listdir(source_folder)

    # Создайте папки классов в целевой папке
    for class_name in class_names:
        class_folder_path = os.path.join(target_folder, class_name)
        os.makedirs(class_folder_path, exist_ok=True)

    data_loader = get_data_loader(source_folder, 0.5, 0.5)

    model = ConvAutoencoderNoPool()
    model.load_state_dict(state_dict=torch.load(
                    f'models_for_duplicates/conv_autoencoder_no_pool_with_cross_NAU_DATASET_last_all{size_type[0]}.pth'))
    model.to(device)
    model.eval()

    latent_vectors = []
    img_paths = []
    labels = []
    classes = data_loader.dataset.classes
    with torch.no_grad():
        for i, (batch, label) in enumerate(data_loader):
            latent = model.encode(batch.to(device))
            latent_vectors.append(latent)
            labels.append(label)
            # Получите индексы изображений в текущем батче
            start_idx = i * data_loader.batch_size
            end_idx = (i + 1) * data_loader.batch_size
            if end_idx >= len(data_loader.dataset):
                end_idx = len(data_loader.dataset)
            batch_indices = list(range(start_idx, end_idx))

            # Получите пути к файлам изображений на основе индексов
            batch_paths = [data_loader.dataset.samples[idx][0] for idx in batch_indices]
            img_paths.extend(batch_paths)  # Добавьте пути к файлам в список

    latent_vectors = torch.cat(latent_vectors, dim=0)
    latent_vectors = latent_vectors.reshape(latent_vectors.size(0), -1)
    labels = torch.cat(labels, dim=0)

    latent_vectors_cpu = latent_vectors.cpu().numpy()
    labels_cpu = labels.cpu().numpy()

    latent_by_class = {}
    img_paths_by_class = {}
    for i, label in enumerate(labels_cpu):
        if label not in latent_by_class:
            latent_by_class[label] = []
            img_paths_by_class[label] = []
        latent_by_class[label].append(latent_vectors_cpu[i])
        img_paths_by_class[label].append(img_paths[i])

    for label, latent_list in latent_by_class.items():
        similarity_matrix = cosine_similarity(latent_list)
        threshold = 0.97
        non_duplicate_indices = []
        for i in range(len(similarity_matrix)):
            is_duplicate = False
            for j in range(i + 1, len(similarity_matrix)):
                # Проверка на дубликаты
                if similarity_matrix[i, j] > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                non_duplicate_indices.append(i)
        for index in non_duplicate_indices:
            new_file_path = os.path.join(target_folder, classes[label],
                                         os.path.basename(img_paths_by_class[label][index]))
            shutil.copy(img_paths_by_class[label][index], new_file_path)




        #images_path = os.path.join(class_source_path, f'{size_type[0]}x{size_type[1]}')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = 0.5
    sd = 0.5
    size_types = [(64, 64), (128, 128), (256, 256)]
    sort_images("D:\\MyPy\\DataSets\\NAU_DATASET_last", target_folder="D:\\MyPy\\DataSets\\NAU_DATASET_SORTED_NOD_last_tr097",
                device=device, size_type=(64, 64),)
