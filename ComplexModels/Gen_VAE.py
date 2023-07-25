from VAE import VariationConvAutoencoderNoPool, train_vae_with_cross

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


def generate_images(vae_model, num_images, device):
    # Генерируем случайные значения в латентном пространстве
    latent_vectors = torch.randn(num_images, 1024).to(device)

    # Передаем векторы в декодер модели
    generated_images = vae_model.decode(latent_vectors)

    # Преобразуем тензоры изображений в изображения
    generated_images = generated_images.cpu().detach()

    # Возвращаем сгенерированные изображения
    return generated_images


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load('VariationConvAutoencoderNoPool_with_Normalize.pt')
    # state = torch.load('models/AutoencoderClassifier_last_without_norms.pt')
    model = state['architecture']
    model.load_state_dict(state['state_dict'])

    # Пример использования
    num_images = 10
    generated_images = generate_images(model, num_images, device)

    # Выводим сгенерированные изображения
    for i in range(num_images):
        image = generated_images[i].permute(1, 2, 0).numpy()
        plt.imshow(image)
        plt.show()

