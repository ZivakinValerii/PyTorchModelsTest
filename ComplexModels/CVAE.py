import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.model_selection import KFold


# Определение модели CVAE
class ConvConditionalVAE(nn.Module):
    def __init__(self, latent_size, num_classes):
        super(ConvConditionalVAE, self).__init__()

        self.num_classes = num_classes

        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 256x4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # Статистические параметры
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_size)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_size)

        # Классификатор
        #self.classifier = nn.Linear(latent_size, num_classes)

        # Декодер
        self.decoder_input = nn.Linear(latent_size + num_classes, 256 * 4 * 4)  # Слой компенсации
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 3x64x64
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(z.size(0), 256, 4, 4)
        x = self.decoder(z)
        return x

    def forward(self, x, labels):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # # Добавляем информацию о классе
        # class_logits = self.classifier(z)
        #
        # # Применяем softmax к logits классов
        # class_probs = F.softmax(class_logits, dim=1)
        #
        # # Умножаем коды классов на вероятности классов
        # weighted_z = z * class_probs[:, :, None, None]

        # Преобразуем метки классов в ожидаемую размерность
        #labels = labels.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # Склеиваем коды классов с кодами изображений
        concat_z = torch.cat([z, labels], dim=1)

        x_recon = self.decode(concat_z)

        return x_recon, mu, logvar


def train_conditional_vae(model, dataloader, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Преобразование меток в one-hot кодирование
            num_classes = len(dataloader.dataset.classes)
            one_hot_labels = F.one_hot(labels, num_classes).float()

            # Прямой проход
            # class_logits
            recon_images, mu, logvar = model(images, one_hot_labels)

            # Вычисление функции потерь
            loss = vae_loss(recon_images, images, mu, logvar)

            # Обратное распространение и обновление весов
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Выводим среднюю потерю на эпохе
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss}")

    print("Training completed.")

def vae_loss(recon_images, images, mu, logvar):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_images, images)

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Classification loss
    #class_loss = F.cross_entropy(class_logits, labels)

    # Total loss
    total_loss = recon_loss + kl_loss
    #+ class_loss

    return total_loss
