import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class VariationConvAutoencoderNoPool(nn.Module):
    def __init__(self, latent_dim):
        super(VariationConvAutoencoderNoPool, self).__init__()

        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 32x32x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64x16x16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128x8x8
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 256x4x4
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # 256x2x2
            nn.ReLU(True),
        )

        # Латентные слои
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)

        # Декодер
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4x256
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 8x8x128
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 16x16x64
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 32x32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),  # 64x64x3
            nn.Tanh()
        )

    def forward(self, x):
        # Проход через энкодер
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        # Вычисление латентного представления
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Параметризация латентного представления
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Проход через декодер
        z = z.view(z.size(0), 256, 2, 2)
        x_hat = self.decoder(z)

        return x_hat, mu, logvar


def vae_loss(criterion, recon_x, x, mu, logvar):
    # Вычисляем criterion потерю для восстановления
    reconstruction_loss = criterion(recon_x, x)

    # Вычисляем KL-дивергенцию между предсказанным и заданным распределениями
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Суммируем оба компонента потерь
    loss = reconstruction_loss + kl_divergence

    return loss

def train_vae_with_cross(model, train_loader, criterion, optimizer, num_epochs, device, n_splits=5):
    model.train()
    loss_list = []
    val_loss_list = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 0
    for train_index, val_index in kf.split(train_loader.dataset):
        train_sampler = SubsetRandomSampler(train_index)
        val_sampler = SubsetRandomSampler(val_index)
        train_loader_fold = torch.utils.data.DataLoader(train_loader.dataset, batch_size=128, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=128, sampler=val_sampler)
        fold += 1
        print(f'Fold {fold}/{n_splits}')
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            pbar = tqdm(total=len(train_loader_fold))
            for inputs, _ in train_loader_fold:
                inputs = inputs.to(device)
                optimizer.zero_grad()
                if model.is_vae:
                    outputs, mu, logvar = model(inputs)
                    loss = vae_loss(criterion, outputs, inputs, mu, logvar)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                pbar.set_postfix({'train_loss': running_loss / ((pbar.n + 1) * inputs.size(0))})
                pbar.update()
            epoch_loss = running_loss / len(train_loader_fold.dataset)
            loss_list.append(epoch_loss)
            pbar.close()
            tqdm.write(
                'Fold [%d/%d], Epoch [%d/%d], Train Loss: %.4f' % (fold, n_splits, epoch + 1, num_epochs, epoch_loss))
            # Evaluate on validation set
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, _ in val_loader:
                    inputs = inputs.to(device)
                    if model.is_vae:
                        outputs, mu, logvar = model(inputs)
                        val_loss = vae_loss(criterion, outputs, inputs, mu, logvar)
                    else:
                        outputs = model(inputs)
                        val_loss = criterion(outputs, inputs)
                    running_val_loss += val_loss.item() * inputs.size(0)
            val_loss = running_val_loss / len(val_loader.dataset)
            val_loss_list.append(val_loss)
            tqdm.write('Fold [%d/%d], Epoch [%d/%d], Validation Loss: %.4f' % (
            fold, n_splits, epoch + 1, num_epochs, val_loss))
    return loss_list, val_loss_list

def simple_train(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        # Перебираем данные в DataLoader
        for batch_data, _ in train_loader:
            # Очищаем градиенты
            optimizer.zero_grad()

            # Передаем данные на устройство
            batch_data = batch_data.to(device)

            # Прямой проход
            recon_batch, mu, logvar = model(batch_data)

            # Вычисляем потерю
            loss = vae_loss(criterion ,recon_batch, batch_data, mu, logvar)

            # Обратное распространение и обновление параметров
            loss.backward()
            optimizer.step()

        # Выводим информацию о потере
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")