import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 32x64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 32x32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 64x16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 128x8x8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 256x4x4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 256x4x4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 256x2x2
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256x4x4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 3x64x64
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

        self.latent_size = 256

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x


class ConvAutoencoderNoPool(nn.Module):
    def __init__(self):
        super(ConvAutoencoderNoPool, self).__init__()

        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 32x32x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64х16x16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128х8x8
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 256х4x4
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # 256х2x2
            nn.ReLU(True),
            # nn.Conv2d(512, 1024, 3, stride=2, padding=1), # 1024х1x1
            # nn.ReLU(True),
        )

        # Декодер
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1), # 2x2x512
            # nn.ReLU(True),
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
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x


class ModifiedClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ModifiedClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.final = nn.LogSoftmax(dim=1)
        # self.final = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.final(x)
        return x


class AutoencoderClassifier(nn.Module):
    def __init__(self, autoencoder, classifier):
        super(AutoencoderClassifier, self).__init__()
        self.autoencoder = autoencoder
        self.classifier = classifier

    def forward(self, x):
        # Пропускаем данные через автоэнкодер
        encoded = self.autoencoder.encoder(x)
        decoded = self.autoencoder.decoder(encoded)

        # Подаем закодированные данные на вход классификатору
        output = self.classifier(encoded)

        return decoded, output


def train_coder(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    loss_list = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(total=len(train_loader))
        for inputs, _ in train_loader:  # tqdm(train_loader, desc='Epoch ' + str(epoch + 1), leave=False):
            # Получаем входные данные и отправляем на устройство
            # inputs, _ = data
            inputs = inputs.to(device)

            # Обнуляем градиенты оптимизатора
            optimizer.zero_grad()

            # Получаем выход модели и вычисляем значение функции потерь
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            # Обновляем параметры модели на основе градиентов и вычисляем статистику обучения
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            # pbar.set_postfix({'loss': running_loss})
            pbar.set_postfix({'loss': running_loss / ((pbar.n + 1) * inputs.size(0))})
            pbar.update()
        epoch_loss = running_loss / len(train_loader.dataset)
        loss_list.append(epoch_loss)
        pbar.close()
        tqdm.write('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, epoch_loss))
    return loss_list


def train_coder_with_cross(model, train_loader, criterion, optimizer, num_epochs, device, n_splits=5):
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
                    outputs = model(inputs)
                    val_loss = criterion(outputs, inputs)
                    running_val_loss += val_loss.item() * inputs.size(0)
            val_loss = running_val_loss / len(val_loader.dataset)
            val_loss_list.append(val_loss)
            tqdm.write('Fold [%d/%d], Epoch [%d/%d], Validation Loss: %.4f' % (
            fold, n_splits, epoch + 1, num_epochs, val_loss))
    return loss_list, val_loss_list


def train_AutoClass(autoencoder, classifier, dataloader, criterion_cls, optimizer_cls,
                    criterion_ae, optimizer_ae, device, num_epochs):
    autoencoder.eval()  # выставляем автоэнкодер в режим оценки
    for param in autoencoder.parameters():
        param.requires_grad = False  # отключаем градиенты у автоэнкодера

    classifier.train()  # переводим классификатор в режим тренировки
    auto_class = AutoencoderClassifier(autoencoder, classifier)  # создаем объединенную модель
    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(total=len(dataloader))
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # обучаем классификатор
            optimizer_cls.zero_grad()
            outputs = auto_class(images)[1]
            loss_cls = criterion_cls(outputs, labels)
            loss_cls.backward()
            optimizer_cls.step()

            # # обучаем автоенкодер
            # optimizer_ae.zero_grad()
            # decoded = auto_class(images)[0]
            # loss_ae = criterion_ae(decoded, images)
            # loss_ae.backward()
            # optimizer_ae.step()

            running_loss += loss_cls.item() * images.size(0)
            # pbar.set_postfix({'loss': running_loss})
            pbar.set_postfix({'loss': running_loss / ((pbar.n + 1) * images.size(0))})
            pbar.update()
        epoch_loss = running_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)
        pbar.close()
        tqdm.write('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, epoch_loss))

    return auto_class, train_losses


def train2(model, optimizer, criterion, train_loader, device):
    # Переключение модели в режим обучения
    model.train()

    # Инициализация прогресс бара
    pbar = tqdm(total=len(train_loader))

    # Инициализация суммы потерь
    running_loss = 0.0

    # Итерация по батчам из train_loader
    for inputs, _ in train_loader:
        # Перенос данных на GPU (если доступен)
        inputs = inputs.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()

        # Прямой проход через модель
        outputs = model(inputs)

        # Вычисление функции потерь
        loss = criterion(outputs, inputs)

        # Обратный проход и оптимизация параметров
        loss.backward()
        optimizer.step()

        # Суммирование потерь
        running_loss += loss.item() * inputs.size(0)

        # Обновление прогресс бара
        pbar.set_postfix({'loss': running_loss / ((pbar.n + 1) * inputs.size(0))})
        pbar.update()

    # Завершение работы прогресс бара
    pbar.close()

    # Вычисление средней потери на эпоху
    epoch_loss = running_loss / len(train_loader.dataset)

    return epoch_loss
