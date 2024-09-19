import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(mnist, batch_size=64, shuffle=True)

input_size = 784  # Размер изображения 28x28
z_dim = 100       # Размер случайного шума
lr = 0.0002       # Скорость обучения
num_epochs = 100  # Количество эпох

generator = Generator(z_dim, input_size)
discriminator = Discriminator(input_size)

optim_g = optim.Adam(generator.parameters(), lr=lr)
optim_d = optim.Adam(discriminator.parameters(), lr=lr)

loss_function = nn.BCELoss()  # Функция потерь для дискриминатора

for epoch in range(num_epochs):
    for real_data, _ in data_loader:
        # Обработка реальных изображений
        real_data = real_data.view(-1, input_size)
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # 1. Обучение дискриминатора на реальных данных
        outputs = discriminator(real_data)
        d_loss_real = loss_function(outputs, real_labels)

        # 2. Обучение дискриминатора на сгенерированных данных
        z = torch.randn(batch_size, z_dim)
        fake_data = generator(z)
        outputs = discriminator(fake_data.detach())
        d_loss_fake = loss_function(outputs, fake_labels)

        # Итоговые потери дискриминатора
        d_loss = d_loss_real + d_loss_fake
        optim_d.zero_grad()
        d_loss.backward()
        optim_d.step()

        # 3. Обучение генератора (цель – обмануть дискриминатор)
        outputs = discriminator(fake_data)
        g_loss = loss_function(outputs, real_labels)

        optim_g.zero_grad()
        g_loss.backward()
        optim_g.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

    # Визуализация сгенерированных изображений каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(64, z_dim)
            fake_images = generator(z).view(-1, 1, 28, 28)
            plt.figure(figsize=(5, 5))
            for i in range(25):
                plt.subplot(5, 5, i + 1)
                plt.imshow(fake_images[i][0].cpu().numpy(), cmap='gray')
                plt.axis('off')
            plt.show()

