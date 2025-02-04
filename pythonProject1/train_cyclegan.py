import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_channels=3, n_resblocks=9):
        super().__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_channels, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_resblocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 3, 7),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x).view(-1)


# Инициализация моделей и оптимизаторов
def initialize_models(device):
    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )

    return G_A2B, G_B2A, D_A, D_B, optimizer_G, optimizer_D


# Обучение
def train_cyclegan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    num_epochs = 500

    # Инициализация
    G_A2B, G_B2A, D_A, D_B, opt_G, opt_D = initialize_models(device)
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Трансформации
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_A = datasets.ImageFolder("dataset/trainA", transform=transform)
    dataset_B = datasets.ImageFolder("dataset/trainB", transform=transform)
    loader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
    loader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for i, (real_A, real_B) in enumerate(zip(loader_A, loader_B)):
            real_A = real_A[0].to(device)
            real_B = real_B[0].to(device)

            # Обучение генераторов
            opt_G.zero_grad()

            same_B = G_A2B(real_B)
            loss_id_B = criterion_identity(same_B, real_B) * 5.0

            same_A = G_B2A(real_A)
            loss_id_A = criterion_identity(same_A, real_A) * 5.0

            # GAN loss
            fake_B = G_A2B(real_A)
            loss_GAN_A2B = criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B)))

            fake_A = G_B2A(real_B)
            loss_GAN_B2A = criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A)))

            # Cycle loss
            recovered_A = G_B2A(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = G_A2B(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B) * 10.0

            # Total loss
            loss_G = loss_id_A + loss_id_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B
            loss_G.backward()
            opt_G.step()

            # Обучение дискриминаторов
            opt_D.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A)))
            loss_real += criterion_GAN(D_B(real_B), torch.ones_like(D_B(real_B)))

            # Fake loss
            loss_fake = criterion_GAN(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)))
            loss_fake += criterion_GAN(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)))

            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            opt_D.step()

        # Сохранение моделей
        if epoch % 10 == 0:
            torch.save(G_A2B.state_dict(), "weights/G_A2B_epoch_99.pth")
            torch.save(G_A2B.state_dict(), "weights/G_A2B_epoch_99.pth")


if __name__ == "__main__":
    train_cyclegan()