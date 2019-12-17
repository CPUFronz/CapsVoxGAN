import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from voxeldata import VoxelData
from constants import DATASET_HDF
from constants import LATENT_SIZE
from constants import CUBE_SIZE
from constants import LEAK_VALUE
from constants import Z_SIZE
from constants import EPOCHS
from constants import BATCH_SIZE
from constants import D_LR
from constants import G_LR
from constants import D_THRESH
from constants import LOG_PATH
from constants import GENERATED_PATH
from constants import CLASSES
from constants import RANDOM_SEED

torch.manual_seed(RANDOM_SEED)

class Generator(torch.nn.Module):
    def __init__(self, cube_size=CUBE_SIZE, latent_size=LATENT_SIZE, z_size=Z_SIZE):
        super(Generator, self).__init__()
        self.cube_len = cube_size
        self.z_size = z_size

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(latent_size, self.cube_len * 8, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*8, self.cube_len*4, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*4, self.cube_len*2, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*2, self.cube_len, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len, 1, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, self.z_size, 1, 1, 1) # torch.Size([100, 200, 1, 1, 1])
        out = self.layer1(out) # torch.Size([100, 512, 4, 4, 4])
        out = self.layer2(out) # torch.Size([100, 256, 8, 8, 8])
        out = self.layer3(out) # torch.Size([100, 128, 16, 16, 16])
        out = self.layer4(out) # torch.Size([100, 64, 32, 32, 32])
        out = self.layer5(out) # torch.Size([100, 1, 64, 64, 64])
        return out


class Discriminator(torch.nn.Module):
    def __init__(self, cube_size=CUBE_SIZE, leak_value=LEAK_VALUE):
        super(Discriminator, self).__init__()
        self.cube_len = cube_size
        self.leak_value = leak_value

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, self.cube_len, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.LeakyReLU(self.leak_value)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.LeakyReLU(self.leak_value)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.LeakyReLU(self.leak_value)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.LeakyReLU(self.leak_value)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*8, 1, kernel_size=4, stride=2, padding=(1,1,1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, 1, self.cube_len, self.cube_len, self.cube_len) # torch.Size([100, 1, 64, 64, 64])
        out = self.layer1(out) # torch.Size([100, 64, 32, 32, 32])
        out = self.layer2(out) # torch.Size([100, 128, 16, 16, 16])
        out = self.layer3(out) # torch.Size([100, 256, 8, 8, 8])
        out = self.layer4(out) # torch.Size([100, 512, 4, 4, 4])
        out = self.layer5(out) # torch.Size([100, 200, 1, 1, 1])
        return out


class GAN():
    def __init__(self):
        self.discriminator = Discriminator()
        self.generator = Generator()

    def train(self, data):
        betas = (0.5, 0.5)
        d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=D_LR, betas=betas)
        g_optim = torch.optim.Adam(self.generator.parameters(), lr=G_LR, betas=betas)
        criterion = torch.nn.BCELoss()

        if not os.path.exists(LOG_PATH):
            os.mkdir(LOG_PATH)
        writer = SummaryWriter(LOG_PATH)

        for epoch in range(EPOCHS):
            for i, X in enumerate(tqdm(data)):
                if X.size()[0] != BATCH_SIZE:
                    # drop last batch due to incompatible size
                    continue

                # ================== train discriminator ==================
                Z = torch.randn(BATCH_SIZE, Z_SIZE)
                real_labels = torch.ones(BATCH_SIZE)
                fake_labels = torch.zeros(BATCH_SIZE)

                d_real = self.discriminator(X)
                d_real_loss = criterion(d_real, real_labels)

                fake = self.generator(Z)
                d_fake = self.discriminator(fake)
                d_fake_loss = criterion(d_fake, fake_labels)

                d_loss = d_real_loss + d_fake_loss

                d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
                d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
                d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu),0))

                if d_total_acu <= D_THRESH:
                    self.discriminator.zero_grad()
                    d_loss.backward()
                    d_optim.step()

                # ================== train generator ==================
                Z = torch.randn(BATCH_SIZE, Z_SIZE)

                fake = self.generator(Z)
                d_fake = self.discriminator(fake)
                g_loss = criterion(d_fake, real_labels)

                self.discriminator.zero_grad()
                self.generator.zero_grad()
                g_loss.backward()
                g_optim.step()

            # # ================== print and log progress ==================
            writer.add_scalar('Loss Discriminator', d_loss.item(), epoch)
            writer.add_scalar('Loss Generator', g_loss.item(), epoch)
            writer.add_scalar('Accuracy Discriminator', d_total_acu, epoch)

            if (epoch % 10) == 0:
                Z = torch.randn(BATCH_SIZE, Z_SIZE)
                generated = self.generator(Z)
                torch.save(generated, GENERATED_PATH + 'example_after_{:04d}_iterations.pt'.format(epoch))

            print('Iter: {0:5d}, D_loss: {1:.4}, G_loss: {2:.4}, D_acu: {3:.4}'.format(epoch, d_loss.item(), g_loss.item(), d_total_acu))

        torch.save(self.discriminator, 'discriminator.pkl')
        torch.save(self.generator, 'generator.pkl')


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = VoxelData(DATASET_HDF, CLASSES)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    gan = GAN()
    gan.train(data_loader)
