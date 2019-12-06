import os
import time
import torch
import h5py
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

from create_dataset import DATASET_HDF

LATENT_SIZE = 200
CUBE_SIZE = 32
LEAK_VALUE = 0.2
Z_SIZE = 200
EPOCHS = 1000
BATCH_SIZE = 400 # for 6GB VRAM
D_LR = 0.001
G_LR = 0.0025
D_THRESH = 0.8
LOG_PATH = './log/'
GENERATED_PATH = './generated_models/'

class VoxelData(Dataset):
    def __init__(self, path):
        self.path = path
        self.index = self._create_index()

    def __getitem__(self, index):
        idx_c, idx_d = self.index[index]
        with h5py.File(self.path) as hdf:
            tensor = torch.Tensor(hdf[idx_c][idx_d])
        return tensor

    def _create_index(self):
        with h5py.File(self.path) as hdf:
            pairs = []
            categories = list(hdf.keys())
            for c in categories:
                datasets = list(hdf[c].keys())
                for d in datasets:
                    pairs.append((c, d))
        return pairs

    def __len__(self):
        return len(self.index)


class Generator(torch.nn.Module):
    def __init__(self, latent_size=LATENT_SIZE, z_size=Z_SIZE):
        super(Generator, self).__init__()
        self.cube_len = CUBE_SIZE
        self.z_size = Z_SIZE

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
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cube_len = CUBE_SIZE
        self.leak_value = LEAK_VALUE

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
        D_solver = torch.optim.Adam(self.discriminator.parameters(), lr=D_LR, betas=betas)
        G_solver = torch.optim.Adam(self.generator.parameters(), lr=G_LR, betas=betas)
        criterion = torch.nn.BCELoss()

        if not os.path.exists(LOG_PATH):
            os.mkdir(LOG_PATH)
        writer = SummaryWriter(LOG_PATH)

        start_time = time.time()

        for epoch in range(EPOCHS):
            for i, X in enumerate(data):
                i_start = time.time()

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
                    D_solver.step()

                # ================== train generator ==================
                Z = torch.randn(BATCH_SIZE, Z_SIZE)

                fake = self.generator(Z)
                d_fake = self.discriminator(fake)
                g_loss = criterion(d_fake, real_labels)

                self.discriminator.zero_grad()
                self.generator.zero_grad()
                g_loss.backward()
                G_solver.step()

                print('Epoch {:3d}, Batch {:3d}/{:3d} done, took: {:.3f}s'.format(epoch+1, i+1, len(data), time.time()-i_start))
                i_start = time.time()

            # # ================== print and log progress ==================
            writer.add_scalar('Loss Discriminator', d_loss.item(), epoch)
            writer.add_scalar('Loss Generator', g_loss.item(), epoch)
            writer.add_scalar('Accuracy Discriminator', d_total_acu, epoch)

            if (epoch % 10) == 0:
                Z = torch.randn(BATCH_SIZE, Z_SIZE)
                generated = self.generator(Z)
                torch.save(generated, GENERATED_PATH + 'example_after_{:4d}_iterations.pt'.format(epoch))

            print('Iter: {0:5d}, D_loss: {1:.4}, G_loss: {2:.4}, D_acu: {3:.4}, took: {4:.4}s'.format(epoch, d_loss.item(), g_loss.item(), d_total_acu, time.time() - start_time))
            start_time = time.time()


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = VoxelData(DATASET_HDF)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    gan = GAN()
    gan.train(data_loader)
