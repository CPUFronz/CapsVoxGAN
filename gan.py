import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from voxeldata import VoxelData
from constants import DATASET_HDF
from constants import CUBE_SIZE
from constants import LEAK_VALUE
from constants import Z_SIZE
from constants import EPOCHS
from constants import BATCH_SIZE
from constants import DISCRIMINATOR_LEARNING_RATE
from constants import GENERATOR_LEARNING_RATE
from constants import DISCRIMINATOR_THRESHOLD
from constants import LOG_PATH
from constants import GENERATED_PATH
from constants import CLASSES
from constants import RANDOM_SEED
from constants import SAVED_DISCRIMINATOR
from constants import SAVED_GENERATOR

torch.manual_seed(RANDOM_SEED)


class Generator(torch.nn.Module):
    def __init__(self, cube_size=CUBE_SIZE, z_size=Z_SIZE):
        super(Generator, self).__init__()
        self.cube_size = cube_size
        self.z_size = z_size

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.z_size, self.cube_size * 8, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_size * 8),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_size * 8, self.cube_size * 4, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_size * 4),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_size * 4, self.cube_size * 2, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_size * 2),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_size * 2, self.cube_size, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_size),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_size, 1, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, self.z_size, 1, 1, 1)  # torch.Size([BATCH_SIZE, 200, 1, 1, 1])
        out = self.layer1(out)                  # torch.Size([BATCH_SIZE, 256, 2, 2, 2])
        out = self.layer2(out)                  # torch.Size([BATCH_SIZE, 128, 4, 4, 4])
        out = self.layer3(out)                  # torch.Size([BATCH_SIZE, 64, 8, 8, 8])
        out = self.layer4(out)                  # torch.Size([BATCH_SIZE, 32, 16, 16, 16])
        out = self.layer5(out)                  # torch.Size([BATCH_SIZE, 1, 32, 32, 32])
        return out


class Discriminator(torch.nn.Module):
    def __init__(self, cube_size=CUBE_SIZE, leak_value=LEAK_VALUE):
        super(Discriminator, self).__init__()
        self.cube_size = cube_size
        self.leak_value = leak_value

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, self.cube_size, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_size),
            torch.nn.LeakyReLU(self.leak_value)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_size, self.cube_size * 2, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_size * 2),
            torch.nn.LeakyReLU(self.leak_value)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_size * 2, self.cube_size * 4, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_size * 4),
            torch.nn.LeakyReLU(self.leak_value)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_size * 4, self.cube_size * 8, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_size * 8),
            torch.nn.LeakyReLU(self.leak_value)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_size * 8, 1, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, 1, self.cube_size, self.cube_size, self.cube_size)  # torch.Size([BATCH_SIZE, 1, 32, 32, 32])
        out = self.layer1(out)                                               # torch.Size([BATCH_SIZE, 32, 16, 16, 16])
        out = self.layer2(out)                                               # torch.Size([BATCH_SIZE, 64, 8, 8, 8])
        out = self.layer3(out)                                               # torch.Size([BATCH_SIZE, 128, 4, 4, 4])
        out = self.layer4(out)                                               # torch.Size([BATCH_SIZE, 256, 2, 2, 2])
        out = self.layer5(out)                                               # torch.Size([BATCH_SIZE, 1, 1, 1, 1])
        return out


class GAN:
    def __init__(self, z_size=Z_SIZE):
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.z_size = z_size

        if os.path.exists(SAVED_DISCRIMINATOR) and os.path.exists(SAVED_GENERATOR):
            self.discriminator = torch.load(SAVED_DISCRIMINATOR)
            self.generator = torch.load(SAVED_GENERATOR)

    def train(self, data):
        betas = (0.5, 0.5)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=DISCRIMINATOR_LEARNING_RATE, betas=betas)
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=GENERATOR_LEARNING_RATE, betas=betas)
        criterion = torch.nn.BCELoss()

        if not os.path.exists(LOG_PATH):
            os.mkdir(LOG_PATH)
        writer = SummaryWriter(LOG_PATH)

        for epoch in range(EPOCHS):
            for i, X in enumerate(tqdm(data)):
                current_batch_size = X.size()[0]
                if current_batch_size != BATCH_SIZE:
                    # drop last batch due to incompatible size
                    continue

                # ================== train discriminator ==================
                noise = torch.randn(current_batch_size, self.z_size)
                real_labels = torch.ones(current_batch_size)
                fake_labels = torch.zeros(current_batch_size)

                discriminator_real_prediction = self.discriminator(X)
                discriminator_real_loss = criterion(discriminator_real_prediction, real_labels)

                fake_models = self.generator(noise)
                discriminator_fake_prediction = self.discriminator(fake_models)
                discriminator_fake_loss = criterion(discriminator_fake_prediction, fake_labels)

                discriminator_loss = discriminator_real_loss + discriminator_fake_loss

                discriminator_accuracy_real = torch.ge(discriminator_real_prediction.squeeze(), 0.5).float()
                discriminator_accuracy_fake = torch.le(discriminator_fake_prediction.squeeze(), 0.5).float()
                discriminator_accuracy = torch.mean(torch.cat((discriminator_accuracy_real, discriminator_accuracy_fake), 0))

                if discriminator_accuracy <= DISCRIMINATOR_THRESHOLD:
                    self.discriminator.zero_grad()
                    discriminator_loss.backward()
                    discriminator_optimizer.step()

                # ================== train generator ==================
                noise = torch.randn(current_batch_size, self.z_size)

                fake_models = self.generator(noise)
                discriminator_fake_prediction = self.discriminator(fake_models)
                generator_loss = criterion(discriminator_fake_prediction, real_labels)

                self.discriminator.zero_grad()
                self.generator.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()

            # # ================== print and log progress ==================
            writer.add_scalar('Loss Discriminator', discriminator_loss.item(), epoch)
            writer.add_scalar('Loss Generator', generator_loss.item(), epoch)
            writer.add_scalar('Accuracy Discriminator', discriminator_accuracy, epoch)

            if (epoch % 10) == 0:
                noise = torch.randn(current_batch_size, self.z_size)
                generated = self.generator(noise)
                torch.save(generated, GENERATED_PATH + 'example_after_{:04d}_iterations.pt'.format(epoch))

            print('Iter: {0:5d}, D_loss: {1:.4}, G_loss: {2:.4}, D_acu: {3:.4}'.format(epoch, discriminator_loss.item(), generator_loss.item(), discriminator_accuracy))

        torch.save(self.discriminator, SAVED_DISCRIMINATOR)
        torch.save(self.generator, SAVED_GENERATOR)


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = VoxelData(DATASET_HDF, CLASSES)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    gan = GAN()
    gan.train(data_loader)
