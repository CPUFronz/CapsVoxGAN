import torch
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm

from gan import VoxelData
from constants import CUBE_SIZE
from constants import DATASET_HDF
from constants import CLASSES
from constants import NUM_CLASSES
from constants import BATCH_SIZE
from constants import RANDOM_SEED

torch.manual_seed(RANDOM_SEED)


class PrimaryPointCapsLayer(torch.nn.Module):
    def __init__(self, prim_vec_size=8, num_points=2048):
        super(PrimaryPointCapsLayer, self).__init__()
        self.capsules = torch.nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', torch.nn.Conv3d(CUBE_SIZE*2, 1024, 1)),
                ('bn3', torch.nn.BatchNorm3d(1024)),
                ('mp1', torch.nn.MaxPool3d(num_points)),
            ]))
            for _ in range(prim_vec_size)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=2)
        return self.squash(u.squeeze())

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        if (output_tensor.dim() == 2):
            output_tensor = torch.unsqueeze(output_tensor, 0)
        return output_tensor


class LatentCapsLayer(torch.nn.Module):
    def __init__(self, latent_caps_size=16, prim_caps_size=1024, prim_vec_size=16, latent_vec_size=64):
        super(LatentCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size
        self.prim_caps_size = prim_caps_size
        self.latent_caps_size = latent_caps_size
        self.W = torch.nn.Parameter(0.01 * torch.randn(latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size))

    def forward(self, x):
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        u_hat_detached = u_hat.detach()

        b_ij = torch.autograd.Variable(torch.zeros(x.size(0), self.latent_caps_size, self.prim_caps_size)).cuda()
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, 1)
            if iteration == num_iterations - 1:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                b_ij = b_ij + torch.sum(v_j * u_hat_detached, dim=-1)

        return v_j.squeeze(-2)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class PointCapsNet(torch.nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, latent_caps_size, latent_vec_size, num_points):
        super(PointCapsNet, self).__init__()
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv3d(1, CUBE_SIZE, kernel_size=4,stride=2, padding=(1,1,1)),
            torch.nn.BatchNorm3d(CUBE_SIZE),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(CUBE_SIZE, CUBE_SIZE * 2, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(CUBE_SIZE * 2),
            torch.nn.ReLU(inplace=True),
        )
        self.primary_point_caps_layer = PrimaryPointCapsLayer(prim_vec_size, num_points)
        self.latent_caps_layer = LatentCapsLayer(latent_caps_size, prim_caps_size, prim_vec_size, latent_vec_size)

        # TODO: use Conv3d
        self.caps_decoder = torch.nn.Sequential(
            torch.nn.Linear(16 * NUM_CLASSES, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1024*4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024*4, CUBE_SIZE**2),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data = data.view(-1, 1, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        x1 = self.conv_layer(data)
        x2 = self.primary_point_caps_layer(x1)
        latent_capsules = self.latent_caps_layer(x2)
        # TODO: I'm not sure about this view ...
        reconstructions = self.caps_decoder(latent_capsules).view(-1, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        return latent_capsules, reconstructions

    def loss(self, data, reconstruction):
        loss = torch.sum(torch.abs(data - reconstruction))
        return loss


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    prim_caps_size = 1024
    prim_vec_size = 16
    latent_caps_size = 32
    latent_vec_size = 16
    num_points = 2048
    caps = PointCapsNet(prim_caps_size, prim_vec_size, latent_caps_size, latent_vec_size, num_points)

    dataset = VoxelData(DATASET_HDF, CLASSES)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(caps.parameters(), lr=0.0001)

    for epoch in range(30):
        train_loss_sum = 0
        for i, X in enumerate(tqdm(data_loader)):
            if X.size()[0] != BATCH_SIZE:
                continue

            _, reconstruction = caps(X)
            train_loss = caps.loss(X, reconstruction)
            train_loss.backward()
            optimizer.step()
            train_loss_sum += train_loss.item()

        print(epoch, train_loss_sum)
