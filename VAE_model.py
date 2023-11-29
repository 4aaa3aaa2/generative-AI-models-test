import torch, time, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
 
 
class VAE(nn.Module):
    def __init__(self, middle_dim=400, latent_dim=20, class_num=10):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784 + class_num, middle_dim)
        self.fc_mu = nn.Linear(middle_dim, latent_dim)
        self.fc_logvar = nn.Linear(middle_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim + class_num, middle_dim)
        self.fc3 = nn.Linear(middle_dim, 784)
        self.recons_loss = nn.BCELoss(reduction='sum')
 
    def encode(self, x, labels):
        x = torch.cat((x, labels), dim=1)
        x = torch.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
 
    def reparametrization(self, mu, logvar):
        # sigma = 0.5*exp(log(sigma^2))= 0.5*exp(log(var))
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        # N(mu, std^2) = N(0, 1) * std + mu
        z = eps * std + mu
        return z
 
    def decode(self, z, labels):
        z = torch.cat((z, labels), dim=1)
        x = torch.relu(self.fc2(z))
        x = F.sigmoid(self.fc3(x))
        return x
 
    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparametrization(mu, logvar)
        x_out = self.decode(z, labels)
        loss = self.loss_func(x_out, x, mu, logvar)
        return loss
 
    def loss_func(self, x_out, x, mu ,logvar):
        reconstruction_loss = self.recons_loss(x_out, x)
        KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu ** 2)
        # KLD_ele = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        # KLD = torch.sum(KLD_ele).mul_(-0.5)
        return reconstruction_loss + KL_divergence
 
 
 
class ImageGenerator(object):
    def __init__(self):
  
        self.epoch = 50
        self.sample_num = 100
        self.batch_size = 128
        self.latent_dim = 20
        self.lr = 0.001
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_dataloader()
        self.model = VAE(latent_dim=self.latent_dim, class_num=10).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
 
    def init_dataloader(self):
        tf = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = MNIST('./data/',
                              train=True,
                              download=True,
                              transform=tf)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_dataset = MNIST('./data/',
                            train=False,
                            download=True,
                            transform=tf)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
 
    def train(self):
        self.model.train()
        print('start training')
        for epoch in range(self.epoch):
            self.model.train()
            loss_mean = 0
            for i, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = F.one_hot(labels, num_classes=10)
                loss = self.model(images.view(images.shape[0], -1), labels)
 
                loss_mean += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_loss = loss_mean / len(self.train_dataloader)
            print('epoch:{}, loss:{:.4f}'.format(epoch, train_loss))
            self.visualize_results(epoch)
 
    @torch.no_grad()
    def visualize_results(self, epoch):
        self.model.eval()
        output_path = 'results/VAE'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
 
        tot_num_samples = self.sample_num
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        z = torch.randn(tot_num_samples, self.latent_dim).to(self.device)
        labels = F.one_hot(torch.Tensor(np.repeat(np.arange(10), 10)).to(torch.int64), num_classes=10).to(self.device)
        generated_images = self.model.decode(z, labels)
        generated_images = generated_images.view(generated_images.shape[0], 1, 28, 28)
        save_image(generated_images, os.path.join(output_path, '{}.jpg'.format(epoch)), nrow=image_frame_dim)
 
if __name__ == '__main__':
    generator = ImageGenerator()
    generator.train()
