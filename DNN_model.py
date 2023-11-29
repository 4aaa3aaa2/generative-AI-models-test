import torch, time, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, class_num=10):
        
      super(DNN, self).__init__()
       
        self.input_dim = input_dim + class_num
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, 7, 7)
        x = self.deconv(x)
        return x


class ImageGenerator(object):
    def __init__(self):
        
        self.epoch = 5
        self.sample_num = 100
        self.batch_size = 64
        self.z_dim = 62
        self.lr = 0.0001
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_dataloader()
        self.model = DNN(input_dim=self.z_dim, output_dim=self.output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss().to(self.device)

    def init_dataloader(self):
        
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
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
        self.output_dim = self.train_dataloader.__iter__().__next__()[0].shape[1]

    def train(self):
        self.model.train()
        print('start training')
        for epoch in range(self.epoch):
            self.model.train()
            loss_mean = 0
            for i, (images, labels) in enumerate(self.train_dataloader):
                z = torch.rand((self.batch_size, self.z_dim)).to(self.device)
                images, labels = images.to(self.device), labels.to(self.device)
                labels = F.one_hot(labels, num_classes=10)
                self.optimizer.zero_grad()
                generated_images = self.model(torch.cat((z, labels), dim=1))
                loss = self.loss(generated_images, images)
                loss_mean += loss.item()
                loss.backward()
                self.optimizer.step()
            train_loss = loss_mean / len(self.train_dataloader)
            val_loss = self.evaluation()
            print('epoch:{}, training loss:{:.4f}, validation loss:{:.4f}'.format(epoch, train_loss, val_loss))
            self.visualize_results(epoch)

    @torch.no_grad()
    def evaluation(self):
        self.model.eval()
        loss_mean = 0
        for i, (images, labels) in enumerate(self.val_dataloader):
            z = torch.rand((images.shape[0], self.z_dim)).to(self.device)
            images, labels = images.to(self.device), labels.to(self.device)
            labels = F.one_hot(labels, num_classes=10)
            generated_images = self.model(torch.cat((z, labels), dim=1))
            loss = self.loss(generated_images, images)
            loss_mean += loss.item()
        return loss_mean / len(self.val_dataloader)

    @torch.no_grad()
    def visualize_results(self, epoch):
        self.model.eval()
        output_path = 'results/DNN'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        tot_num_samples = self.sample_num
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        z = torch.rand((tot_num_samples, self.z_dim)).to(self.device)
        labels = F.one_hot(torch.Tensor(np.repeat(np.arange(10), 10)).to(torch.int64), num_classes=10).to(self.device)
        generated_images = self.model(torch.cat((z, labels), dim=1))
        save_image(generated_images, os.path.join(output_path, '{}.jpg'.format(epoch)), nrow=image_frame_dim)


if __name__ == '__main__':
    generator = ImageGenerator()
    generator.train()
