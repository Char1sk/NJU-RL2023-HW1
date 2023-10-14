import numpy as np
from abc import abstractmethod

import random

import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg16_bn, resnet18
from torchvision import transforms as tfs
from PIL import Image


class DaggerAgent:
    def __init__(self,):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass


class MyDaggerAgent(DaggerAgent):
    def __init__(self, writer, time_try):
        super().__init__()
        # init your model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model = vgg16_bn(num_classes=8).to(self.device)
        # self.model = resnet18(num_classes=8).to(self.device)
        self.model = MyNet(time_try).to(self.device)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optm = torch.optim.Adam(self.model.parameters(), 1e-3)
        
        self.model_init_weight = self.model.state_dict()
        self.optm_init_weight = self.optm.state_dict()
        
        self.writer = writer
        self.actions = [2, 5, 4, 3, 12, 11, 1, 0]
    
    # train your model with labeled data
    def update(self, data_batch, label_batch, round, is_long):
        self.model.train()
        if not is_long:
            self.reboot()
        # self.model.train(data_batch, label_batch)
        dataset = MyDataset(data_batch, label_batch, self.actions)
        dataloader = DataLoader(dataset, 40, shuffle=True)
        
        # for ep in tqdm.trange(40):
        #     for idx, (data, label) in enumerate(dataloader):
        steps = 0
        total = 1000
        with tqdm.tqdm(total=total) as pbar:
            while steps < total:
                for idx, (data, label) in enumerate(dataloader):
                    data, label = data.to(self.device), label.to(self.device)
                    
                    pred = self.model(data)
                    
                    loss = self.loss(pred, label)
                    # print(pred[0], label[0], loss.item())
                    loss.backward()
                    
                    self.optm.step()
                    self.optm.zero_grad()
                    
                    # self.writer.add_scalar(f'loss/train_{round}', loss.item(), ep*len(dataloader)+idx)
                    self.writer.add_scalar(f'loss/train_{round}', loss.item(), steps)
                    steps += 1
                    pbar.update(1)
                    if steps >= total:
                        break
    
    # select actions by your model
    def select_action(self, data_batch):
        # if random.random() < 0.15:
        #     act = random.choice(self.actions)
        # else:
        #     x = tfs.ToTensor()(data_batch)
        #     x = x.unsqueeze(0)
        #     x = x.to(self.device)
            
        #     x = self.infer(x)
            
        #     idx = x.max(dim=1).indices.item()
        #     act = self.actions[idx]
        x = tfs.ToTensor()(data_batch)
        x = x.unsqueeze(0)
        x = x.to(self.device)
        
        x = self.infer(x)
        
        idx = x.max(dim=1).indices.item()
        act = self.actions[idx]
        
        # label_predict = self.model.predict(data_batch)
        # act = random.choice([0,1,2,3,4,5,11,12])
        return act
    
    def infer(self, x):
        with torch.no_grad():
            self.model.eval()
            x = self.model(x)
            x = torch.softmax(x, dim=1)
        return x
    
    def reboot(self):
        self.model.load_state_dict(self.model_init_weight)
        self.optm.load_state_dict(self.optm_init_weight)
    
    def save_model(self, id):
        torch.save(self.model.state_dict(), f'./saves/{id}.pt')
    
    def load_model(self, id):
        self.model.load_state_dict(torch.load(f'./saves/{id}.pt'))


# here is an example of creating your own Agent
class ExampleAgent(DaggerAgent):
    def __init__(self, necessary_parameters=None):
        super(DaggerAgent, self).__init__()
        # init your model
        self.model = None

    # train your model with labeled data
    def update(self, data_batch, label_batch):
        self.model.train(data_batch, label_batch)

    # select actions by your model
    def select_action(self, data_batch):
        label_predict = self.model.predict(data_batch)
        return label_predict


class MyDataset(Dataset):
    def __init__(self, data_batch, label_batch, actions) -> None:
        super().__init__()
        self.data_batch = data_batch
        self.label_batch = label_batch
        self.actions = torch.Tensor(actions)
    
    def __getitem__(self, idx):
        data = self.data_batch[idx]
        # data = np.array(data, dtype=np.uint8)
        # data = Image.fromarray(data)
        data = tfs.ToTensor()(data)
        
        label = self.label_batch[idx]
        label = int(label)
        label_onehot = (self.actions == label*torch.ones_like(self.actions)).to(torch.float32)
        
        return (data, label_onehot)
    
    def __len__(self):
        assert len(self.data_batch) == len(self.label_batch)
        return len(self.label_batch)


class MyNet(nn.Module):
    def __init__(self, time_try):
        super().__init__()
        
        in_nc = 3 if not time_try else 6
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_nc, 8, 7, 3),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(8, 16, 7, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.convs = nn.Sequential(
            self.conv_block_1,
            self.conv_block_2
        )
        
        self.flatten = nn.Flatten()
        
        self.fc_block_1 = nn.Sequential(
            nn.Linear(5376, 512),
            nn.ReLU(True)
        )
        self.fc_block_2 = nn.Sequential(
            nn.Linear(512, 8)
        )
        self.fcs = nn.Sequential(
            self.fc_block_1,
            self.fc_block_2
        )
        
        self.mods = nn.Sequential(
            nn.Conv2d(in_nc, 8, 7, 2, 3),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),
            
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(16*27*20, 512),
            nn.ReLU(True),
            nn.Linear(512,8)
        )
    
    def forward(self, x):
        # x = self.convs(x)
        # # 1x16x21x16
        # x = self.flatten(x)
        # # 1x5376
        # x = self.fcs(x)
        x = self.mods(x)
        return x


