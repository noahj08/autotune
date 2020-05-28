import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import choose


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.hidden = nn.Linear(2,3)
        nn.init.kaiming_normal_(self.hidden.weight)
        self.output = nn.Linear(3,1)
        nn.init.kaiming_normal_(self.output.weight)
    def forward(self,x):
        out = self.hidden(x.float())
        out = F.relu(out)
        out = self.output(out)
        out = F.sigmoid(out)
        return out

class Nron(nn.Module):
    def __init(self):
        super(Nron,self).__init__()
        self.layer = nn.Linear(2,1)
        #self.layer2 = nn.Linear(1,1)
        #nn.init.kaiming_normal_(self.layer.weight)
    def forward(self,x):
        out = self.layer(x.float())
        out = F.sigmoid(out)
        #out = self.layer2(out)
        return out

class ThreeLayer(nn.Module):
    def __init__(self):
        super(ThreeLayer, self).__init__()
        self.hidden = nn.Linear(2,5)
        nn.init.kaiming_normal_(self.hidden.weight)
        self.hidden2 = nn.Linear(5,5)
        nn.init.kaiming_normal_(self.hidden2.weight)
        self.output = nn.Linear(5,1)
        nn.init.kaiming_normal_(self.output.weight)
    def forward(self,x):
        out = self.hidden(x.float())
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.relu(out)
        out = self.output(out)
        out = F.sigmoid(out)
        return out

def get_optimizer(model):
    """This is where users choose their optimizer and define the
       hyperparameter space they'd like to search."""
    optimizer_class = optim.SGD
    lr = choose(np.logspace(-5, 0, base=10))
    momentum = choose(np.linspace(0.1, .9999))
    print(model)
    return optimizer_class(model.parameters(), lr=lr, momentum=momentum)


DATA_DIR = "../../data/lin/"
MODEL_CLASS = Nron#SimpleNet#ConvNet
LOSS_FN = F.binary_cross_entropy
HYPERPARAM_NAMES = ["lr", "momentum"]  # This is unfortunate.
EPOCHS = 10
BATCH_SIZE = 64
POPULATION_SIZE = 15  # Number of models in a population
EXPLOIT_INTERVAL = 0.5  # When to exploit, in number of epochs
USE_SQLITE = True # If False, you'll need to set up a local Postgres server
