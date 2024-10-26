import torch 
from torch import nn
from utils import d2l

def main():
    print(torch.cuda.is_available())

    # Begin.
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    def init_weight(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std = 0.01)
    net.apply(init_weight)
    
    loss = nn.CrossEntropyLoss()

    trainer = torch.optim.SGD(net.parameters(), lr = 0.1)
    
    num_epochs = 10
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    # End.

if __name__ == "__main__":
    main()