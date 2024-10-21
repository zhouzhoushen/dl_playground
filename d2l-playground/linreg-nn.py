from torch import nn
import numpy as np
import torch
from torch.utils import data
from utils import d2l

def load_array(data_arrays, batch_size, is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)

def main():
    print(torch.cuda.is_available())

    # Begin.
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    
    loss = nn.MSELoss()
    
    trainer = torch.optim.SGD(net.parameters(), lr = 0.03)

    epochs = 3
    for epoch in range(epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss {float(l):f}")
    # End.

if __name__ == "__main__":
    main()