import torch
from torch import nn
from utils import d2l

def main():
    print(torch.cuda.is_available())

    # Begin.
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(
        torch.normal(0, 0.01, size = (num_inputs, num_hiddens), requires_grad = True)
    )
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad = True))
    W2 = nn.Parameter(
        torch.normal(0, 0.01, size = (num_hiddens, num_outputs), requires_grad = True)
    )
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad = True))
    params = [W1, b1, W2, b2]
    
    def relu(X):
        a = torch.zeros_like(X)
        return torch.max(X, a)

    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(X @ W1 + b1)
        return (H @ W2 + b2)

    loss = nn.CrossEntropyLoss()
    
    num_epochs = 10
    lr = 0.1

    updater = torch.optim.SGD(params, lr = lr)
    
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
        # End.

if __name__ == "__main__":
    main()