import torch

class InterfaceA(torch.nn.Module):
    """A simple interface to build a simple neural network learner.
       It has a lot of bugs inside. It is inconvinient because it is not flexible 
    """
    def __init__(self, input_dim: int = 100, output_dim: int = 1) -> None:
        super().__init__()
        self.hidden_dim = 52
        self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, self.hidden_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.BatchNorm1d(self.hidden_dim),
                                        torch.nn.Linear(self.hidden_dim, output_dim))
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr = 3e-4)
        self.loss = torch.nn.MSELoss()

    def forward(self, x: torch.TensorType, y:torch.TensorType):
        predictions = self.net(x)
        loss = self.loss(predictions, y)
        return loss

    def learn(self, n_epochs: int, loader: torch.utils.data.DataLoader):
        for epoch in range(n_epochs):
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                loss = self.forward(batch_x, batch_y)
                loss.backward()
                self.optimizer.step()
            print(f"epoch {epoch} loss = {loss.detach().item()}")

def test():
    X = torch.rand(size = (100, 100))
    y = torch.rand(size = (100, 1))
    train_dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10)

    learner = InterfaceA(100, 1)
    learner.hidden_dim = 34
    learner.learn(n_epochs=10, loader=train_loader)

if __name__ == "__main__": test()

