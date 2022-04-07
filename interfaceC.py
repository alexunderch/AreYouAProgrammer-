from .interfaceA import InterfaceA
from .InterfaceB import InterfaceB
import torch
class InterfaceC(object):
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def load_data(self):
        """Loading Data method. Should always return datasets"""
        dataset = InterfaceB(self.batch_size)
        train_set, test_set = dataset.preprocess()
        return train_set, test_set
    
    def learn(self, input_dim, hidden_dim, output_dim, n_epochs: int):
        """Learning method"""
        train_dataset, _ = self.load_data()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10)

        learner = InterfaceA(input_dim, output_dim)
        learner.hidden_dim = hidden_dim
        learner.learn(n_epochs=n_epochs, loader=train_loader)
    
    def validate(self):
        """Oops!"""

    def test(self):
        """Oops!"""
    
    def show_to_boss(self):
        """Oops!"""


def main():
    interface = InterfaceC(batch_size=20)
    interface.learn()

    interface.show_to_boss()


if __name__ == "__main__": main()