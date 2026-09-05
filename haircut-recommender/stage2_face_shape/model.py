import torch
import torch.nn as nn


class FaceShapeMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for Face Shape Classification.

    Architecture:
      - Input Layer -> Linear(input_size, hidden_size)
      - Activation  -> ReLU()
      - Output Layer -> Linear(hidden_size, num_classes) (raw logits)
    """
    
    def __init__(self, input_size=4, hidden_size=32, num_classes=5, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass:
          input -> first linear layer -> ReLU -> second linear layer -> return logits
        """
        x = self.fc1(x)
        x = self.relu(x)
        x=self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x=self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x=self.dropout(x)
        x = self.fc4(x)
        return x


if __name__ == "__main__":
    model = FaceShapeMLP(input_size=4, hidden_size=32, num_classes=5)
    dummy_input = torch.randn(1, 4)   # simulate one sample with 4 features
    output = model(dummy_input)
    print(output.shape)   # should print: torch.Size([1, 5])
