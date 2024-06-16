import torch
from torch import nn

from SNGPModule import SNGPModule
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


class NeuralNetworkSNGP(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNetworkSNGP, self).__init__()
    # Define hidden layers (replace with your desired architecture)
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    # SNGP layer with relevant parameters
    self.sngp = SNGPModule(hidden_size, num_classes, 0.1, 1.0, 1.0, 1.0, 5, torch.device("cpu"))

  def forward(self, x):
    # Pass through hidden layers
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    # Pass to SNGP layer for prediction
    logits = self.sngp(x)
    return logits


num_input_features = 10
num_target_features = 14

# Each sample (100 samples) has 50 timesteps and 2 features
data = torch.randn(100, 50, num_input_features)
# Labels for each time series
labels = torch.randint(0, num_target_features, size=(100,))

dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = NeuralNetworkSNGP(num_input_features, 64, num_target_features)  # Assuming 2 features and 3 classes

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
  for i, (timeseries, labels) in enumerate(dataloader):
    optimizer.zero_grad()
    logits = model(timeseries)

    labels = F.one_hot(labels, num_classes=num_target_features)  # One-hot encode labels with 3 classes
    labels = labels.view(labels.size(0), num_target_features)  # Reshape to [batch_size, num_classes]
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    # Print training progress (optional)
    if i % 50 == 0:
      print(f"Epoch: {epoch+1}, Step: {i}, Loss: {loss.item():.4f}")

new_data = torch.randn(10, 50, num_input_features)
prediction = model(new_data).argmax(dim=1)
print(prediction)
for p in prediction:
    predicted_class = p.argmax(dim=0).item()
    print(f"Predicted class for new data: {predicted_class}")


