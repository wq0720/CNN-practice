import torch
import torch.nn as nn

from CNN import simplecnn

model = simplecnn(num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(model)
x = torch.randn(1, 3, 32, 32)
output = model(x)
print(output.shape)

