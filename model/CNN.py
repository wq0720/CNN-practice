import torch.nn as nn
import inspect

print(">>> CNN.py loaded from:", __file__)

class simplecnn(nn.Module):
    def __init__(self, num_classes=10):
        super(simplecnn, self).__init__()
        print(">>> simplecnn __init__ from:", inspect.getfile(simplecnn))

        # ===== 只保留这一套 feature，别加其他 Conv2d 了 =====
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # 32x32 -> 16x16

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # 16x16 -> 8x8
        )

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feature(x)          # (N,32,8,8)
        x = x.view(x.size(0), -1)    # (N,32*8*8)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
