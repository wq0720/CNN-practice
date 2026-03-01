import torch
import torch.nn as nn
import torchvision # this is the main torchvision package, for datasets and transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torchvision import models

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay    
#=========define parameters===============
batch_size = 32 # batch size for training
num_classes = 2 # two classes: cardigans and jumpers
learning_rate = 0.001 # learning rate for optimizer
num_epochs = 20 # number of training epochs

#========= device will determine whether to run the trainning on GPU or CPU=============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#========= dataset loading==============
train_dir = "dataset_split/train"
val_dir   = "dataset_split/val"
test_dir  = "dataset_split/test"
#=========transformations===============
transform = transforms.Compose([ # use transforms.compose method to reformat images for modeling
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#=========load datasets==================
train_dataset = datasets.ImageFolder(train_dir, transform=transform) 
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


images, labels = next(iter(train_loader))
print("images shape:", images.shape)  # 应该是 [B, 3, 128, 128]
print("labels shape:", labels.shape)  # [B]
print("labels sample:", labels[:10])



# Compute class weights
from collections import Counter

targets = train_dataset.targets
class_counts = Counter(targets)

total = sum(class_counts.values())

weights = [
    total / class_counts[0],
    total / class_counts[1]
]

class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)


#==========cnn model from scratch==========
class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNeuralNet, self).__init__()
        self.relu_fc = nn.ReLU()

        #=======layer 1=======
        self.conv1 = nn.Conv2d(
            in_channels=3, # input channel: 3 for RGB images
            out_channels=16, # output channel: learning 16 features
            kernel_size=3, # filter size
            stride=1, # step size
            padding=1 # to keep spatial dimensions
        )
        self.relu1 = nn.ReLU() # activation function
        self.pool1 = nn.MaxPool2d(2, 2) # down-sampling

        #=======layer 2=======
        self.conv2 = nn.Conv2d(
            in_channels=16, # input channel: 16 from previous layer
            out_channels=32, # output channel: learning 32 features
            kernel_size=3, # filter size
            stride=1, # step size
            padding=1 # to keep spatial dimensions
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        #=======layer 3=======
        self.conv3 = nn.Conv2d(
            in_channels=32, # input channel: 32 from previous layer
            out_channels=64, # output channel: learning 64 features
            kernel_size=3, # filter size
            stride=1, # step size
            padding=1 # to keep spatial dimensions
        )
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 28 * 28, 512) # fully connected layer 1
        self.fc2 = nn.Linear(512, num_classes) # fully connected layer 2
#=========forward pass==========progresses data across layers
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x))) # layer 1 p(r(c(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = x.view(-1, 64 * 28 * 28) # flatten the tensor, making it suitable for the fully connected layers, one dimensional
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)
        return x

num_classes = 2
#model = ConvNeuralNet(num_classes).to(device)  
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True                        
# Set Loss function with criterion
#criterion = nn.CrossEntropyLoss() # this is a common loss function for classification problems, it combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights) 

# Set optimizer with optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  
optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

total_step = len(train_loader)
# Initialize lists to store training and validation metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
# We use the pre-defined number of epochs to determine how many iterations to train the network on
for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0
# Load in the data in batches using the train_loader object
    for images, labels in train_loader:  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record training loss
        running_loss += loss.item()
        running_acc += (outputs.argmax(dim=1) == labels).float().mean().item()
        n_batches += 1
    # Compute average losses and accuracies,使图像曲线更平滑
    #train_loss = running_loss / n_batches
    #train_acc = running_acc / n_batches
    # Record training loss and accuracy
    train_losses.append(running_loss / n_batches)
    train_accuracies.append(running_acc / n_batches)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    # Validate the model
    
    model.eval()
    
    with torch.no_grad():
        val_loss = 0.0
        val_acc = 0.0
        n_batches = 0
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.argmax(dim=1) == labels).float().mean().item()
            n_batches += 1
    val_losses.append(val_loss / n_batches)
    val_accuracies.append(val_acc / n_batches)

    
    print('Accuracy of the network on the {} train images: {} %'.format(1000, 100 * correct / total))
# make the plots
epochs = range(1, num_epochs + 1)



plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, val_accuracies, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.show()


# Collect all predictions and labels for the confusion matrix
all_preds = [] # Store all predicted labels
all_labels = [] # Store all true labels

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        all_preds.append(predicted.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    print(all_preds.shape, all_labels.shape)
    #class_names = test_dataset.classes
    class_names = val_dataset.classes

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
    disp.plot()
    plt.title(f"Confusion Matrix (N={len(all_labels)})")
    plt.show()



print(class_names)
