import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score

# Set device to GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Normalize function
image_normalize = lambda x: x / 255.0

# Prepare MNIST dataset
full_train_set = datasets.MNIST(
    root='/files/',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(image_normalize)])
)
test_loader = DataLoader(
    datasets.MNIST(
        root='/files/',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(image_normalize)])
    ),
    batch_size=50,
    shuffle=True
)

# Split train set into train (90%) and validation (10%)
train_size = int(0.9 * len(full_train_set))
val_size = len(full_train_set) - train_size
train_set, val_set = random_split(full_train_set, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=50, shuffle=True)
val_loader = DataLoader(val_set, batch_size=50, shuffle=False)

# Set the learning rate and optimizer (Adam)
lr = 0.001

# Loss function
loss_fn = nn.CrossEntropyLoss()

def init_conv2d_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        m.weight = nn.Parameter(torch.abs(m.weight))
        m.bias.data.fill_(0.01)

# Training function with validation
def train_model(model, optimizer, train_loader, val_loader, num_epochs=11):
    model.to(device)

    start_time = time.time()  # Start time to measure training duration
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                running_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_train_loss = running_train_loss / len(train_loader)
        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = val_correct / total_val

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    end_time = time.time()  # End time
    print(f"\nTraining Time: {end_time - start_time:.2f} seconds\n")

# Evaluation (Prediction + Metrics Calculation)
def predict(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

# Metric calculation (Precision, Recall, F1, Accuracy, Balanced Accuracy)
def score(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
    }

# Additional code for calculating number of parameters in the models
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Architecture 1: Logistic Regression (No Hidden Layers)
model_lr = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)  # Output layer for 10 classes
)

optimizer_lr = torch.optim.Adam(model_lr.parameters(), lr=lr)


# Train the models
print("Training Logistic Regression (No Hidden Layers)...")
train_model(model_lr, optimizer_lr, train_loader, val_loader, num_epochs=11)


# Evaluate models
print("Evaluating Logistic Regression (No Hidden Layers)...")
y_true_lr, y_pred_lr = predict(model_lr, test_loader)
metrics_lr = score(y_true_lr, y_pred_lr)
print("\nLogistic Regression (No Hidden Layers) Metrics:")
for metric, value in metrics_lr.items():
    print(f"{metric.capitalize()}: {value}\n")


print(f"\nLogistic Regression (No Hidden Layers) Parameters: {count_parameters(model_lr)}")

# Architecture 2: Logistic Regression with two hidden layers (200 neurons each)
model_2hl = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 200),  # Hidden layer 1
    nn.ReLU(),            # Activation function ReLU
    nn.Linear(200, 200),  # Hidden layer 2
    nn.ReLU(),            # Activation function ReLU
    nn.Linear(200, 10)    # Output layer for 10 classes
)

optimizer_2hl = torch.optim.Adam(model_2hl.parameters(), lr=lr)

print("Training Logistic Regression with Two Hidden Layers...")
train_model(model_2hl, optimizer_2hl, train_loader, val_loader, num_epochs=11)

print("Evaluating Logistic Regression with Two Hidden Layers...")
y_true_2hl, y_pred_2hl = predict(model_2hl, test_loader)
metrics_2hl = score(y_true_2hl, y_pred_2hl)
print("\nLogistic Regression with Two Hidden Layers Metrics:")
for metric, value in metrics_2hl.items():
    print(f"{metric.capitalize()}: {value}\n")

print(f"Logistic Regression with Two Hidden Layers Parameters: {count_parameters(model_2hl)}")

# Architecture 3: Convolutional Neural Network (CNN) with one convolutional layer
model_cnn = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding='same'),     # Convolution layer: 32 filters of 5x5, stride=1, padding='same'
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),     # Max pooling: 2x2 kernel with stride=2
    nn.Flatten(),     # Flatten the output of the convolutional layers
    # Fully connected layer with 1024 neurons
    nn.Linear(in_features=32 * 14 * 14, out_features=1024),  # 32 filters * (28/2) * (28/2) = 32 * 14 * 14
    nn.ReLU(),
    nn.Linear(in_features=1024, out_features=10)     # Output layer: 10 classes (for MNIST)
)

model_cnn.apply(init_conv2d_weights)
optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=lr)

print("Training CNN (1 Convolutional Layer)...")
train_model(model_cnn, optimizer_cnn, train_loader, val_loader, num_epochs=11)

print("Evaluating CNN (1 Convolutional Layer)...")
y_true_cnn, y_pred_cnn = predict(model_cnn, test_loader)
metrics_cnn = score(y_true_cnn, y_pred_cnn)
print("\nCNN (1 Convolutional Layer) Metrics:")
for metric, value in metrics_cnn.items():
    print(f"{metric.capitalize()}: {value}\n")

print(f"\nCNN (1 Convolutional Layer) Parameters: {count_parameters(model_cnn)}")

# Architecture 4: Convolutional Neural Network (CNN) with two convolutional layers
model_cnn2 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding='same'),     # Convolution layer: 32 filters of 5x5, stride=1, padding='same'
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),     # Max pooling: 2x2 kernel with stride=2
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),     # Convolution layer: 64 filters of 5x5, stride=1, padding='same'
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),     # Max pooling: 2x2 kernel with stride=2
    nn.Flatten(),     # Flatten the output of the convolutional layers
    # Fully connected layer with 1024 neurons
    nn.Linear(in_features=64 * 7 * 7, out_features=1024),  # 64 filters * (28/4) * (28/4) = 64 * 7 * 7
    nn.ReLU(),
    nn.Linear(in_features=1024, out_features=10)     # Output layer: 10 classes (for MNIST)
)

model_cnn2.apply(init_conv2d_weights)
optimizer_cnn2 = torch.optim.Adam(model_cnn2.parameters(), lr=lr)

print("Training CNN (2 Convolutional Layer)...")
train_model(model_cnn2, optimizer_cnn2, train_loader, val_loader, num_epochs=11)

print("Evaluating CNN (2 Convolutional Layer)...")
y_true_cnn2, y_pred_cnn2 = predict(model_cnn2, test_loader)
metrics_cnn2 = score(y_true_cnn2, y_pred_cnn2)
print("\nCNN (2 Convolutional Layer) Metrics:")
for metric, value in metrics_cnn2.items():
    print(f"{metric.capitalize()}: {value}\n")

print(f"\nCNN (2 Convolutional Layer) Parameters: {count_parameters(model_cnn2)}")

#Compare 50/100 minibatches

dropout_rate = 0.5
lr = 0.001  # Learning rate

test_loader_100 = DataLoader(
    datasets.MNIST(
        root='/files/',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(image_normalize)])
    ),
    batch_size=100,
    shuffle=True
)

# Define weight initialization function
def init_conv2d_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)  # He initialization
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Training function
def train_model_until_accuracy(model, optimizer, train_loader, val_loader, target_accuracy=0.99, num_epochs=5):
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        # Validation phase
        model.eval()
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_accuracy = val_correct / total_val
        print(f"Epoch {epoch + 1}, Train Loss: {running_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Stop if target accuracy is reached
        if val_accuracy >= target_accuracy:
            end_time = time.time()
            print(f"Reached {target_accuracy:.2f} validation accuracy in {epoch + 1} epochs.")
            print(f"Training Time: {end_time - start_time:.2f} seconds\n")
            return epoch + 1

    print(f"Did not reach {target_accuracy:.2f} validation accuracy within {num_epochs} epochs.")
    return num_epochs

# Train and evaluate model with batch_size=50
print("Training model_cnn3 with batch size 50...")
train_loader_50 = DataLoader(train_set, batch_size=50, shuffle=True)
val_loader_50 = DataLoader(val_set, batch_size=50, shuffle=False)

model_cnn3_50 = nn.Sequential(
    nn.Conv2d(1, 32, (5, 5), stride=1, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), stride=2),
    nn.Conv2d(32, 64, (5, 5), stride=1, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), stride=2),
    nn.Flatten(),
    nn.Dropout(p=dropout_rate),
    nn.Linear(64 * 7 * 7, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10)
)

model_cnn3_50.apply(init_conv2d_weights)
optimizer_50 = torch.optim.Adam(model_cnn3_50.parameters(), lr=lr)

# Train the model with batch_size=50
train_model_until_accuracy(model_cnn3_50, optimizer_50, train_loader_50, val_loader_50, target_accuracy=0.99, num_epochs=1000)

print("Evaluating CNN (2 Convolutional Layer 50 batch)...")
y_true_cnn, y_pred_cnn = predict(model_cnn3_50, test_loader)
metrics_cnn3_50 = score(y_true_cnn, y_pred_cnn)
print("\nCNN (2 Convolutional Layer 50 batch) Metrics:")
for metric, value in metrics_cnn3_50.items():
    print(f"{metric.capitalize()}: {value}\n")

print(f"\nCNN (2 Convolutional Layer 50 batch) Parameters: {count_parameters(model_cnn3_50)}")

# Train and evaluate model with batch_size=100
print("Training model_cnn3 with batch size 100...")
train_loader_100 = DataLoader(train_set, batch_size=100, shuffle=True)
val_loader_100 = DataLoader(val_set, batch_size=100, shuffle=False)

model_cnn3_100 = nn.Sequential(
    nn.Conv2d(1, 32, (5, 5), stride=1, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), stride=2),
    nn.Conv2d(32, 64, (5, 5), stride=1, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), stride=2),
    nn.Flatten(),
    nn.Dropout(p=dropout_rate),
    nn.Linear(64 * 7 * 7, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10)
)

model_cnn3_100.apply(init_conv2d_weights)
optimizer_100 = torch.optim.Adam(model_cnn3_100.parameters(), lr=lr)

# Train the model with batch_size=100
train_model_until_accuracy(model_cnn3_100, optimizer_100, train_loader_100, val_loader_100, target_accuracy=0.99, num_epochs=1000)

print("Evaluating CNN (2 Convolutional Layer 100 batch)...")
y_true_cnn, y_pred_cnn = predict(model_cnn3_100, test_loader_100)
metrics_cnn3_100 = score(y_true_cnn, y_pred_cnn)
print("\nCNN (2 Convolutional Layer 100 batch) Metrics:")
for metric, value in metrics_cnn3_100.items():
    print(f"{metric.capitalize()}: {value}\n")

print(f"\nCNN (2 Convolutional Layer 100 batch) Parameters: {count_parameters(model_cnn3_100)}")

def init_conv2d_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        m.weight = nn.Parameter(torch.abs(m.weight))
        m.bias.data.fill_(0.01)

# Train and evaluate model with batch_size=50
print("Training model_cnn3 with batch size 50...")
train_loader_50 = DataLoader(train_set, batch_size=50, shuffle=True)
val_loader_50 = DataLoader(val_set, batch_size=50, shuffle=False)

model_cnn3_50.apply(init_conv2d_weights)
optimizer_50 = torch.optim.Adam(model_cnn3_50.parameters(), lr=lr)

# Train the model with batch_size=50
train_model_until_accuracy(model_cnn3_50, optimizer_50, train_loader_50, val_loader_50, target_accuracy=0.99)

print("Evaluating CNN (2 Convolutional Layer 50 batch)...")
y_true_cnn, y_pred_cnn = predict(model_cnn3_50, test_loader)
metrics_cnn3_50 = score(y_true_cnn, y_pred_cnn)
print("\nCNN (2 Convolutional Layer 50 batch) Metrics:")
for metric, value in metrics_cnn3_50.items():
    print(f"{metric.capitalize()}: {value}\n")


print(f"\nCNN (2 Convolutional Layer 50 batch) Parameters: {count_parameters(model_cnn3_50)}")


# Define a function to visualize filter weights
def visualize_filters(model, layer_num, num_filters=5):
    # Get the weights of the specified layer
    layer = model[layer_num]
    if isinstance(layer, nn.Conv2d):
        weights = layer.weight.data.cpu().numpy()

        # Plot the first 'num_filters' filters
        fig, axes = plt.subplots(1, num_filters, figsize=(15, 15))
        for i in range(num_filters):
            axes[i].imshow(weights[i, 0, :, :], cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'Filter {i+1}')
        plt.show()


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


# Define a function to visualize activations
def image_conv(model, batch, layer_num, channel_num, activation_after=True):
    # Register hooks to capture activations
    model[layer_num].register_forward_hook(get_activation(f'layer_{layer_num}'))
    # Forward pass through the model
    model(batch)
    # Get the activation
    act = activation[f'layer_{layer_num}']
    # Plot the original image and the convolution image side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Display the original image
    original_image = batch[0][0].cpu().numpy()  # Assuming single-channel input (e.g., MNIST)
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    # Display the activation map
    activation_map = act[0][channel_num].cpu().numpy()
    axes[1].imshow(activation_map, cmap='viridis')
    axes[1].set_title(f'Layer {layer_num}, Channel {channel_num}, Activation After: {activation_after}')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


# Load a sample image from the test_loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x / 255.0)])
test_loader = DataLoader(datasets.MNIST(root='/files/', train=False, download=True, transform=transform), batch_size=1,
                         shuffle=True)
sample_image, _ = next(iter(test_loader))
sample_image = sample_image.to(device)

# Visualize 5 different filters from the first convolutional layer
visualize_filters(model_cnn3_50, 0)

# Visualize convolution images before and after ReLU for each layer
image_conv(model_cnn3_50, sample_image, 0, 0, activation_after=False)
image_conv(model_cnn3_50, sample_image, 1, 0, activation_after=True)
image_conv(model_cnn3_50, sample_image, 3, 0, activation_after=False)
image_conv(model_cnn3_50, sample_image, 4, 0, activation_after=True)