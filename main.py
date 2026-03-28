import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import timm
import random

# https://docs.pytorch.org/docs/stable/backends.html#torch.backends.cudnn.benchmark
torch.backends.cudnn.benchmark = True

# read about __init__.py and __main__.py

# CONFIG?
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Resizing input images to 299 x 299, default input size for Xception (CNN)
IMG_SIZE_OLD = (299, 299)
IMG_SIZE = 299
# Number of images trained at once
BATCH_SIZE = 32
NUM_EPOCHS = 1
DATA_PATH = "datasets/Dataset"


def read_data(data_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=4, subset=1.0):

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    loaders_dict = {}

    for split in ["Train", "Validation", "Test"]:
        path = os.path.join(data_dir, split)
        dataset = datasets.ImageFolder(path, transform=transform)

        # Train on a subset of images, ideally to speed up intermediate steps
        if subset < 1.0:
            subset_size = int(len(dataset) * subset)
            indices = random.sample(range(len(dataset)), subset_size)
            dataset = Subset(dataset, indices)

        shuffle = True if split == "Train" else False

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

        loaders_dict[split] = loader

    return loaders_dict["Train"], loaders_dict["Validation"], loaders_dict["Test"]

def train_model(num_classes=2, train_loader=None, val_loader=None, optimizer=None):
    model = timm.create_model("xception", pretrained=True, num_classes=num_classes)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier layer
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    model.to(DEVICE)

    # Maybe take out of function
    criterion = nn.CrossEntropyLoss()

    # Trains unfrozen layers
    if optimizer is None:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
              f"Training Loss: {avg_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total

def main():
    print("pytorch version:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    print(f"Using this: {DEVICE}")

    # CHANGE TO 1.0 OR REMOVE SUBSET WHEN READY
    train_loader, val_loader, test_loader = read_data(DATA_PATH, subset=0.1)

    print(f"training batches: {len(train_loader)}")
    print(f"validation batches: {len(val_loader)}")
    print(f"test batches: {len(test_loader)}")

    # Auto set number of classes based on folders
    train_dir = os.path.join(DATA_PATH, "Train")
    num_classes = len([name for name in os.listdir(train_dir)
                       if os.path.isdir(os.path.join(train_dir, name))])
    train_model(
        num_classes=num_classes,
        train_loader=train_loader,
        val_loader=val_loader
    )

    return

if __name__ == "__main__":
    main()