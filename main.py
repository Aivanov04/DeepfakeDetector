import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import timm
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch.nn.functional as F
import time

# https://docs.pytorch.org/docs/stable/backends.html#torch.backends.cudnn.benchmark
torch.backends.cudnn.benchmark = True

# read about __init__.py and __main__.py

# CONFIG... use nvidia-smi to check GPU activity
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Resizing input images to 299 x 299, default input size for Xception (CNN)
IMG_SIZE_OLD = (299, 299)
IMG_SIZE = 299
# Number of images trained at once
BATCH_SIZE = 32
NUM_EPOCHS = 2
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
        # Track time taken per epoch
        start_time = time.time()

        model.train()
        running_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_acc, val_prec, val_rec, val_f1, val_auc, val_fpr, val_fnr, val_cm = evaluate(model, val_loader)

        # End time taken
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}\n"
              f"Loss: {avg_loss:.4f}\n"
              f"Acc: {val_acc:.4f}\n"
              f"F1: {val_f1:.4f}\n"
              f"AUC: {val_auc:.4f}\n"
              f"FPR: {val_fpr:.4f}\n"
              f"FNR: {val_fnr:.4f}\n"
              f"Time: {epoch_time:.2f}s\n")

    return model

def evaluate(model, loader):
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)

            probs = F.softmax(outputs, dim=1)[:, 1]  # probability of class 1 ("fake")
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_predictions)

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    return acc, precision, recall, f1, auc, fpr, fnr, cm

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

    model = train_model(
        num_classes=num_classes,
        train_loader=train_loader,
        val_loader=val_loader
    )

    print("\nFinal Test Evaluation")

    test_acc, test_prec, test_rec, test_f1, test_auc, test_fpr, test_fnr, test_cm = evaluate(model, test_loader)

    print("\nConfusion Matrix")
    print(test_cm)

    print(f"Test Acc: {test_acc:.4f}\n"
          f"Precision: {test_prec:.4f}\n"
          f"Recall: {test_rec:.4f}\n"
          f"F1: {test_f1:.4f}\n"
          f"AUC: {test_auc:.4f}\n"
          f"FPR: {test_fpr:.4f}\n"
          f"FNR: {test_fnr:.4f}")

if __name__ == "__main__":
    main()