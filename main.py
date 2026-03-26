import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

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


def read_data(data_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=4):

    # Resize, convert to tensor, normalize [-1,1]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    datasets_dict = {}
    loaders_dict = {}

    for split in ["Train", "Validation", "Test"]:
        path = os.path.join(data_dir, split)
        ds = datasets.ImageFolder(path, transform=transform)
        shuffle = True if split == "Train" else False
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

        datasets_dict[split] = ds
        loaders_dict[split] = loader

    return loaders_dict["Train"], loaders_dict["Validation"], loaders_dict["Test"]

def main():
    print("pytorch version:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0))

    print(f"Using this: {DEVICE}")

    train_loader, val_loader, test_loader = read_data("datasets/Dataset")

    print(f"training batches: {len(train_loader)}")
    print(f"validation batches: {len(val_loader)}")
    print(f"test batches: {len(test_loader)}")

    return

if __name__ == "__main__":
    main()