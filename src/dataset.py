from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BinaryImageDataset(Dataset):
    def __init__(self, split_dir, transform=None):
        self.transform = transform
        self.classes = ["fake", "real"]
        self.samples = []

        split_path = Path(split_dir)
        for class_dir in sorted(p for p in split_path.iterdir() if p.is_dir()):
            name = class_dir.name.lower()
            if name.startswith("fake"):
                label = 0
            elif name.startswith("real"):
                label = 1
            else:
                continue

            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(f"No images found in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_loaders(data_dir, batch_size=32):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    train_data = BinaryImageDataset(Path(data_dir) / "train", transform=transform)
    val_data = BinaryImageDataset(Path(data_dir) / "val", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader, train_data.classes
