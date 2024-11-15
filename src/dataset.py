import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class SemanticSegmentationDataset(Dataset):
    CLASSES = ["background", "flower", "plant"]

    def __init__(self, data_dir, classes=None, augmentation=None, preprocessing=None):
        self.target_size = (864, 864)
        self.data_dir = data_dir
        # 假設圖像和標籤都在同一目錄下
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.images = [file for file in os.listdir(data_dir) if file.endswith(".jpg")]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.images[idx])
        label_name = os.path.join(
            self.data_dir, self.images[idx].replace(".jpg", "_mask.png")
        )

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

        # Pad image and label to 840x840
        image, label = self.pad_to_target_size(image, label)

        masks = [(label == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1)

        image = image / 255.0
        image = np.transpose(image, (2, 0, 1)).astype("float32")
        mask = np.transpose(mask, (2, 0, 1)).astype("float32")

        return image, mask

    def pad_to_target_size(self, image, label):
        target_height, target_width = self.target_size
        height, width = image.shape[:2]

        pad_height = max(0, target_height - height)
        pad_width = max(0, target_width - width)

        pad_height1, pad_height2 = pad_height // 2, pad_height - pad_height // 2
        pad_width1, pad_width2 = pad_width // 2, pad_width - pad_width // 2

        image_padded = np.pad(
            image,
            ((pad_height1, pad_height2), (pad_width1, pad_width2), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        label_padded = np.pad(
            label,
            ((pad_height1, pad_height2), (pad_width1, pad_width2)),
            mode="constant",
            constant_values=self.class_values[0],
        )

        return image_padded, label_padded


def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.savefig("temp.png")
