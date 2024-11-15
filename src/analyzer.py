import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans


class Analyzer:
    def __init__(self, model_path: str, device="cuda", max_clusters=6, threshold=0.3):
        self.device = device
        self.model = torch.load(model_path, map_location=torch.device(device))
        self.max_clusters = max_clusters
        self.threshold = threshold

    def preprocess_image(self, image, pad_value=0):
        height, width = image.shape[:2]
        pad_height = (32 - height % 32) % 32
        pad_width = (32 - width % 32) % 32

        pad_height1, pad_height2 = pad_height // 2, pad_height - pad_height // 2
        pad_width1, pad_width2 = pad_width // 2, pad_width - pad_width // 2

        image_padded = np.pad(
            image,
            ((pad_height1, pad_height2), (pad_width1, pad_width2), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )

        image_padded_float = image_padded / 255.0
        image_transposed = np.transpose(image_padded_float, (2, 0, 1)).astype("float32")

        return (
            image_transposed,
            (pad_height1, pad_height2, pad_width1, pad_width2),
            (height, width),
        )

    def predict(self, image_tensor):
        with torch.no_grad():
            x_tensor = torch.from_numpy(image_tensor).to(self.device).unsqueeze(0)
            pr_mask = self.model(x_tensor)
            pr_mask = pr_mask.squeeze().cpu().detach().numpy()
        return pr_mask

    def postprocess_mask(self, pr_mask, pad_info, original_size):
        pad_height1, _, pad_width1, _ = pad_info
        height, width = original_size

        pr_mask = pr_mask[
            :, pad_height1 : height + pad_height1, pad_width1 : width + pad_width1
        ]
        class_indices = np.argmax(pr_mask, axis=0)
        return class_indices

    def create_class_mask(self, class_indices, target_class):
        class_mask = np.zeros_like(class_indices, dtype=np.uint8)
        class_mask[class_indices == target_class] = 1
        return class_mask

    def find_last_above_threshold(self, data, threshold):
        for i in range(len(data) - 1, -1, -1):
            if data[i] < -1 * threshold:
                return i
        return 0

    def extract_colors(self, image, mask):
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        roi = masked_image.reshape((-1, 3))
        roi = roi[np.all(roi != 0, axis=1)]
        distortions = []
        dominant_colors = []
        for n_clusters in range(1, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=150).fit(
                roi
            )
            distortions.append(kmeans.inertia_)
            dominant_colors.append(kmeans.cluster_centers_)

        relative_changes = np.diff(distortions) / distortions[:-1]
        n_clusters = (
            self.find_last_above_threshold(relative_changes, self.threshold) + 1
        )

        return dominant_colors[n_clusters - 1], n_clusters

    def process(self, image, target_class=1):
        image_transposed, pad_info, original_size = self.preprocess_image(image)
        pr_mask = self.predict(image_transposed)
        class_indices = self.postprocess_mask(pr_mask, pad_info, original_size)
        class_mask = self.create_class_mask(class_indices, target_class)
        dominant_colors, optimal_k = self.extract_colors(image, class_mask)

        return dominant_colors, optimal_k, class_mask
