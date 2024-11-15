from src.analyzer import Analyzer
from tqdm import tqdm
import yaml
import os
import cv2
import csv
from src.utils import resize_with_aspect_ratio, make_palette


def main(analyzer: Analyzer, root_dir: str, save_dir: str, default_size = (864, 864)):
    csv_output = []
    for subdir in tqdm(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            output_dir = os.path.join(save_dir, subdir)
            os.makedirs(output_dir, exist_ok=True)
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                if os.path.isfile(file_path) and file.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    try:
                        resized_image = resize_with_aspect_ratio(image, default_size)
                        dominant_colors, optimal_k, _ = analyzer.process(resized_image)
                        result = make_palette(resized_image, dominant_colors, optimal_k)
                        output_path = os.path.join(output_dir, "output_" + file)
                        cv2.imwrite(
                            output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                        )
                        row = [subdir, file]
                        for color in dominant_colors:
                            row.append(
                                f"{int(color[0])},{int(color[1])},{int(color[2])}"
                            )
                        csv_output.append(row)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    with open("output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dir",
                "filename",
                "dominant_color1",
                "dominant_color2",
                "dominant_color3",
                "dominant_color4",
            ]
        )
        writer.writerows(csv_output)


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml", "r"))
    model_path = config["model_path"]
    default_size = config["default_image_size"]
    device = os.getenv("DEVICE", config["device"])

    analyzer = Analyzer(model_path, device=device)
    main(analyzer, "/mnt/e/orchid_production", "/mnt/e/orchid_output", default_size)
