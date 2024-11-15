import base64
import cv2
import numpy as np

def byte2Numpy(data, rgb=True):
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)[:, :, 0:3]
    if rgb:
        frame = frame[:, :, ::-1]
    return frame

def numpy2base64(np_array):
    _, buffer = cv2.imencode(".jpg", np_array)
    byte_data = buffer.tobytes()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str

def resize_with_aspect_ratio(image, target_size: tuple = (864, 864)):
    target_height, target_width = target_size
    height, width = image.shape[:2]

    
    target_aspect_ratio = target_width / target_height
    aspect_ratio = width / height

    if aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def make_palette(image, dominant_colors, optimal_k):
    h, w, _ = image.shape
    color_len = w // optimal_k
    palette = np.zeros((100, w, 3), dtype=np.uint8)

    for i, color in enumerate(dominant_colors):
        palette[:, i * color_len : (i + 1) * color_len, :] = color

    combined_image = np.concatenate((image, palette), axis=0)
    return combined_image

