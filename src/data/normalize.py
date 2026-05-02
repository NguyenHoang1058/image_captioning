from PIL import Image
import numpy as np

def process_image(image, size=(224, 224)):
    image = image.resize(size)

    image = np.array(image)

    image = image / 255.0

    return image

def normalize_images(data):
    new_data = []

    for item in data:
        new_item = item.copy()

        image = new_item["image"]
        image = process_image(image)

        new_item["image"] = image
        new_data.append(new_item)

    return new_data