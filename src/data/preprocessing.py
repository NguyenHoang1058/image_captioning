import string
import re
from PIL import Image
import matplotlib.pyplot as plt
from src.data.feature_engineering import encode_dataset
from src.data.normalize import normalize_images

def clean_text(text):
    text=text.lower()
    text=text.translate(str.maketrans('', '', string.punctuation))
    return text

def clean_caption(caption):
    # to lowercase
    caption=caption.lower()

    # remove special char and number
    caption=re.sub(r"[^a-z0-9\s]", "", caption)

    # remove white space
    caption = re.sub(r"\s+", " ", caption).strip()

    return caption

def clean_captions(data):
    new_data = []

    for item in data:
        new_item = item.copy()

        captions = new_item["caption"]

        if isinstance(captions, list):
            new_item["caption"] = [clean_caption(c) for c in captions]
        else:
            new_item["caption"] = clean_caption(captions)

        new_data.append(new_item)

    return new_data
 
def clean_data(data):
    data=clean_captions(data)
    #data = remove_invalid(data)

    return data

def show_cleaning_exsample(data, index=0):
    item=data[index]

    image=item["image"]
    captions=item["caption"]

    if isinstance(captions, list):
        original_caption = captions[0]
    else:
        original_caption = captions

    cleaned_caption = clean_caption(original_caption)

    plt.imshow(image)
    plt.axis("off")

    plt.title(
        f"Before: {original_caption}\n\nAfter: {cleaned_caption}",
        fontsize=10
    )

    plt.show()

# Hàm xử lý missing value
def remove_invalid(data):
    new_data = []

    for item in data:
        new_item = item.copy()
        captions = new_item["caption"]

        # xử lý caption
        if isinstance(captions, list):
            captions = [c for c in captions if c is not None and c.strip() != ""]
            if len(captions) == 0:
                continue
            new_item["caption"] = captions
        else:
            if captions is None or captions.strip() == "":
                continue

        new_data.append(new_item)

    return new_data


def prepare_data(data, word2idx):
    # 1. clean caption + remove invalid
    data = clean_data(data)

    # 2. encode caption → số
    data = encode_dataset(data, word2idx)

    return data