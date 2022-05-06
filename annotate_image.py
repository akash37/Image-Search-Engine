import json
from io import BytesIO
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import re
import requests

model = tf.keras.applications.Xception(weights="imagenet")


def read_image(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


def annotate_image(image: Image.Image):
    image = np.asarray(image.resize((299, 299))) [..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0
    result = decode_predictions(model.predict(image), 1)[0]
    return re.sub('[^a-zA-Z0-9]+', ' ', result[0][1])


def get_text_from_image(url):
    image = read_image(requests.get(url, timeout=5).content)
    text = annotate_image(image)
    return text


with open("annotated-image-collection.json") as f:
    data = json.loads(f.read())

print(len(data))
