from selenium import webdriver
import warnings
import time
import json
from io import BytesIO
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import re
import requests

warnings.filterwarnings('ignore')
driver_path = r'C:\Users\Akash Gupta\Downloads\chromedriver_win32\chromedriver'
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
    image = read_image(requests.get(url).content)
    text = annotate_image(image)
    return text


def clean_image_url(url):
    if '?crop' in url:
        url = url.split('?crop', 1)[0]
    elif '?itok' in url:
        url = url.split('?itok', 1)[0]
    elif '?quality' in url:
        url = url.split('?quality', 1)[0]
    elif '?format' in url:
        url = url.split('?format', 1)[0]
    elif '?ixlib' in url:
        url = url.split('?ixlib', 1)[0]
    elif '?q' in url:
        url = url.split('?q', 1)[0]
    elif '?width' in url:
        url = url.split('?width', 1)[0]
    elif '?fit' in url:
        url = url.split('?fit', 1)[0]
    elif '?mode' in url:
        url = url.split('?mode', 1)[0]
    elif '?s' in url:
        url = url.split('?s', 1)[0]
    elif '?w' in url:
        url = url.split('?w', 1)[0]
    return url


def fetch_image_urls(query, max_links_to_fetch, wd: webdriver, sleep_between_interactions=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))
    image_urls = set()
    image_dict = {}
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)
        # get all image thumbnail results
        text = wd.find_elements_by_css_selector
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                image_link = actual_image.get_attribute('src')
                image_link = clean_image_url(image_link)
                if image_link and 'http' in image_link and image_link not in image_urls:
                    image_dict[image_link] = []
                    # annotate the image
                    try:
                        #annotated_text = get_text_from_image(image_link)
                        image_urls.add(image_link)
                        image_dict[image_link].append(actual_image.get_attribute('alt'))
                        #image_dict[image_link].append(annotated_text)
                    except Exception:
                        continue

            image_count = len(image_urls)
            if len(image_urls) >= max_links_to_fetch:
                break

        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(30)
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result start point further down
        results_start = len(thumbnail_results)
    return image_dict


def search_and_download(search_term, driver_path, number_images=30):
    with webdriver.Chrome(executable_path=driver_path) as wd:
        res = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=0.5)
    return res


keywords = ["dog", "cat", "lion", "tiger", "horse", "ireland", "dublin", "whale", "elephant", "computers", "technology",
            "monkey", "real madrid", "football", "FIFA", "italy", "germany", "england", "apple", "mango",
            "buildings", "DCU", "batman", "superman", "ironman", "marvel", "airplane", "ronaldo", "messi", "pisa",
            "france", "war", "tanks", "cow", "church", "temples", "books", "animals", "food", "tables", "chairs",
            "einstein", "tree", "cars", "snakes", "leopard", "cricket"]


def generate_dataset():
    dataset = {}
    for i in range(0, len(keywords)):
        dataset.update(search_and_download(search_term=keywords[i], driver_path=driver_path))
    return dataset


image_dataset = generate_dataset()
with open('image-collection.json', 'w') as fp:
    json.dump(image_dataset, fp)


