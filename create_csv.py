import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import json

text_model = SentenceTransformer('all-mpnet-base-v2')

# creates the csv 
# df = pd.DataFrame(columns=["image_url", "text"])
#
# with open("annotated-image-collection.json") as f:
#     data = json.loads(f.read())
#
#
# for image_url, value in data.items():
#     text = ""
#     if len(value[0].split(" ")) > 10:
#         text = " ".join(value[0].split(" ")[0:9])
#     else:
#         text = value[0]
#
#     if len(value) > 1:
#         text = text + " " + value[1]
#     kvp = {'image_url': image_url, 'text': text}
#     df = df.append(kvp, ignore_index=True)
#
# df.to_csv("image-dataset.csv", index=False)

image_dataset = pd.read_csv("image-dataset.csv")
text_embeddings = text_model.encode(image_dataset["text"])
print(text_embeddings)
print()
print()


def get_relevant_image_url(query):
    query_embeddings = text_model.encode(query)
    print(query_embeddings)
    for i in range(0, len(text_embeddings)):
        score = cosine_similarity([text_embeddings[i]], [query_embeddings])[0][0]
        if score > 0.5:
            print(image_dataset["image_url"][i])


get_relevant_image_url("golden retriever")
