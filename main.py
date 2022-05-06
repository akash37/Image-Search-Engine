from flask import Flask, jsonify
from flask import request
from flask_cors import CORS, cross_origin
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

text_model = SentenceTransformer('all-mpnet-base-v2')
image_dataset = pd.read_csv("image-dataset.csv")
text_embeddings = text_model.encode(image_dataset["text"])

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
CORS(app)


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/index', methods=['GET'])
#@cross_origin(supports_credentials=True)
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    query = request.args.get('search')
    #print(type(query))
    ans = get_relevant_image_url(query)
    finalList = []
    for k, v in ans.items():
        finalList.append(k)
    data = {'image_url': finalList}
    print(ans)
    return jsonify(data)


def get_relevant_image_url(query):
    query_embeddings = text_model.encode(query)
    ans = {}
    for i in range(0, len(text_embeddings)):
        score = cosine_similarity([text_embeddings[i]], [query_embeddings])[0][0]
        if score > 0.5:
            ans[image_dataset["image_url"][i]] = score
    return dict(sorted(ans.items(), key=lambda item: item[1], reverse=True))


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(debug=True)
