import json
import numpy as np
import requests
from PIL import Image
from io import BytesIO

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)



def load_image_from_url(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    return img

def preprocess_image(image, target_size=(224, 224)):
    image_resized = image.resize(target_size)
    image_array = np.array(image_resized)
    return np.expand_dims(image_array / 255., axis=0)  # Normalize and add batch dimension

def get_image_prediction(image_url, api_url):

    image = load_image_from_url(image_url)

    preprocessed_image = preprocess_image(image)

    data = preprocessed_image.tolist()

    response = requests.post(api_url, json={'data': data})

    print("API response:", response.text)  # Print the response text
    # Return the response JSON
    return response.json()



