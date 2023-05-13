from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
import json
import base64
from tensorflow.keras.optimizers import Adam


app = Flask(__name__)
api = Api(app)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


requests_count = {}

@app.before_request
def count_requests():
    path = request.path
    if path not in requests_count:
        requests_count[path] = 0
    requests_count[path] += 1




treatment_links = {
"Apple Scab": "https://www2.gov.bc.ca/gov/content/industry/agriculture-seafood/animals-and-crops/plant-health/insects-and-plant-diseases/tree-fruits/apple-scab",
"Apple Black Rot": "https://extension.umn.edu/plant-diseases/black-rot-apple",
"Apple Cedar Rust": "https://extension.umn.edu/plant-diseases/cedar-apple-rust",
"Apple Healthy": "https://www.almanac.com/plant/apples",
"Blueberry Healthy": "https://extension.umn.edu/fruit/growing-blueberries-home-garden",
"Cherry Healthy": "https://www.almanac.com/plant/cherries",
"Cherry Powdery Mildew": "https://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew/",
"Corn Cercospora": "https://extension.umn.edu/corn-pest-management/gray-leaf-spot-corn",
"Corn Common Rust": "https://extension.umn.edu/corn-pest-management/common-rust-corn",
"Corn Healthy": "https://www.almanac.com/plant/corn",
"Corn Northern Leaf Blight": "https://extension.umn.edu/corn-pest-management/northern-corn-leaf-blight",
"Grape Black Rot": "https://www.gardeningknowhow.com/edible/fruits/grapes/black-rot-grape-treatment.htm",
"Grape Esca": "https://grapes.extension.org/grapevine-measles/",
"Grape Healthy": "https://www.rhs.org.uk/fruit/grapes/grow-your-own",
"Grape Leaf Blight": "https://www.planthealthaustralia.com.au/wp-content/uploads/2013/11/Bacterial-blight-of-grapevine-FS.pdf",
"Orange Haunglongbing": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8636133/",
"Peach Bacterial Spot": "https://www.aces.edu/blog/topics/crop-production/bacterial-spot-treatment-in-peaches/",
"Peach Healthy": "https://www.masterclass.com/articles/how-to-grow-a-peach-tree-in-your-backyard",
"Pepper Bacterial Spot": "https://extension.wvu.edu/lawn-gardening-pests/plant-disease/fruit-vegetable-diseases/bacterial-leaf-spot-of-pepper",
"Pepper Healthy": "https://bonnieplants.com/blogs/how-to-grow/growing-peppers",
"Potato Early Blight": "https://www.planetnatural.com/pest-problem-solver/plant-disease/early-blight/",
"Potato Healthy": "https://www.gardendesign.com/vegetables/potatoes.html",
"Potato Late Blight": "https://www.planetnatural.com/pest-problem-solver/plant-disease/late-blight/",
"Raspberry Healthy": "https://www.rhs.org.uk/fruit/raspberries/grow-your-own",
"Soybean Healthy": "https://www.rhs.org.uk/vegetables/soya-beans/grow-your-own",
"Squash Powdery Mildew": "https://www.gardeningknowhow.com/edible/vegetables/squash/powdery-mildew-in-squash.htm",
"Strawberry Healthy": "https://www.rhs.org.uk/fruit/strawberries/grow-your-own",
"Strawberry Leaf Scorch": "https://www.gardeningknowhow.com/edible/fruits/strawberry/strawberries-with-leaf-scorch.htm",
"Tomato Bacterial Spot": "https://extension.umn.edu/disease-management/bacterial-spot-tomato-and-pepper",
"Tomato Early Blight": "https://extension.umn.edu/disease-management/early-blight-tomato-and-potato",
"Tomato Healthy": "https://www.rhs.org.uk/vegetables/tomatoes/grow-your-own",
"Tomato Late Blight": "https://extension.wvu.edu/lawn-gardening-pests/plant-disease/fruit-vegetable-diseases/late-blight-tomatoes",
"Tomato Leaf Mold": "https://www.lovethegarden.com/uk-en/article/tomato-leaf-mould",
"Tomato Septoria Leaf Spot": "https://extension.wvu.edu/lawn-gardening-pests/plant-disease/fruit-vegetable-diseases/septoria-leaf-spot",
"Tomato Spider Mites": "https://pnwhandbooks.org/insect/vegetable/vegetable-pests/hosts-pests/tomato-spider-mite",
"Tomato Target Spot": "https://apps.lucidcentral.org/ppp/text/web_full/entities/tomato_target_spot_163.htm",
"Tomato Mosaic Virus": "https://www.almanac.com/pest/mosaic-viruses",
"Tomato Yellow Leaf Curl Virus": "https://plantix.net/en/library/plant-diseases/200036/tomato-yellow-leaf-curl-virus",
}

class_names = ["Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy", "Blueberry Healthy", "Cherry Healthy", "Cherry Powdery Mildew",
               "Corn Cercospora", "Corn Common Rust", "Corn Healthy", "Corn Northern Leaf Blight", "Grape Black Rot", "Grape Esca", "Grape Healthy", "Grape Leaf Blight",
               "Orange Haunglongbing", "Peach Bacterial Spot", "Peach Healthy", "Pepper Bacterial Spot", "Pepper Healthy", "Potato Early Blight", "Potato Healthy", "Potato Late Blight",
               "Raspberry Healthy", "Soybean Healthy", "Squash Powdery Mildew", "Strawberry Healthy", "Strawberry Leaf Scorch", "Tomato Bacterial Spot", "Tomato Early Blight",
               "Tomato Healthy", "Tomato Late Blight", "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites", "Tomato Target Spot", "Tomato Mosaic Virus",
               "Tomato Yellow Leaf Curl Virus"]

model_path = "C:/Users/fthsl/OneDrive/Masaüstü/disease-api/model/resnet_model.h5"


resnet_model = load_model(model_path, compile=False)
resnet_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])




class Test(Resource):
    def post(self):
        try:
            # Get image data from the request
            global class_names
            img_data = request.json['data']
            data = np.array(img_data)

            prediction = resnet_model.predict(data)
            predicted_class = np.argmax(prediction)
            second_predicted_class = np.argsort(np.max(prediction, axis=0))[-2]
            third_predicted_class = np.argsort(np.max(prediction, axis=0))[-3]
            fourth_predicted_class = np.argsort(np.max(prediction, axis=0))[-4]
            fifth_predicted_class = np.argsort(np.max(prediction, axis=0))[-5]

            predicted_prob = prediction[0][predicted_class]
            second_predicted_prob = prediction[0][second_predicted_class]
            third_predicted_prob = prediction[0][third_predicted_class]
            fourth_predicted_prob = prediction[0][fourth_predicted_class]
            fifth_predicted_prob = prediction[0][fifth_predicted_class]

            predicted_class_name = class_names[predicted_class]
            second_class_name = class_names[second_predicted_class]
            third_class_name = class_names[third_predicted_class]
            fourth_class_name = class_names[fourth_predicted_class]
            fifth_class_name = class_names[fifth_predicted_class]
            response_dict = {
                "plantHealthModels": [
                    {
                        "plantName": predicted_class_name.split()[0],
                        "diseaseName": predicted_class_name,
                        "treatmentLink": treatment_links[predicted_class_name],
                        "probability": predicted_prob
                    },
                    {
                        "plantName": second_class_name.split()[0],
                        "diseaseName": second_class_name,
                        "treatmentLink": treatment_links[second_class_name],
                        "probability": second_predicted_prob
                    },
                    {
                        "plantName": third_class_name.split()[0],
                        "diseaseName": third_class_name,
                        "treatmentLink": treatment_links[third_class_name],
                        "probability": third_predicted_prob
                    },
                    {
                        "plantName": fourth_class_name.split()[0],
                        "diseaseName": fourth_class_name,
                        "treatmentLink": treatment_links[fourth_class_name],
                        "probability": fourth_predicted_prob
                    },
                    {
                        "plantName": fifth_class_name.split()[0],
                        "diseaseName": fifth_class_name,
                        "treatmentLink": treatment_links[fifth_class_name],
                        "probability": fifth_predicted_prob
                    }
                ]
            }

            if predicted_prob < 0.5:
                return json.dumps({"error":"Undefined Class"})
            elif predicted_prob < 0.7:
                first_try = requests_count["/test"]
                if requests_count["/test"] - first_try >= 3:
                    first_try = requests_count["/test"]
                    return json.dumps(response_dict, cls=NumpyEncoder)
                return json.dumps({"error":"Take a closer photograph of the leaf and try again."})
            else:
                return json.dumps(response_dict, cls=NumpyEncoder)
        except Exception as e:
            print("Error: ", str(e))
            return json.dumps({"error": str(e)})

api.add_resource(Test, '/test')



@app.route('/count')
def count():
    print(requests_count)
    return requests_count


@app.route('/tahmin')
def user_view3():
    query_string = request.query_string.decode('utf-8')
    image_url = query_string[4:]
    response = get_image_prediction(image_url, "http://127.0.0.1:5000/test")
    return response

#########
@app.route('/tahmin2')
def user_view4():
    image_path = "C:/Users/fthsl/OneDrive/Masaüstü/disease-api/test/AppleCedarRust3.JPG"
    response = get_image_prediction_path(image_path, "http://127.0.0.1:5000/test")
    return response
def load_image_from_file(file_path):
    img = Image.open(file_path)
    return img

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


def get_image_prediction_path(image_path, api_url):

    image = load_image_from_file(image_path)

    preprocessed_image = preprocess_image(image)


    data = preprocessed_image.tolist()

    response = requests.post(api_url, json={'data': data})

    print("API response:", response.text)  # Print the response text
    # Return the response JSON
    return response.json()



if __name__ == '__main__':
    app.run(debug=True)


