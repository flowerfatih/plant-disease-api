import json
import numpy as np
from flask import Flask, request
from flask_restful import Resource, Api
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from constants import model_path, plant_class_names, treatment_links, plant_response_dict, schedule, flower_model_path, \
    flower_response_dict, flower_class_names
from utils import NumpyEncoder, get_image_prediction

app = Flask(__name__)
api = Api(app)

plant_resnet_model = load_model(model_path, compile=False)
plant_resnet_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


class PlantHealthAnalyse(Resource):
    def post(self, response_dict=plant_response_dict):
        try:
            # Get image data from the request
            img_data = request.json['data']
            data = np.array(img_data)

            prediction = plant_resnet_model.predict(data)
            for i in range(1, 5, 1):
                plant_response_dict["plantHealthModels"].append({
                    "plantName": plant_class_names[np.argsort(np.max(prediction, axis=0))[-i]].split()[0],
                    "diseaseName": plant_class_names[np.argsort(np.max(prediction, axis=0))[-i]],
                    "treatmentLink": treatment_links[plant_class_names[np.argsort(np.max(prediction, axis=0))[-i]]],
                    "probability": prediction[0][np.argsort(np.max(prediction, axis=0))[-i]],
                    "wateringSchedule": schedule[plant_class_names[np.argsort(np.max(prediction, axis=0))[-i]]]["hourly"]
                })

            probability = prediction[0][np.argsort(np.max(prediction, axis=0))[-1]]
            if probability < 0.5:
                return json.dumps({"error": "Undefined. Take another leaf photograph and try again."})
            else:
                return json.dumps(plant_response_dict, cls=NumpyEncoder)
        except Exception as e:
            print("Error: ", str(e))
            return json.dumps({"error": str(e)})


api.add_resource(PlantHealthAnalyse, '/plantHealthAnalyse')


@app.route('/analyseHealth')
def classify():
    query_string = request.query_string.decode('utf-8')
    image_url = query_string[4:]
    response = get_image_prediction(image_url, "http://127.0.0.1:5000/plantHealthAnalyse")
    return response


# region Api For Flower Model

flower_resnet_model = load_model(flower_model_path, compile=False)
flower_resnet_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

class FlowerClassification(Resource):
    def post(self, response_dict=flower_response_dict):
        try:
            # Get image data from the request
            img_data = request.json['data']
            data = np.array(img_data)

            prediction = flower_resnet_model.predict(data)
            for i in range(1,5,1):
                flower_response_dict["FlowerClassification"].append({
                        "flowerName": flower_class_names[np.argsort(np.max(prediction, axis=0))[-i]].split()[0],
                        "wateringSchedule": schedule[flower_class_names[np.argsort(np.max(prediction, axis=0))[-i]]],
                        "growingLink": treatment_links[flower_class_names[np.argsort(np.max(prediction, axis=0))[-i]]],
                        "probability": prediction[0][np.argsort(np.max(prediction, axis=0))[-i]]
                    })

            if prediction[0][np.argsort(np.max(prediction, axis=0))[-1]] < 0.5:
                return json.dumps(flower_response_dict, cls=NumpyEncoder)
            else:
                return json.dumps(flower_response_dict, cls=NumpyEncoder)
        except Exception as e:
            print("Error: ", str(e))
            return json.dumps({"error": str(e)})

api.add_resource(FlowerClassification, '/flowerClassification')

@app.route('/classifyFlower')
def classifyFlower():
    query_string = request.query_string.decode('utf-8')
    image_url = query_string[4:]
    response = get_image_prediction(image_url, "http://127.0.0.1:5000/flowerClassification")
    return response
# endregion

if __name__ == '__main__':
    app.run(debug=True)
