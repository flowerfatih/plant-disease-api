import json

import numpy as np
from flask import Flask, request
from flask_restful import Resource, Api
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from constants import model_path, plant_class_names, treatment_links, schedule, flower_model_path, flower_class_names
from utils import NumpyEncoder, get_image_prediction

app = Flask(__name__)
api = Api(app)

plant_resnet_model = load_model(model_path, compile=False)
plant_resnet_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

flower_response_dict = {"FlowerClassification": []}


class PlantHealthAnalyse(Resource):
    def post(self):
        try:
            # Get image data from the request
            img_data = request.json['data']
            data = np.array(img_data)

            prediction = plant_resnet_model.predict(data)
            disease_name_1 = plant_class_names[np.argsort(np.max(prediction, axis=0))[-1]]
            plant_name_1 = disease_name_1.split()[0]
            treatment_link_1 = treatment_links[disease_name_1]
            probability_1 = prediction[0][np.argsort(np.max(prediction, axis=0))[-1]]
            watering_schedule_1 = schedule[disease_name_1]["hourly"]

            disease_name_2 = plant_class_names[np.argsort(np.max(prediction, axis=0))[-2]]
            plant_name_2 = disease_name_2.split()[0]
            treatment_link_2 = treatment_links[disease_name_2]
            probability_2 = prediction[0][np.argsort(np.max(prediction, axis=0))[-2]]
            watering_schedule_2 = schedule[disease_name_2]["hourly"]

            disease_name_3 = plant_class_names[np.argsort(np.max(prediction, axis=0))[-3]]
            plant_name_3 = disease_name_3.split()[0]
            treatment_link_3 = treatment_links[disease_name_3]
            probability_3 = prediction[0][np.argsort(np.max(prediction, axis=0))[-3]]
            watering_schedule_3 = schedule[disease_name_3]["hourly"]

            disease_name_4 = plant_class_names[np.argsort(np.max(prediction, axis=0))[-4]]
            plant_name_4 = disease_name_4.split()[0]
            treatment_link_4 = treatment_links[disease_name_4]
            probability_4 = prediction[0][np.argsort(np.max(prediction, axis=0))[-4]]
            watering_schedule_4 = schedule[disease_name_4]["hourly"]

            plant_response_dict = {"plantHealthModels": [
                {
                    "plantName": plant_name_1,
                    "diseaseName": disease_name_1,
                    "treatmentLink": treatment_link_1,
                    "probability": probability_1,
                    "wateringSchedule": watering_schedule_1
                },
                {
                    "plantName": plant_name_2,
                    "diseaseName": disease_name_2,
                    "treatmentLink": treatment_link_2,
                    "probability": probability_2,
                    "wateringSchedule": watering_schedule_2
                },
                {
                    "plantName": plant_name_3,
                    "diseaseName": disease_name_3,
                    "treatmentLink": treatment_link_3,
                    "probability": probability_3,
                    "wateringSchedule": watering_schedule_3
                },
                {
                    "plantName": plant_name_4,
                    "diseaseName": disease_name_4,
                    "treatmentLink": treatment_link_4,
                    "probability": probability_4,
                    "wateringSchedule": watering_schedule_4
                }
            ]}

            if probability_1 < 0.5:
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
    def post(self):
        try:
            # Get image data from the request
            img_data = request.json['data']
            data = np.array(img_data)

            prediction = flower_resnet_model.predict(data)

            flower_name_1 = flower_class_names[np.argsort(np.max(prediction, axis=0))[-1]].split()[0]
            treatment_link_1 = treatment_links[flower_name_1]
            probability_1 = prediction[0][np.argsort(np.max(prediction, axis=0))[-1]]
            watering_schedule_1 = schedule[flower_name_1]["hourly"]

            flower_name_2 = flower_class_names[np.argsort(np.max(prediction, axis=0))[-2]].split()[0]
            treatment_link_2 = treatment_links[flower_name_2]
            probability_2 = prediction[0][np.argsort(np.max(prediction, axis=0))[-2]]
            watering_schedule_2 = schedule[flower_name_2]["hourly"]

            flower_name_3 = flower_class_names[np.argsort(np.max(prediction, axis=0))[-3]].split()[0]
            treatment_link_3 = treatment_links[flower_name_3]
            probability_3 = prediction[0][np.argsort(np.max(prediction, axis=0))[-3]]
            watering_schedule_3 = schedule[flower_name_3]["hourly"]

            flower_name_4 = flower_class_names[np.argsort(np.max(prediction, axis=0))[-4]].split()[0]
            treatment_link_4 = treatment_links[flower_name_4]
            probability_4 = prediction[0][np.argsort(np.max(prediction, axis=0))[-4]]
            watering_schedule_4 = schedule[flower_name_4]["hourly"]

            flower_response_dict = {"plantHealthModels": [
                {
                    "plantName": flower_name_1,
                    "growingLink": treatment_link_1,
                    "probability": probability_1,
                    "wateringSchedule": watering_schedule_1
                },
                {
                    "plantName": flower_name_2,
                    "treatmentLink": treatment_link_2,
                    "probability": probability_2,
                    "wateringSchedule": watering_schedule_2
                },
                {
                    "plantName": flower_name_3,
                    "treatmentLink": treatment_link_3,
                    "probability": probability_3,
                    "wateringSchedule": watering_schedule_3
                },
                {
                    "plantName": flower_name_4,
                    "treatmentLink": treatment_link_4,
                    "probability": probability_4,
                    "wateringSchedule": watering_schedule_4
                }
            ]}

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
