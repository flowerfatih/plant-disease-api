import json
import numpy as np
from flask import Flask, request
from flask_restful import Resource, Api
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from constants import requests_count, model_path, class_names, treatment_links, response_dict, schedule
from utils import NumpyEncoder, get_image_prediction

app = Flask(__name__)
api = Api(app)


@app.before_request
def count_requests():
    path = request.path
    if path not in requests_count:
        requests_count[path] = 0
    requests_count[path] += 1


plant_resnet_model = load_model(model_path, compile=False)
plant_resnet_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


class PlantHealthAnalyse(Resource):
    def post(self, response_dict=response_dict):
        try:
            # Get image data from the request
            img_data = request.json['data']
            data = np.array(img_data)

            prediction = plant_resnet_model.predict(data)
            for i in range(1, 5, 1):
                response_dict["plantHealthModels"].append({
                    "plantName": class_names[np.argsort(np.max(prediction, axis=0))[-i]].split()[0],
                    "diseaseName": class_names[np.argsort(np.max(prediction, axis=0))[-i]],
                    "treatmentLink": treatment_links[class_names[np.argsort(np.max(prediction, axis=0))[-i]]],
                    "probability": prediction[0][np.argsort(np.max(prediction, axis=0))[-i]],
                    "wateringSchedule": schedule[class_names[np.argsort(np.max(prediction, axis=0))[-i]]]["hourly"]
                })

            if prediction[0][np.argsort(np.max(prediction, axis=0))[-1]] < 0.5:
                return json.dumps({"error": "Undefined. Take a closer photograph of the leaf and try again."})
            elif prediction[0][np.argsort(np.max(prediction, axis=0))[-1]] < 0.7:
                # TODO: (Fatih) Check here
                global first_try
                first_try = requests_count["/predict"]
                if requests_count["/predict"] - first_try >= 3:
                    first_try = requests_count["/predict"]
                    return json.dumps(response_dict, cls=NumpyEncoder)
                return json.dumps({"error": "Take a closer photograph of the leaf and try again."})
            else:
                return json.dumps(response_dict, cls=NumpyEncoder)
        except Exception as e:
            print("Error: ", str(e))
            return json.dumps({"error": str(e)})


api.add_resource(PlantHealthAnalyse, '/plantHealthAnalyse')


@app.route('/count')
def count():
    print(requests_count)
    return requests_count


@app.route('/analyseHealth')
def classify():
    query_string = request.query_string.decode('utf-8')
    image_url = query_string[4:]
    response = get_image_prediction(image_url, "http://127.0.0.1:5000/plantHealthAnalyse")
    return response


# region Api For Flower Model

# flower_resnet_model = load_model(model_path, compile=False)
# flower_resnet_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
#
# class FlowerClassification(Resource):
#     def post(self, response_dict=response_dict):
#         try:
#             # Get image data from the request
#             img_data = request.json['data']
#             data = np.array(img_data)
#
#             prediction = flower_resnet_model.predict(data)
#             for i in range(1,5,1):
#                 response_dict["plantHealthModels"].append({
#                         "plantName": class_names[np.argsort(np.max(prediction, axis=0))[-i]].split()[0],
#                         "diseaseName": class_names[np.argsort(np.max(prediction, axis=0))[-i]],
#                         "treatmentLink": treatment_links[class_names[np.argsort(np.max(prediction, axis=0))[-i]]],
#                         "probability": prediction[0][np.argsort(np.max(prediction, axis=0))[-i]]
#                     })
#
#             if prediction[0][np.argsort(np.max(prediction, axis=0))[-1]] < 0.5:
#                 return json.dumps(response_dict, cls=NumpyEncoder)
#             elif prediction[0][np.argsort(np.max(prediction, axis=0))[-1]] < 0.7:
#                 global first_try
#                 first_try = requests_count["/predict"]
#                 if requests_count["/predict"] - first_try >= 3:
#
#                     first_try = requests_count["/predict"]
#                     return json.dumps(response_dict, cls=NumpyEncoder)
#                 return json.dumps({"error":"Take a closer photograph of the leaf and try again."})
#             else:
#                 return json.dumps(response_dict, cls=NumpyEncoder)
#         except Exception as e:
#             print("Error: ", str(e))
#             return json.dumps({"error": str(e)})
#
# api.add_resource(FlowerClassification, '/flowerClassification')

# @app.route('/classifyFlower')
# def classifyFlower():
#     query_string = request.query_string.decode('utf-8')
#     image_url = query_string[4:]
#     response = get_image_prediction(image_url, "http://127.0.0.1:5000/flowerClassification")
#     return response
# endregion

if __name__ == '__main__':
    app.run(debug=True)
