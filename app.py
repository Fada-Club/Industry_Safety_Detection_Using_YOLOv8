import sys
import os
from isd.pipeline.training_pipeline import TrainPipeline
from isd.exception import isdException
from isd.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from ultralytics import YOLO  # Import YOLO from the Ultralytics library
from PIL import Image

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.model = self.load_model()

    def load_model(self):
        model_path = "model/best.pt"  # Path to your saved YOLOv8 model
        model = YOLO(model_path)  # Load the model using Ultralytics
        return model

    def predict(self, image_path):
        results = self.model(image_path)  # Make predictions on the image

        # If results are a list, save each one; usually there's only one
        for result in results:
            result.save()  # This saves the results to the default directory
        return results


@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return Response("Training pipeline executed successfully.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)  # Decode and save the uploaded image
        
        # Update this path to point to the saved image in the data directory
        clApp.predict(os.path.join('./data', clApp.filename))  # Perform prediction
        opencodedbase64 = encodeImageIntoBase64("runs/detect/exp/inputImage.jpg")  # Change path if necessary
        result = {"image": opencodedbase64.decode('utf-8')}

    except ValueError as val:
        print(val)
        return Response("Value not found inside JSON data")
    except KeyError:
        return Response("Key value error: incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080)
