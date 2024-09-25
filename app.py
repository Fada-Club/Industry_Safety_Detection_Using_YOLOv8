import sys
import os
from isd.pipeline.training_pipeline import TrainPipeline
from isd.exception import isdException
from isd.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from ultralytics import YOLO  # Import YOLO from the Ultralytics library
from PIL import Image
import shutil

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "data/inputImage.jpg"  # Correct file path without duplicating "data"
        self.model_path = "model/best.pt"      # Path to the best trained model
        self.model = YOLO(self.model_path)     # Load YOLOv8 model

clApp = ClientApp()  # Initialize the ClientApp instance


@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training completed successfully!"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        # Check if the image is in the request
        if 'image' not in request.json:
            return Response("No image file found in the request.", status=400)

        # Ensure the "data" directory exists
        if not os.path.exists('data'):
            os.makedirs('data')

        # Decode and save the image to the expected location
        image = request.json['image']
        decodeImage(image, clApp.filename)  # Save to "data/inputImage.jpg"
        
        if not os.path.exists(clApp.filename):
            return Response("Failed to save the image", status=400)

        # Predict using YOLOv8 model and save the result
        results = clApp.model.predict(source=clApp.filename, save=True)  # Save the results automatically
        
        # Assuming the output is saved in the "runs/detect" folder, get the last prediction
        output_dir = results[0].save_dir
        output_image_path = os.path.join(output_dir, os.path.basename(clApp.filename))

        # Encode the image in Base64 and return the result
        opencodedbase64 = encodeImageIntoBase64(output_image_path)
        result = {"image": opencodedbase64.decode('utf-8')}

        # Clean up: Remove the 'runs' folder if no further use
        runs_folder = os.path.dirname(output_dir)
        if os.path.exists(runs_folder):
            shutil.rmtree(runs_folder)  # Delete the entire 'runs' folder

    except ValueError as val:
        print(val)
        return Response("Value not found inside JSON data", status=400)
    except KeyError:
        return Response("Key value error: incorrect key passed", status=400)
    except Exception as e:
        print(e)
        return Response(f"An error occurred during prediction: {str(e)}", status=500)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)