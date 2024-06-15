from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from inference_sdk import InferenceHTTPClient
import numpy as np
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
model = load_model('model_simplerelu_2.h5')
IMG_SIZE = 224

CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="rRZj7C7r9U5L7NKXTmwg"
)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)/255.0
    img = img.reshape(1, 224,224,3)
    # predictions = model.predict_classes(i)
    predictions = model.predict(img)
    
    return predictions

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template('index.html')


@app.route("/deteksi", methods=['GET', 'POST'])
def deteksi():
    return render_template('deteksi.html', predicted_class=None, img_data=None)

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        
        file = request.files['brain']
        img_path = "static/" + file.filename
        file.save(img_path)
        predictions = predict_image(img_path)
        print(predictions)
        predicted_index = np.argmax(predictions)
        predicted_label = CLASS_NAMES[predicted_index]
        
        
        result = CLIENT.infer(img_path, model_id="alzheimer-5oeur/1")['predictions'][0]['class']
        print(result)
        return render_template('deteksi.html', predicted_class=predicted_label, predicted_class_2=result, img_data=img_path)
    return render_template('deteksi.html', predicted_class=None,predicted_class_2=None,img_data=None)

if __name__ == '__main__':
    app.run(debug=False)
