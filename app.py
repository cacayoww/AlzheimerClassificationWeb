from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)
print('halo')
model = load_model('model_simplerelu_2.h5')
print('hai')
IMG_SIZE = 224

CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']


def predict_image(img_bytes):
    img = Image.open(BytesIO(img_bytes)).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return predictions

@app.route("/", methods=['GET', 'POST'])
def main():
    print('haii')
    return render_template('index.html')


@app.route("/deteksi", methods=['GET', 'POST'])
def deteksi():
    print('haii')
    return render_template('deteksi.html', predicted_class=None, img_data=None)

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    print('halo')
    if request.method == 'POST':
        file = request.files['brain']
        img_bytes = file.read()
        predictions = predict_image(img_bytes)
        predicted_index = np.argmax(predictions)
        predicted_label = CLASS_NAMES[predicted_index]
        
        
        
        return render_template('deteksi.html', predicted_class=predicted_label, img_data=file)
    return render_template('deteksi.html', predicted_class=None, img_data=None)

if __name__ == '__main__':
    print('mana nih')
    app.run(debug=False)
