from time import sleep
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
from flask_wtf import FlaskForm
from wtforms import FileField
from flask_uploads import configure_uploads, IMAGES, UploadSet
import pickle
import threading
import base64
from VF import predict_vf
from werkzeug.utils import secure_filename
from q10 import predict_q10

init_Base64 = 21


app = Flask(__name__)

app.config['SECRET_KEY'] = 'thisisasecret'
app.config['UPLOADED_IMAGES_DEST'] = 'uploads/images'

images = UploadSet('images', IMAGES)
configure_uploads(app, images)

class MyForm(FlaskForm):
    image = FileField('image')

@app.route('/')
def home():

    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    print(request.form.values())
    all_features = [x for x in request.form.values()]

    int_features = [int(x) for x in all_features[2:12]] # A1-A10
    int_features.append(int(all_features[1])) # age in months
    qScore = int_features[2:12].count(1)
    int_features.append(qScore) # q score
    int_features.append(int(all_features[0])) # sex
    int_features.append(int(all_features[12])) # jaundice
    int_features.append(int(all_features[13])) # ASD family history

    img = request.files['filename']

    basepath = os.path.dirname(__file__)

    filepath = os.path.join(basepath, secure_filename(img.filename))

    # Temporarily save image to filepath
    img.save(filepath)

    # Get probability using vggFace
    vf = predict_vf(filepath)[0]

    # Get probability using q10
    qchat10 = predict_q10([int_features])[0][0]

    sleep(2)

    # Delete image from database(filepath)
    os.remove(filepath)

    return render_template('results.html', Pred_vgg=vf, Pred_q10=qchat10)

@app.route('/results',methods=['POST'])
def results():
    data = request.get_json(force=True)
    return jsonify(data)

if __name__ == "__main__":
    app.run()
