import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
import tensorflow as tf
from keras.models import Sequential,load_model,model_from_json
from keras.layers import Dense, Dropout,Activation,MaxPooling2D,Conv2D,Flatten
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.utils import load_img
from keras.preprocessing import image
import numpy as np
import h5py
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import codecs
import os
import csv


app = Flask(__name__) #Initialize the flask App

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#model = load_model('model.pk')
sc = pickle.load(open('model.pk', 'rb'))
@app.route("/")
@app.route("/index")
def index():
	return render_template('index.html')

@app.route('/chart')
def chart():
	return render_template('chart.html')


@app.route('/login')
def login():
	return render_template('login.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        #df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)	



@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    return render_template('prediction.html')
	


@app.route('/predict',methods=['POST'])
def predict():
    dataset2 = request.files['datasetfile2']
    df2 = pd.read_csv(dataset2,encoding = 'unicode_escape')
    
    result = sc.predict(df2)
    prediction = result[0]
    return render_template('prediction.html', prediction_text= prediction)

@app.route('/performance')
def performance():
	return render_template('performance.html')
    
if __name__ == "__main__":
    app.run(debug=True)
