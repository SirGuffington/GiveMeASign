import cv2
from flask import Flask, jsonify, request, render_template, redirect
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from datetime import date
import numpy as np


# initialize our Flask application
app= Flask(__name__)
app.config['UPLOAD_PATH'] = '/Users/jamesguffey/PycharmProjects/CS478/GiveMeASign/Uploads'
allowed_extensions = ['jpg', 'png', 'mp4']

classes = [ 'Speed limit (20km/h)',
            'Speed limit (30km/h)',
            'Speed limit (50km/h)',
            'Speed limit (60km/h)',
            'Speed limit (70km/h)',
            'Speed limit (80km/h)',
            'End of speed limit (80km/h)',
            'Speed limit (100km/h)',
            'Speed limit (120km/h)',
            'No passing',
            'No passing veh over 3.5 tons',
            'Right-of-way at intersection',
            'Priority road',
            'Yield',
            'Stop',
            'No vehicles',
            'Veh > 3.5 tons prohibited',
            'No entry',
            'General caution',
            'Dangerous curve left',
            'Dangerous curve right',
            'Double curve',
            'Bumpy road',
            'Slippery road',
            'Road narrows on the right',
            'Road work',
            'Traffic signals',
            'Pedestrians',
            'Children crossing',
            'Bicycles crossing',
            'Beware of ice/snow',
            'Wild animals crossing',
            'End speed + passing limits',
            'Turn right ahead',
            'Turn left ahead',
            'Ahead only',
            'Go straight or right',
            'Go straight or left',
            'Keep right',
            'Keep left',
            'Roundabout mandatory',
            'End of no passing',
            'End no passing veh > 3.5 tons' ]

def check_file_extension(filename):
    return filename.split('.')[-1] in allowed_extensions

@app.route("/upload", methods=["GET"])
def home():
    return render_template('upload.html')

@app.route("/upload", methods=["POST"])
def classify():
    model = tf.keras.models.load_model('/Users/jamesguffey/PycharmProjects/CS478/GiveMeASign/traffic_classifier.h5')
    size = (30, 30)
    if request.method == 'POST':
        uploaded_file = request.files['file']
        #Ensure filename doesn't contain illegal characters
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            #Ensure file is appropriate type
            if check_file_extension(filename):
                uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
                #Handle video
                if filename.split('.')[-1] == 'mp4':
                    vidcap = cv2.VideoCapture(os.path.join(app.config['UPLOAD_PATH'], filename))

                    success, imag = vidcap.read()
                    count = 0

                    while success:
                        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
                        cv2.imwrite("frame%d.jpg" % count, imag)  # save frame as JPEG file
                        success, imag = vidcap.read()

                        img_PIL = Image.open(r"frame%d.jpg" % count)

                        img = img_PIL.resize((30, 30))
                        x = image.img_to_array(img)
                        x = np.expand_dims(x, axis=0)

                        images = np.vstack([x])
                        pred = np.argmax(model.predict(images), axis=-1)
                        sign = classes[pred[0]]
                        print(sign)

                        count += 1
                #Handle image
                else:
                    img = load_img(os.path.join(app.config['UPLOAD_PATH'], filename))
                    img = tf.image.resize(img, size)
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)

                    result = model.predict(x)
                    #print(result)

                    text = "The image you uploaded was: "

                    if result[0][0] == 1.0:  # if the result is Speed limit (20km/h)
                        return text + "Speed limit (20km/h)!"
                    elif result[0][1] == 1.0:  # if the result is Speed limit (30km/h)
                        return text + "Speed limit (30km/h)!"
                    elif result[0][2] == 1.0:  # if the result is Speed limit (50km/h)
                        return text + "Speed limit (50km/h)!"
                    elif result[0][3] == 1.0:  # if the result is Speed limit (60km/h)
                        return text + "Speed limit (60km/h)!"
                    elif result[0][4] == 1.0:  # if the result is Speed limit (70km/h)
                        return text + "Speed limit (70km/h)!"
                    elif result[0][5] == 1.0:  # if the result is Speed limit (80km/h)
                        return text + "Speed limit (80km/h)!"
                    elif result[0][6] == 1.0:  # if the result is End of speed limit (80km/h)
                        return text + "End of speed limit (80km/h)!"
                    elif result[0][7] == 1.0:  # if the result is Speed limit (100km/h)
                        return text + "Speed limit (100km/h)!"
                    elif result[0][8] == 1.0:  # if the result is Speed limit (120km/h)
                        return text + "Speed limit (120km/h)!"
                    elif result[0][9] == 1.0:  # if the result is No passing
                        return text + "No passing!"
                    elif result[0][10] == 1.0:  # if the result is No passing veh over 3.5 tons
                        return text + "No passing veh over 3.5 tons!"
                    elif result[0][11] == 1.0:  # if the result is Right-of-way at intersection
                        return text + "Right-of-way at intersection!"
                    elif result[0][12] == 1.0:  # if the result is Priority road
                        return text + "Priority road!"
                    elif result[0][13] == 1.0:  # if the result is Yield
                        return text + "Yield!"
                    elif result[0][14] == 1.0:  # if the result is Stop
                        return text + "Stop!"
                    elif result[0][15] == 1.0:  # if the result is No vehicles
                        return text + "No vehicles!"
                    elif result[0][16] == 1.0:  # if the result is Veh > 3.5 tons prohibited
                        return text + "Vehicle > 3.5 tons prohibited!"
                    elif result[0][17] == 1.0:  # if the result is No entry
                        return text + "No entry!"
                    elif result[0][18] == 1.0:  # if the result is General caution
                        return text + "General caution!"
                    elif result[0][19] == 1.0:  # if the result is Dangerous curve left
                        return text + "Dangerous curve left!"
                    elif result[0][20] == 1.0:  # if the result is Dangerous curve right
                        return text + "Dangerous curve right!"
                    elif result[0][21] == 1.0:  # if the result is Double curve
                        return text + "Double curve!"
                    elif result[0][22] == 1.0:  # if the result is Bumpy road
                        return text + "Bumpy road!"
                    elif result[0][23] == 1.0:  # if the result is Slippery road
                        return text + "Slippery road!"
                    elif result[0][24] == 1.0:  # if the result is Road narrows on the right
                        return text + "Road narrows on the right!"
                    elif result[0][25] == 1.0:  # if the result is Road work
                        return text + "Road work!"
                    elif result[0][26] == 1.0:  # if the result is Traffic signals
                        return text + "Traffic signals!"
                    elif result[0][27] == 1.0:  # if the result is Pedestrians
                        return text + "Pedestrians!"
                    elif result[0][28] == 1.0:  # if the result is Children crossing
                        return text + "Children crossing!"
                    elif result[0][29] == 1.0:  # if the result is Bicycles crossing
                        return text + "Bicycles crossing!"
                    elif result[0][30] == 1.0:  # if the result is Beware of ice/snow
                        return text + "Beware of ice/snow!"
                    elif result[0][31] == 1.0:  # if the result is Wild animals crossing
                        return text + "Wild animals crossing!"
                    elif result[0][32] == 1.0:  # if the result is End speed + passing limits
                        return text + "End speed + passing limits!"
                    elif result[0][33] == 1.0:  # if the result is Turn right ahead
                        return text + "Turn right ahead!"
                    elif result[0][34] == 1.0:  # if the result is Turn left ahead
                        return text + "Turn left ahead!"
                    elif result[0][35] == 1.0:  # if the result is Ahead only
                        return text + "Ahead only!"
                    elif result[0][36] == 1.0:  # if the result is Go straight or right
                        return text + "Go straight or right!"
                    elif result[0][37] == 1.0:  # if the result is Go straight or left
                        return text + "Go straight or left!"
                    elif result[0][38] == 1.0:  # if the result is Keep right
                        return text + "Keep right!"
                    elif result[0][39] == 1.0:  # if the result is Keep left
                        return text + "Keep left!"
                    elif result[0][40] == 1.0:  # if the result is Roundabout mandatory
                        return text + "Roundabout mandatory!"
                    elif result[0][41] == 1.0:  # if the result is End of no passing
                        return text + "End of no passing!"
                    elif result[0][42] == 1.0:  # if the result is End no passing veh > 3.5 tons
                        return text + "End no passing vehicle > 3.5 tons!"
                    else:
                        return "Unable to recognize image."



#  main thread of execution to start the server
if __name__=='__main__':
    app.run(debug=True)