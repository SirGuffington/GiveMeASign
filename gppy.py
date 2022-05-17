# Importing all packages
import os
import cv2
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import pandas as pd 
from keras.regularizers import l2
from keras.models import Sequential, load_model
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, classification_report

# Function to read images and their labels from a specified directory
def readImages(image_path, dir_name, num_of_classes):
	content = []
	label_list = []
	for i in range(num_of_classes):
		path = os.path.join(image_path,dir_name,str(i))
		image_dir = os.listdir(path)
		for a in image_dir:
			try:
				image = Image.open(path + '\\'+ a)
				image = image.resize((30,30))
				image = np.array(image)
				content.append(image)
				label_list.append(i)
			except:
				print("Image Loading Failed")
	content = np.array(content)
	label_list = np.array(label_list)
	return content, label_list

# Function to plot accuracy graphs  	
def plotAccuracy(history):
	plt.figure(0)
	plt.plot(history.history['accuracy'], label='Training accuracy')
	plt.plot(history.history['val_accuracy'], label='Validation accuracy')
	plt.title('Accuracy')
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.legend()
	plt.show()
	plt.figure(1)
	plt.plot(history.history['loss'], label='Training loss')
	plt.plot(history.history['val_loss'], label='Validation loss')
	plt.title('Loss')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend()
	plt.show()
	
# Function to get Test Scores
def getTestScores(test_filename):
	yTest = pd.read_csv(test_filename)
	label_list = yTest["ClassId"].values
	test_img_path = yTest["Path"].values
	content=[]
	for img in test_img_path:
		image = Image.open(img)
		image = image.resize((30,30))
		content.append(np.array(image))
	XTest=np.array(content)

	XPred = model.predict(XTest)
	XPredLabel=np.argmax(XPred,axis=1)
	return label_list, XPredLabel
	
# Function to plot Confusion Matrix	
def plot_conf_mat(Y_GT, Y_Pred):
  fig, ax = plt.subplots(figsize=(10,7))
  ax = sns.heatmap(confusion_matrix(Y_GT, Y_Pred), annot=True, cbar=False)
  plt.xlabel("Road Sign Ground Truth")
  plt.ylabel("Road Sign Prediction")
  plt.show()

# Data Members Declaration & Initialization
num_of_classes = 43
content = []
label_list = []
train_dir_name = 'train'
image_path = os.getcwd()
test_filename = 'Test.csv'
num_of_epochs = 15 
model_filename = "road_sign_model.h5" 
  
# Reading Train Images for model input
content, label_list = readImages(image_path,train_dir_name,num_of_classes)
print(content.shape, label_list.shape)

# Splitting training and testing dataset
XTrain, XTest, yTrain, yTest = train_test_split(content, label_list, test_size=0.2, random_state=42)
print(XTrain.shape, XTest.shape, yTrain.shape, yTest.shape)

# One hot encoding to convert class vector (integers) to binary class matrix
yTrain = to_categorical(yTrain, num_of_classes)
yTest = to_categorical(yTest, num_of_classes)

# CNN Model Architecture
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=XTrain.shape[1:],kernel_regularizer=l2(0.0006)))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu',kernel_regularizer=l2(0.0006)))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu',kernel_regularizer=l2(0.0006)))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',kernel_regularizer=l2(0.0006)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',kernel_regularizer=l2(0.0006)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',kernel_regularizer=l2(0.0006)))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=128, kernel_size=(1, 1), activation='relu',kernel_regularizer=l2(0.0006)))
model.add(Conv2D(filters=128, kernel_size=(1, 1), activation='relu',kernel_regularizer=l2(0.0006)))
model.add(Conv2D(filters=128, kernel_size=(1, 1), activation='relu',kernel_regularizer=l2(0.0006)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu',kernel_regularizer=l2(0.0006)))
model.add(Dropout(rate=0.25))
model.add(Dense(43, activation='softmax',kernel_regularizer=l2(0.0006)))

# Model Compilation 
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate = 0.01), metrics=['accuracy'])

# Model Fitting
history = model.fit(XTrain, yTrain, batch_size=32, epochs=num_of_epochs, validation_data=(XTest, yTest))

# Model Serialization
model.save(model_filename)

# Plotting Training and Validation Accuracy
plotAccuracy(history)

# Test Scores retreival
label_list, XPredLabel = getTestScores(test_filename)
 
# Printing Testing Acuracy 
print(accuracy_score(label_list, XPredLabel))

#Plotting Confusion Matrix
plot_conf_mat(label_list, XPredLabel)

# Printing Classification Report
print(classification_report(label_list, XPredLabel))