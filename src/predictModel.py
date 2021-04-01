import numpy as np
import cv2 as cv

from preprocess import process_image
from generateModel import generate, getPickle, updateModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, recall_score, precision_score, f1_score

categories = ['binata', 'buhay', 'dalaga', 'eksamen', 'ewan', 'gunita', 'halaman', 'hapon', 'isip', 'kailangan',
             'karaniwan', 'kislap', 'larawan', 'mabuti', 'noon', 'opo', 'papaano', 'patuloy', 'roon', 'subalit',
              'talaga', 'ugali', 'wasto']

test_image_path = "Braille2C_Datasets\\wasto\\wasto_1.jpg"

image = cv.imread(test_image_path,1)
image = process_image(image)
image = np.expand_dims(image, 0)

generate()
pickle = getPickle()
model, xtrain, xtest, ytrain, ytest = updateModel(pickle)

print('Predicting xtest...')

prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)

print('Prediction Integer is :', prediction[0])
print('Prediction is :', categories[prediction[0]])
print('Recall xtest: ', recall_score(ytest, prediction, pos_label='positive', average='macro'))
print('Precision xtest: ', precision_score(ytest, prediction, pos_label='positive', average='macro'))