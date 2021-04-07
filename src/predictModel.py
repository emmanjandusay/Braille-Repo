import numpy as np
import cv2 as cv
import os.path

from preprocess import process_image
from generateModel import generate, getPickle, updateModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, recall_score, precision_score, f1_score

categories = ['binata', 'buhay', 'dalaga', 'eksamen', 'ewan', 'gunita', 'halaman', 'hapon', 'isip', 'kailangan',
             'karaniwan', 'kislap', 'larawan', 'mabuti', 'noon', 'opo', 'papaano', 'patuloy', 'roon', 'subalit',
              'talaga', 'ugali', 'wasto']

test_image_path = "..\\test-images\\test1.png"

image = cv.imread(test_image_path,1)
image = process_image(image)
image = np.expand_dims(image, 0)

if os.path.isfile('braille-model.pickle'):
    print('Model found...')
    pickle = getPickle()
    model, xtrain, xtest, ytrain, ytest = updateModel(pickle)

    print('Predicting xtest...')

    prediction = model.predict(image)
    accuracy = model.score(xtest, ytest)

    print('Prediction Integer is :', prediction[0])
    print('Prediction is :', categories[prediction[0]])
    print('Recall xtest: ', recall_score(ytest, prediction, pos_label='positive', average='macro'))
    print('Precision xtest: ', precision_score(ytest, prediction, pos_label='positive', average='macro'))

else:
    print('Model not found...')
    generate()
    pickle = getPickle()
    model, xtrain, xtest, ytrain, ytest = updateModel(pickle)

    print('Predicting xtest...')

    prediction = model.predict(image)
    accuracy = model.score(xtest, ytest)

    print('Prediction Integer is :', prediction[0])
    print('Prediction is :', categories[prediction[0]])
    print('Recall xtest: ', recall_score(ytest, prediction, pos_label='positive', average='macro'))
    print('Precision xtest: ', precision_score(ytest, prediction, pos_label='positive', average='macro'))