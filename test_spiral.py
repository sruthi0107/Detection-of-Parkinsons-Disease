from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib 
from xgboost import XGBClassifier
from skimage import feature
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import cv2
from cv2 import resize
import os
from PIL import Image

def quantify_image(image):
    # compute the histogram of oriented gradients feature vector for
    # the input image
    features = feature.hog(image, orientations=9,
        pixels_per_cell=(10, 10), cells_per_block=(2, 2),
        transform_sqrt=True, block_norm="L1")

    # return the feature vector
    return features

def specify_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))

    # threshold the image such that the drawing appears as white
    # on a black background
    image = cv2.threshold(image, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # quantify the image
    features = quantify_image(image)

    # update the data and labels lists, respectively
    return features

def s_model(image):
    #im1 = cv2.imread(image)
    #testX=specify_image(im1)
    image = np.array(image)
    testX=specify_image(image)
    model=joblib.load('spiral_model.pkl')
    predictions = model.predict([testX])
    #predictions = model.predict([image])
    return predictions
'''
im1 = cv2.imread('D:\Prasad\detection of parkinsons\detect-parkinsons\V03PE01.png')
testX=specify_image(im1)
#testX=specify_image(image)
model=joblib.load('D:\Prasad\detection of parkinsons\detect-parkinsons\spiral_model.pkl')
predictions = model.predict([testX])
#predictions = model.predict([image])
print(predictions)
'''