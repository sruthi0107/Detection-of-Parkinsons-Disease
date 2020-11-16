# USASGE
# python park.py --dataset dataset/spiral
# python park.py --dataset dataset/wave

# import the necessary packages
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

def load_split(path):
    # grab the list of images in the input directory, then initialize
    # the list of data (i.e., images) and class labels
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        # load the input image, convert it to grayscale, and resize
        # it to 200x200 pixels, ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))

        # threshold the image such that the drawing appears as white
        # on a black background
        image = cv2.threshold(image, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # quantify the image
        features = quantify_image(image)

        # update the data and labels lists, respectively
        data.append(features)
        labels.append(label)

    # return the data and labels
    return (np.array(data), np.array(labels))

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
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-t", "--trials", type=int, default=5,
    help="# of trials to run")
args = vars(ap.parse_args())

# define the path to the training and testing directories
trainingPath = os.path.sep.join([args["dataset"], "training"])
#testingPath = os.path.sep.join([args["dataset"], "testing"])

# loading the training and testing data
print("[INFO] loading data...")
(trainX, trainY) = load_split(trainingPath)
im1 = cv2.imread("V09PO01.png")
testX=specify_image(im1)
testX.reshape(-1,1)
#testX=specify_image("V01HE01.png")
#(testX, testY) = load_split(testingPath)

# encode the labels as integers
le = LabelEncoder()
trainY = le.fit_transform(trainY)
#testY = le.transform(testY)

# initialize our trials dictionary
trials = {}

# loop over the number of trials to run
for i in range(0, args["trials"]):
    # train the model
    print("[INFO] training model {} of {}...".format(i + 1,
        args["trials"]))
    model = RandomForestClassifier(n_estimators=100)
    #model = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    #model = GaussianNB()
    #model = XGBClassifier()
    model.fit(trainX, trainY)
    joblib.dump(model, 'wave_model.pkl') 

    # make predictions on the testing data and initialize a dictionary
    # to store our computed metrics

    predictions = model.predict([testX])
    print(le.inverse_transform(predictions))

