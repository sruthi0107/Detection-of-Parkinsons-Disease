# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:25:43 2020

@author: user
"""
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

from flask import Flask,render_template,request
import test_spiral
import test_wave
from test_spiral import quantify_image,specify_image,s_model
from test_wave import quantify_image,specify_image,w_model
app=Flask(__name__)
@app.route("/")
@app.route("/home")
def main_p():
    return render_template('Main_page.html')

@app.route("/Spiral_test")
def spiral():
    return render_template('Spiral_test.html')
@app.route("/Wave_test")
def wave():
    return render_template('Wave_test.html')

@app.route("/image_data/<image_type>",methods=['GET','POST'])
def image_data(image_type):
    image=""
    if(request.method=='POST'):
        image=request.files['image']
    img=Image.open(image)
    if(image_type=='spiral'):
        pred=s_model(img)
    else:
        pred=w_model(img)
    
    if(pred==1):
        return render_template('parkinsons.html')
    else:
        return render_template('healthy.html')


if __name__=='__main__':
    app.run(debug=True)