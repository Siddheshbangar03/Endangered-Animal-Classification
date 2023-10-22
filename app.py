import streamlit as st
import pickle
import numpy as np
import pandas as pd
import itertools
import tensorflow as tf
import tf.keras
from sklearn import metrics, model_selection
from sklearn.metrics import confusion_matrix
from tf.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tf.keras.models import Sequential
from tf.keras import optimizers
from tf.keras.preprocessing import image
from tf.keras.layers import Dropout, Flatten, Dense
from tf.keras import applications
from tf.keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import datetime
import time
import os
import random
import shutil
import glob
from sklearn.preprocessing import LabelEncoder
from tf.keras.optimizers import Adam
from tf.keras.applications.resnet50 import preprocess_input
import streamlit as st

st.set_page_config(
    page_title="Endangered Animal Classification",
    page_icon = ":lion_face:",
    initial_sidebar_state = 'auto'
)

vgg16 = applications.VGG16(include_top=False, weights='imagenet')
# Load your pre-trained CNN model
model = tf.keras.models.load_model('endanger.h5')
img_width, img_height = 224, 224
train_data_dir = "Data\Train"
batch_size = 50
datagen = ImageDataGenerator(rescale=1. / 255)
generator_top = datagen.flow_from_directory(
         train_data_dir,
         target_size=(img_width, img_height),
         batch_size=batch_size,
         class_mode='categorical',
         shuffle=False)

def test_single_image(path):
    animals = ['African_Elephant','Amur_Leopard','Arctic_Fox','Chimpanzee','Jaguars','Lion','Orangutan','Panda','Panthers','Rhino','cheetahs','Himalyan_Bear','Goat','Leopard','Changtang']
    images = read_image(path)
    time.sleep(.5)
    bt_prediction = vgg16.predict(images)
    
    predict_prob=model.predict([bt_prediction])
    predict_classes=np.argmax(predict_prob,axis=1)
    for idx, animals, x in zip(range(0,15), animals , predict_prob[0]):
        print("ID: {}, Label: {} {}%".format(idx, animals, round(x*100,2) ))
    print('Final Decision:')
    time.sleep(.5)
    for x in range(3):
        print('.'*(x+1))
        time.sleep(.2)

    class_predicted=model.predict([bt_prediction])
    class_predicted=np.argmax(class_predicted,axis=1)
    class_dictionary = generator_top.class_indices
    inv_map = {v: k for k, v in class_dictionary.items()}
    print("ID: {}, Label: {}".format(class_predicted[0], inv_map[class_predicted[0]]))
    endanger = st.write("Label: {}".format(inv_map[class_predicted[0]]))

def read_image(file_path):
    print("[INFO] loading and preprocessing image...")
    image = load_img(file_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.
    return image

#path = "Data/Test/Arctic_Fox/367.jpg"
st.title(':red[Endangered] *_Animal Classification_*')
st.header('Machine Learning Mini Project', divider='green')

with st.form('form', clear_on_submit = True):
    path = st.text_input("Please Select the Image path for Identyfying Animal")
    st.form_submit_button(label='Identify the Animal')
st.image(path, caption="Uploaded Image of Animal", use_column_width=True)
test_single_image(path)

st.info('Please click the below button to know more about the Animal.', icon="ℹ️")
col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
with col1:
    if st.button('African Elephant'):
        st.write('This is Endangered Animal. Left 415000')
with col2:
    if st.button('Amur Leopard'):
        st.write('This is Endangered Animal. Left - 320')
with col3:
    if st.button('Arctic Fox'):
        st.write('This is Endangered Animal. Left - 2000')
with col4:
    if st.button('Chimpanzee'):
        st.write('This is Endangered Animal. Left - 52800')
with col5:
    if st.button('Jaguars'):
        st.write('This is Endangered Animal. Left - 173000')
with col1:
    if st.button('Lion'):
        st.write('This is Endangered Animal. Left - 70000')
with col2:
    if st.button('Orangutan'):
        st.write('This is Endangered Animal. Left - 100000')
with col3:
    if st.button('Panda'):
        st.write('This is Endangered Animal. Left - 2464')
with col4:
    if st.button('Panthers'):
        st.write('This is Endangered Animal. Left - 2767')
with col5:
    if st.button('Rhino'):
        st.write('This is Endangered Animal. Left - 27000')
with col1:
    if st.button('Cheetahs'):
        st.write('This is Endangered Animal. Left - 6517')
with col2:
    if st.button('Himalyan Bear'):
        st.write('This is Endangered Animal. Left - 100000')
with col3:
    if st.button('Goat'):
        st.write('This is Not Endangered Animal.')
with col4:
    if st.button('Leopard'):
        st.write('This is Not Endangered Animal.')
with col5:
    if st.button('Changtang'):
        st.write('This is Endangered Animal. Left - 15220')

st.text("")
st.text("")
st.text("")
st.text("")
st.text("")

st.subheader('Made by')
st.subheader(':green[Aditya Satheesan]')
st.subheader(':green[Siddhesh Bangar]')
st.subheader(':green[Zaid Khalfe]')
st.subheader(':green[Samrudhi Jagadale]')

