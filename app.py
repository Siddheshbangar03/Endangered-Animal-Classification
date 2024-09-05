import streamlit as st
import pickle
import numpy as np
import pandas as pd
import itertools
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import datetime
import time
import os
import random
import shutil
import glob
from sklearn import metrics, model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf


# Streamlit app config
st.set_page_config(
    page_title="Endangered Animal Classification",
    page_icon=":lion_face:",
    initial_sidebar_state='auto'
)

# Load pre-trained models
vgg16 = applications.VGG16(include_top=False, weights='imagenet')
model = tf.keras.models.load_model('endanger.h5')

img_width, img_height = 224, 224
train_data_dir = "Data/Train"
batch_size = 50
datagen = ImageDataGenerator(rescale=1. / 255)
generator_top = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Animal categories
animals = ['African_Elephant', 'Amur_Leopard', 'Arctic_Fox', 'Chimpanzee', 'Jaguars', 'Lion', 'Orangutan', 'Panda',
           'Panthers', 'Rhino', 'Cheetahs', 'Himalayan_Bear', 'Goat', 'Leopard', 'Changtang']

def read_image(file):
    img = load_img(file, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    return img

def test_single_image(image_file):
    images = read_image(image_file)
    bt_prediction = vgg16.predict(images)
    predict_prob = model.predict([bt_prediction])
    predict_classes = np.argmax(predict_prob, axis=1)
    
    class_predicted = model.predict([bt_prediction])
    class_predicted = np.argmax(class_predicted, axis=1)
    class_dictionary = generator_top.class_indices
    inv_map = {v: k for k, v in class_dictionary.items()}
    
    return inv_map[class_predicted[0]], round(np.max(predict_prob[0]) * 100, 2)

# Streamlit App Interface
st.title('Endangered Animal Classification')
st.header('Machine Learning Mini Project')

# Upload Image
uploaded_file = st.file_uploader("Upload an image of an animal", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Test the uploaded image
    animal_name, confidence = test_single_image(uploaded_file)
    
    st.success(f"Identified Animal: {animal_name} with {confidence}% confidence.")
    
    # Display endangered information based on the identified animal
    if animal_name == 'African_Elephant':
        st.write('This is an Endangered Animal. Left: 415,000')
    elif animal_name == 'Amur_Leopard':
        st.write('This is an Endangered Animal. Left: 320')
    elif animal_name == 'Arctic_Fox':
        st.write('This is an Endangered Animal. Left: 2,000')
    elif animal_name == 'Chimpanzee':
        st.write('This is an Endangered Animal. Left: 52,800')
    elif animal_name == 'Jaguars':
        st.write('This is an Endangered Animal. Left: 173,000')
    elif animal_name == 'Lion':
        st.write('This is an Endangered Animal. Left: 70,000')
    elif animal_name == 'Orangutan':
        st.write('This is an Endangered Animal. Left: 100,000')
    elif animal_name == 'Panda':
        st.write('This is an Endangered Animal. Left: 2,464')
    elif animal_name == 'Panthers':
        st.write('This is an Endangered Animal. Left: 2,767')
    elif animal_name == 'Rhino':
        st.write('This is an Endangered Animal. Left: 27,000')
    elif animal_name == 'Cheetahs':
        st.write('This is an Endangered Animal. Left: 6,517')
    elif animal_name == 'Himalayan_Bear':
        st.write('This is an Endangered Animal. Left: 100,000')
    elif animal_name == 'Goat':
        st.write('This is not an Endangered Animal.')
    elif animal_name == 'Leopard':
        st.write('This is not an Endangered Animal.')
    elif animal_name == 'Changtang':
        st.write('This is an Endangered Animal. Left: 15,220')
