from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

images1 = []
ages1 = []
genders1 = []
race1 = []

images_data1 = os.listdir(r'C:\Users\Dell\Desktop\age\images')
for imag in images_data1:
    split = imag.split('_')
    ages1.append(int(split[0]))
    genders1.append(int(split[1]))
    race1.append(int(split[2]))
    imag = cv2.imread(r'C:\\Users\\Dell\\Desktop\\age\\images\\' + imag)
    imag = cv2.resize(imag,(200,200))
    images1.append(imag)
images1 = np.array(images1)
images1.shape

#preprocessing
images1 = (images1) / 255.0
ages1 = np.array(ages1)
genders1 = np.array(genders1)
race1 = np.array(race1)

#train test split
x_train_age1, x_test_age1, y_train_age1, y_test_age1 = train_test_split(images1, ages1, test_size=0.2)
x_train_genders1, x_test_genders1, y_train_genders1, y_test_genders1 = train_test_split(images1, genders1, test_size=0.2)
x_train_race1, x_test_race1, y_train_race1, y_test_race1 = train_test_split(images1, race1, test_size=0.2)

#building model

agemodel = Sequential()
agemodel.add(Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)))
agemodel.add(MaxPooling2D((2,2)))
agemodel.add(Conv2D(64, (3,3), activation='relu'))
agemodel.add(MaxPooling2D((2,2)))
agemodel.add(Conv2D(128, (3,3), activation='relu'))
agemodel.add(MaxPooling2D((2,2)))
agemodel.add(Flatten())
agemodel.add(Dense(128, activation='relu'))
agemodel.add(Dense(32, activation='softmax'))

agemodel.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

genmodel = Sequential()
genmodel.add(Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)))
genmodel.add(MaxPooling2D((2,2)))
genmodel.add(Conv2D(64, (3,3), activation='relu'))
genmodel.add(MaxPooling2D((2,2)))
genmodel.add(Conv2D(128, (3,3), activation='relu'))
genmodel.add(MaxPooling2D((2,2)))
genmodel.add(Flatten())
genmodel.add(Dense(64, activation='relu'))
genmodel.add(Dropout(0.5))
genmodel.add(Dense(1, activation='sigmoid'))

genmodel.compile(loss='binary_crossentropy',
             optimizer=optimizers.Adam(lr=0.0001),
             metrics=['accuracy'])

racemodel = Sequential()
racemodel.add(Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)))
racemodel.add(MaxPooling2D((2,2)))
racemodel.add(Conv2D(64, (3,3), activation='relu'))
racemodel.add(MaxPooling2D((2,2)))
racemodel.add(Conv2D(128, (3,3), activation='relu'))
racemodel.add(MaxPooling2D((2,2)))
racemodel.add(Flatten())
racemodel.add(Dense(64, activation='relu'))
racemodel.add(Dense(10, activation='softmax'))

racemodel.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history1 = agemodel.fit(x_train_age1, y_train_age1, epochs=10, shuffle=True)

agemodel.evaluate(x_test_age1,y_test_age1)
agemodel.save('agemodel.h5')

history2 = genmodel.fit(x_train_genders1, y_train_genders1, epochs=10, shuffle=True)
genmodel.evaluate(x_test_genders1,y_test_genders1)
genmodel.save('gender.h5')

history3 = racemodel.fit(x_train_race1, y_train_race1, epochs=10, shuffle=True)
racemodel.evaluate(x_test_race1,y_test_race1)
racemodel.save('race.h5')

# testing or evaluating all models together
img = cv2.imread(r"C:\Users\Dell\Downloads\Kiara-Advani-Photo.jpg")
img_new = cv2.resize(img, (200, 200))
plt.imshow(img_new)
img_new = img_new / 255.0
img_new = np.expand_dims(img_new, axis = 0)
gen_pred1 = genmodel.predict(img_new)
age_pred1 = agemodel.predict(img_new)
age_pred1 = np.argmax(age_pred1)
race_pred1 = racemodel.predict(img_new)
race_pred1 = np.argmax(race_pred1)
print("age is: ",age_pred1)
race_classes = ['white','black','asian','indian','others']
print("race is:", race_classes[race_pred1])

if gen_pred1 >= 0.5:
    gen_pred1 = 'female'
else:
    gen_pred1 = 'male'
print("gender is :",gen_pred1)
