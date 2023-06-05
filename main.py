import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split
from cvzone.FaceDetectionModule import FaceDetector


path = "Face-image-dataset"
images = []
age = []
gender = []

detector = FaceDetector()

for img in os.listdir(path):
  print(img)
  if img!='.git':
    ages = img.split("_")[0]
    genders = img.split("_")[1]
    img = cv2.imread(str(path)+"/"+str(img))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
     
    # Crop the face out of an image and size the image at 200,200
    # img, bbox = detector.findFaces(img, draw=False) 
    # X = bbox[0]['bbox'][0]
    # Y = bbox[0]['bbox'][1]
    # W = bbox[0]['bbox'][2]
    # H = bbox[0]['bbox'][3]

    # croppedImg = img[Y:Y+H, X:X+W]
    # resizedImg = [cv2.resize(croppedImg, (200, 200))]
            
    # images.append(resizedImg)
    
    images.append(img)
    age.append(ages)


age = np.array(age,dtype=np.int64)
images = np.array(images)

# Third class
x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, age, random_state=42)



age_model = Sequential()

age_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200,200,3)))
#age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
#age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))
              
age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
#age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
#age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Flatten())
age_model.add(Dropout(0.2))
age_model.add(Dense(512, activation='relu'))

age_model.add(Dense(1, activation='linear', name='age'))
              
age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(age_model.summary()) 

history_age = age_model.fit(x_train_age, y_train_age,
                        validation_data=(x_test_age, y_test_age), epochs=10)

age_model.save('age_model_50epochs.h5')


history = history_age

# Plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()