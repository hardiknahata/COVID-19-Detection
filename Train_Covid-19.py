import os
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2

# Creating data and labels
data = []
labels = []

# Importing Covid Samples
path1 = "dataset/covid/"

for i in os.listdir(path1):
    labels.append(i[:5])
    str = path1+i
    img = cv2.imread(str)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    data.append(img)

# Importing Normal Samples
path2 = "dataset/normal/"

for i in os.listdir(path2):
    labels.append(i[:6])
    str = path2+i
    img = cv2.imread(str)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    data.append(img)

# Converting the data and labels to NumPy arrays
# Normalizing Image Intensities to the range [0, 255]
data = np.array(data) / 255.0
labels = np.array(labels)

# One-Hot Encoding the labels
lb = LabelBinarizer()
labels_num = lb.fit_transform(labels) # Converts to numerical
labels = to_categorical(labels_num)   # Converts Numerical to One-Hot Encoding

# Splitting the Train and Test sets
(xtrain, xtest, ytrain, ytest) = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=42)

# Defining the Augmented Images Generator
aug = ImageDataGenerator(rotation_range=10,
#                          zoom_range = 0.1,
#                          width_shift_range=0.2,
#                          height_shift_range=0.2,
                         fill_mode="nearest")

# Importing the Base Model
baseModel = VGG16( weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3) ))

# Freeze all the layers in the base model so that they aren't updated during training
for layer in baseModel.layers:
    layer.trainable = False

# Building the Model Architecture
model = Sequential()
model.add(baseModel)
model.add( MaxPooling2D(pool_size = (4, 4) ))
model.add( Flatten() )
model.add( Dense(64, activation='relu') )
model.add( Dropout(0.2))
model.add( Dense( 2, activation='softmax') )
# model.summary()

# Setting Model Parameters and Optimizer
epochs = 25
lr = 1e-3   # Learning Rate
BS = 8     # Batch Size
adam = Adam (lr = lr, decay = lr/epochs)
model.compile( loss = "binary_crossentropy", optimizer = adam, metrics = ["accuracy"] )

# Fitting the Model
history = model.fit_generator(
    aug.flow(xtrain, ytrain, batch_size=BS),
    steps_per_epoch = len(xtrain)//BS,
    validation_data = (xtest, ytest),
    validation_steps = len(xtest)//BS,
    epochs=epochs)

model.save("covid-19.h5")

# Calculating Predictions on the Test Set
predIdxs = model.predict(xtest, batch_size=BS)

# For every data point/image in the Test Set, we find 
# the index of the label with highest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# Printing the Classification Report of the Model
print(classification_report(ytest.argmax(axis=1), predIdxs, target_names=lb.classes_))


# Plotting the Training & Validation Accuracy
train_acc = history.history['acc']
val_acc = history.history['val_acc']
plt.style.use("ggplot")
plt.figure()
plt.plot(train_acc, 'b-', label='Training Accuracy')
plt.plot(val_acc, 'r-', label='Validation Accuracy')
plt.title('Training & Validation Accuracy on COVID-19 Dataset')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig("acc_plot.jpg")
plt.show()


# Plotting the Training & Validation Loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.style.use("ggplot")
plt.figure()
plt.plot(train_loss, 'm-', label='Training loss')
plt.plot(val_loss, 'c-', label='Validation loss')
plt.title('Training & Validation Loss on COVID-19 Dataset')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig("loss_plot.jpg")
plt.show()


# Calculating Accuracy, Sensitivity, and Specificity
cm = confusion_matrix(ytest.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))



