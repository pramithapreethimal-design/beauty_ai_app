from google.colab import drive
drive.mount('/content/drive')

import os
os.listdir("/content/drive/MyDrive")

os.listdir("/content/drive/MyDrive/Colab_Datasets")

dataset_path = "/content/drive/MyDrive/Colab_Datasets/archive.zip"
print(os.path.exists(dataset_path))

import zipfile

extract_path = "/content/drive/MyDrive/Colab_Datasets"

with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

os.listdir(extract_path)

dataset_path = "/content/drive/MyDrive/Colab_Datasets/Oily-Dry-Skin-Types"

import os
os.listdir(dataset_path)

!pip install tensorflow opencv-python



import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_dir = dataset_path + "/train"
valid_dir = dataset_path + "/valid"

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

train_generator.class_indices

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=valid_generator
)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    dataset_path + "/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator)
print("Test Accuracy:", test_acc)

model.save("skin_type_model.h5")

from google.colab import files
files.download("skin_type_model.h5")
