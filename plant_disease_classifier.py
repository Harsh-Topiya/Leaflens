import kagglehub
abdallahalidev_plantvillage_dataset_path = kagglehub.dataset_download('abdallahalidev/plantvillage-dataset')

print('Data source import complete.')

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import shutil
import tempfile

# === CONFIGURATION ===
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
BASE_DIR = '/kaggle/input/plantvillage-dataset/plantvillage dataset'

# === TEMP MERGE DIRECTORY ===
temp_dir = tempfile.mkdtemp()

# Merge Color, Grayscale, and Segmented into one directory for training
folders_to_merge = ['color', 'grayscale', 'segmented']

print("Merging folders...")
for folder in folders_to_merge:
    folder_path = os.path.join(BASE_DIR, folder)
    for class_dir in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_dir)
        if os.path.isdir(class_path):
            dest_class_path = os.path.join(temp_dir, class_dir)
            os.makedirs(dest_class_path, exist_ok=True)
            for img in os.listdir(class_path):
                src = os.path.join(class_path, img)
                dst = os.path.join(dest_class_path, f"{folder}_{img}")
                shutil.copyfile(src, dst)

print("Merging completed.")

# === DATA PREPROCESSING ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    temp_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    temp_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === DEEP CNN MODEL ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# === MODEL COMPILATION ===
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === TRAIN MODEL ===
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# === EVALUATE MODEL ===
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")

# === SAVE MODEL ===
model.save('plant_disease_model_combined.h5')
print("Model saved as plant_disease_model_combined.h5")


