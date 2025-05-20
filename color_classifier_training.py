import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import time

IMG_WIDTH = 64
IMG_HEIGHT = 64
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 30
MODEL_SAVE_PATH = 'saved_models/custom_color_classifier.keras'
TRAIN_DATA_DIR = 'data/color_classification/train'
TEST_DATA_DIR = 'data/color_classification/test'
TASK_NAME = "Car Color"

if not os.path.exists(TRAIN_DATA_DIR) or not os.path.exists(TEST_DATA_DIR):
    print(f"ERROR: Training ({TRAIN_DATA_DIR}) or Testing ({TEST_DATA_DIR}) data directory not found.")
    exit()

try:
    CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DATA_DIR) if os.path.isdir(os.path.join(TRAIN_DATA_DIR, d))])
    if not CLASS_NAMES:
        raise FileNotFoundError("No class subdirectories found in training data folder.")
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"Found {NUM_CLASSES} {TASK_NAME.lower()} classes: {CLASS_NAMES}")
except FileNotFoundError as e:
    print(f"Error accessing data directories: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest', validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

try:
    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', color_mode='rgb', classes=CLASS_NAMES)

    print("Loading validation data...")
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', color_mode='rgb', classes=CLASS_NAMES)

    print("Loading test data...")
    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=1,
        class_mode='categorical', shuffle=False, color_mode='rgb', classes=CLASS_NAMES)

    if train_generator.samples == 0 or validation_generator.samples == 0:
         print("ERROR: No training or validation images found/loaded. Check dataset paths and structure.")
         exit()
    if test_generator.samples == 0:
         print("WARNING: No test images found/loaded. Evaluation will be skipped.")

except Exception as e:
    print(f"Error during data loading: {e}")
    exit()

input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

def build_color_classifier(input_shape, num_classes):
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    print(f"--- Training {TASK_NAME} Classifier ---")
    model = build_color_classifier(input_shape, NUM_CLASSES)
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    print(f"Starting {TASK_NAME} training...")
    start_time = time.time()
    history = model.fit(
        train_generator, steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        validation_data=validation_generator, validation_steps=max(1, validation_generator.samples // BATCH_SIZE),
        epochs=EPOCHS, callbacks=[checkpoint, early_stopping]
    )
    training_time = time.time() - start_time
    print(f"{TASK_NAME} training finished in {training_time:.2f} seconds.")

    if test_generator.samples > 0:
        print(f"\nEvaluating {TASK_NAME} classifier...")
        try:
            best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        except Exception as e:
            print(f"Could not load best model, using last epoch model: {e}")
            best_model = model

        test_loss, test_acc = best_model.evaluate(test_generator, steps=test_generator.samples)
        print(f'\n{TASK_NAME} Test Accuracy: {test_acc:.4f}')

        if test_acc < 0.70:
             print(f"WARNING: {TASK_NAME} model accuracy is below the 70% minimum requirement!")
        else:
             print(f"{TASK_NAME} model accuracy meets the minimum 70% requirement.")

        print("\nGenerating Classification Report and Confusion Matrix...")
        test_generator.reset()
        y_pred = best_model.predict(test_generator, steps=test_generator.samples)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes

        report = classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES, zero_division=0)
        matrix = confusion_matrix(y_true, y_pred_classes)
        print(f'\n{TASK_NAME} Classification Report:')
        print(report)
        print(f'\n{TASK_NAME} Confusion Matrix:')
        print(matrix)
    else:
        print("\nSkipping evaluation as no test data was found.")

    print(f"\n{TASK_NAME} model training complete. Best model potentially saved to {MODEL_SAVE_PATH}")