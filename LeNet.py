import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np

class LeNetModel:
    def __init__(self, train_dataset, test_dataset, validation_dataset, y_test):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset
        self.y_test = y_test

    def preprocess(self, image, label):
        # Resize the image
        image = tf.image.resize(image, [32, 32])
        # Normalize the image
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def augment(self, image, label):
        # Augmentation
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
        return image, label

    def run(self):
        # Apply the preprocessing function to the datasets
        le_train = self.train_dataset.map(self.preprocess)
        le_test = self.test_dataset.map(self.preprocess)
        le_val = self.validation_dataset.map(self.preprocess)

        # Apply the augmentation function only to the training set
        self.train_dataset = self.train_dataset.map(self.augment)

        LE_model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(6, (5,5), activation='tanh', input_shape=(32,32,3), padding='same'),
            tf.keras.layers.AveragePooling2D(),
            tf.keras.layers.Conv2D(16, (5,5), activation='tanh', padding='valid'),
            tf.keras.layers.AveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='tanh'),
            tf.keras.layers.Dense(84, activation='tanh'),
            tf.keras.layers.Dense(100, activation='softmax')
        ])

        LE_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        LE_model.fit(le_train.batch(32), validation_data=le_val.batch(32), epochs=50)
        LE_preds = LE_model.predict(le_test.batch(32))
        LE_model.evaluate(le_test.batch(32))

        LE_labels = np.argmax(LE_preds, axis=1)
        accuracy = accuracy_score(self.y_test, LE_labels)
        recall = recall_score(self.y_test, LE_labels, average = None)
        f1 = f1_score(self.y_test, LE_labels, average = None)
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")