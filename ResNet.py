import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np

class ResNetModel:
    def __init__(self, train_dataset, test_dataset, validation_dataset, y_test):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset
        self.y_test = y_test

    def preprocess(self, image, label):
        # Resize the image
        image = tf.image.resize(image, [223, 224])
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
        res_train = self.train_dataset.map(self.preprocess)
        res_test = self.test_dataset.map(self.preprocess)
        res_val = self.validation_dataset.map(self.preprocess)

        # Apply the augmentation function only to the training set
        self.train_dataset = self.train_dataset.map(self.augment)

        res_model = tf.keras.applications.ResNet50(weights=None, input_shape=(224, 224, 3), classes=100)
        res_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        res_model.fit(res_train.batch(32), validation_data=res_val.batch(32), epochs=30)
        res_preds = res_model.predict(res_test.batch(32))
        res_model.evaluate(res_test.batch(32))

        RES_labels = np.argmax(res_preds, axis=1)
        accuracy = accuracy_score(self.y_test, RES_labels)
        recall = recall_score(self.y_test, RES_labels, average = None)
        f1 = f1_score(self.y_test, RES_labels, average = None)
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")