import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

class InceptionNetModel:
    def __init__(self, train_dataset, test_dataset, validation_dataset, y_test):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset
        self.y_test = y_test

    def preprocess(self, image, label):
        image = tf.image.resize(image, [299, 299])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def augment(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
        return image, label

    def run(self):
        inc_train = self.train_dataset.map(self.preprocess)
        inc_test = self.test_dataset.map(self.preprocess)
        inc_val = self.validation_dataset.map(self.preprocess)

        self.train_dataset = self.train_dataset.map(self.augment)

        inc_model = tf.keras.applications.InceptionV3(weights=None, input_shape=(299, 299, 3), classes=100)
        inc_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        inc_model.fit(inc_train.batch(32), validation_data=inc_val.batch(32), epochs=30)
        inc_preds = inc_model.predict(inc_test.batch(32))
        inc_model.evaluate(inc_test.batch(32))

        inc_labels = np.argmax(inc_preds, axis=1)
        accuracy = accuracy_score(self.y_test, inc_labels)
        recall = recall_score(self.y_test, inc_labels, average = None)
        f1 = f1_score(self.y_test, inc_labels, average = None)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")