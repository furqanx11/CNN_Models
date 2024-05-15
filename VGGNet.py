import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

class VGGNetModel:
    def __init__(self, train_dataset, test_dataset, validation_dataset, y_test):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset
        self.y_test = y_test

    def preprocess(self, image, label):
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def augment(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
        return image, label

    def run(self):
        vgg_train = self.train_dataset.map(self.preprocess)
        vgg_test = self.test_dataset.map(self.preprocess)
        vgg_val = self.validation_dataset.map(self.preprocess)

        self.train_dataset = self.train_dataset.map(self.augment)

        VGG_model = tf.keras.applications.VGG16(weights=None, input_shape=(224, 224, 3), classes=100)
        VGG_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        VGG_model.fit(vgg_train.batch(32), validation_data=vgg_val.batch(32), epochs=30)
        VGG_preds = VGG_model.predict(vgg_test.batch(32))
        VGG_model.evaluate(vgg_test.batch(32))

        VGG_preds_classes = np.argmax(VGG_preds, axis=1)
        accuracy = accuracy_score(self.y_test, VGG_preds_classes)
        recall = recall_score(self.y_test, VGG_preds_classes, average = None)
        f1 = f1_score(self.y_test, VGG_preds_classes, average = None)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")