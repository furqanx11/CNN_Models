import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

class AlexNetModel:
    def __init__(self, train_dataset, test_dataset, validation_dataset, y_test):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset
        self.y_test = y_test

    def preprocess(self, image, label):
        image = tf.image.resize(image, [222, 227])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def augment(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
        return image, label

    def run(self):
        alex_train = self.train_dataset.map(self.preprocess)
        alex_test = self.test_dataset.map(self.preprocess)
        alex_val = self.validation_dataset.map(self.preprocess)

        self.train_dataset = self.train_dataset.map(self.augment)

        AlexNet_model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
            tf.keras.layers.MaxPooling2D((3,3), strides=(2,2)),
            tf.keras.layers.Conv2D(256, (5,5), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((3,3), strides=(2,2)),
            tf.keras.layers.Conv2D(384, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(384, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((3,3), strides=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(100, activation='softmax')
        ])

        AlexNet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        AlexNet_model.fit(alex_train.batch(32), validation_data=alex_val.batch(32), epochs=50)
        AlexNet_preds = AlexNet_model.predict(alex_test.batch(32))
        AlexNet_model.evaluate(alex_test.batch(32))

        Alex_net_labels = np.argmax(AlexNet_preds, axis=1)
        accuracy = accuracy_score(self.y_test, Alex_net_labels)
        recall = recall_score(self.y_test, Alex_net_labels, average = "macro")
        f1 = f1_score(self.y_test, Alex_net_labels, average = "macro")

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Recall:", recall)
        print(f"F1-score:", f1)