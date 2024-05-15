''' Using the CIFAR-100 dataset from TensorFlow Datasets
The dataset contains 60,000 32x32 color images
divided into 100 different classes each containing 600 images. 

The small size of the images makes the dataset manageable to work with, even on less powerful hardware. 
'''

# import tensorflow_datasets as tfds
# import tensorflow as tf
# from sklearn.metrics import accuracy_score, recall_score, f1_score
# import numpy as np

# # Load the CIFAR-100 dataset
# dataset, info = tfds.load('cifar100', with_info=True, as_supervised=True)

# # The dataset is a dictionary mapping from split names to tf.data.Dataset instances
# train_dataset = dataset['train']
# test_dataset = dataset['test']
# y_test = np.concatenate([np.array([y]) for x, y in iter(test_dataset)])

# # You can iterate over a tf.data.Dataset instance just like any other Python iterable
# for image, label in train_dataset.take(1):
#     print("Image shape: ", image.shape)
#     print("Label: ", label)

# # Number of categories
# num_classes = info.features['label'].num_classes
# print('Number of categories:', num_classes)

# # Number of training samples
# num_train_samples = info.splits['train'].num_examples
# print('Number of training samples:', num_train_samples)

# # Number of test samples
# num_test_samples = info.splits['test'].num_examples
# print('Number of test samples:', num_test_samples)

# # Define the size of the validation set
# num_validation_samples = 5000

# # Create the validation set from the first 5000 samples of the training set
# validation_dataset = dataset['train'].take(num_validation_samples)

# # Create the new training set from the rest
# train_dataset = dataset['train'].skip(num_validation_samples)

# # Define a function to count the number of elements in a dataset
# def count_elements(dataset):
#     return len(list(dataset))

# # Print the size of each dataset
# print('Train dataset size:', count_elements(train_dataset))
# print('Validation dataset size:', count_elements(validation_dataset))
# print('Test dataset size:', count_elements(test_dataset))

# from InceptionNet import InceptionNetModel

# # Initialize the model
# Inception_model = InceptionNetModel(train_dataset, test_dataset, validation_dataset)

# # Run the model
# Inception_model.run()

# from VGGNet import VGGNetModel

# # Initialize the model
# VGG_model = VGGNetModel(train_dataset, test_dataset, validation_dataset)

# # Run the model
# VGG_model.run()

# from AlexNet import AlexNetModel

# # Initialize the model
# Alex_model = AlexNetModel(train_dataset, test_dataset, validation_dataset)

# # Run the model
# Alex_model.run()

# from ResNet import ResNetModel

# # Initialize the model
# Res_model = ResNetModel()

# # Run the model
# Res_model.run()

# from LeNet import LeNetModel

# # Initialize the model
# Le_model = LeNetModel()

# # Run the model
# Le_model.run()

import tensorflow_datasets as tfds
from InceptionNet import InceptionNetModel
from VGGNet import VGGNetModel
from AlexNet import AlexNetModel
from ResNet import ResNetModel
from LeNet import LeNetModel
import numpy as np

class Main:
    def __init__(self):
        # Load the CIFAR-100 dataset
        self.dataset, self.info = tfds.load('cifar100', with_info=True, as_supervised=True)

        # Define the size of the validation set
        self.num_validation_samples = 5000

        # Create the validation set from the first 5000 samples of the training set
        self.validation_dataset = self.dataset['train'].take(self.num_validation_samples)

        # Create the new training set from the rest
        self.train_dataset = self.dataset['train'].skip(self.num_validation_samples)

        # The dataset is a dictionary mapping from split names to tf.data.Dataset instances
        self.test_dataset = self.dataset['test']

        self.y_test = np.concatenate([np.array([y]) for x, y in iter(self.test_dataset)])

    def run(self):
        # Initialize and run the InceptionNet model
        Inception_model = InceptionNetModel(self.train_dataset, self.test_dataset, self.validation_dataset, self.y_test)
        Inception_model.run()

        # Initialize and run the VGGNet model
        VGG_model = VGGNetModel(self.train_dataset, self.test_dataset, self.validation_dataset, self.y_test)
        VGG_model.run()

        # Initialize and run the AlexNet model
        Alex_model = AlexNetModel(self.train_dataset, self.test_dataset, self.validation_dataset, self.y_test)
        Alex_model.run()

        # Initialize and run the ResNet model
        Res_model = ResNetModel(self.train_dataset, self.test_dataset, self.validation_dataset, self.y_test)
        Res_model.run()

        # Initialize and run the LeNet model
        Le_model = LeNetModel(self.train_dataset, self.test_dataset, self.validation_dataset, self.y_test)
        Le_model.run()

if __name__ == "__main__":
    main = Main()
    main.run()