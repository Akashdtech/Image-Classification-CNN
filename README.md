# Image-Classification-CNN
This project implements an image classification pipeline using the CIFAR-10 dataset. The pipeline preprocesses image data, defines a Convolutional Neural Network (CNN) model, and evaluates its performance.

Data Preprocessing:

    Normalization of image pixel values by dividing by 255.0 to scale them between 0 and 1.

Data Visualization:

    Visualization of individual and multiple sample images from the training set using matplotlib to understand the data.

Model Definition:

    A CNN model is defined using TensorFlow/Keras.
    Two Conv2D layers with ReLU activation and MaxPooling2D layers for feature extraction.
    A Flatten layer to reshape the output for the fully connected layers.
    Dense layers for classification: one with 64 units and ReLU activation, and the output layer with 10 units and softmax activation.

Model Training:

    The model is compiled with the Adam optimizer and SparseCategoricalCrossentropy loss function, then trained for 10 epochs using the training data (X_train, y_train).

Model Evaluation:

    The model is evaluated on the test data (X_test, y_test) to check its accuracy.

Performance Metrics:

    Accuracy and classification report are generated using scikit-learn’s accuracy_score and classification_report to assess model performance.

This repository provides a straightforward implementation of image classification using a CNN, showcasing the process of training and evaluating a deep learning model on the CIFAR-10 dataset.
