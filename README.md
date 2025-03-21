# MNIST Image Classification with OOP

## Overview
This project implements three different classification models for the MNIST dataset using Object-Oriented Programming (OOP). The models include:
1. **Random Forest** (RF)
2. **Feed-Forward Neural Network** (NN)
3. **Convolutional Neural Network** (CNN)

Each model follows a common interface, `MnistClassifierInterface`, which defines two methods:
- `train(X_train, y_train)`: Trains the model using the provided dataset.
- `predict(X_test)`: Predicts labels for the given test dataset.

A wrapper class, `MnistClassifier`, allows switching between models using an input parameter.

## Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Training a Model
```python
from mnist_classifier import MnistClassifier

# Initialize a classifier with an algorithm ('rf', 'nn', or 'cnn')
classifier = MnistClassifier(algorithm='cnn')
classifier.train(X_train, y_train)
```

### Making Predictions
```python
predictions = classifier.predict(X_test)
```

## Models Overview
- **Random Forest (`rf`)**: A tree-based ensemble learning model for classification.
- **Feed-Forward Neural Network (`nn`)**: A fully connected neural network trained using backpropagation.
- **Convolutional Neural Network (`cnn`)**: A deep learning model specialized for image classification.

## License
This project is open-source and available under the MIT License.

