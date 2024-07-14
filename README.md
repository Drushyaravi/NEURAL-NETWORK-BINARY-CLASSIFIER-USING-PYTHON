**Neural Network Binary Classifier**

**Overview:**
This repository contains a Python implementation of a neural network binary classifier, designed to classify input data into one of two categories (0 or 1).

**Files:**
model.py: Contains the implementation of the neural network binary classifier.
data_loader.py: Loads and preprocesses the input data.
trainer.py: Trains the neural network model using the input data.
evaluator.py: Evaluates the performance of the trained model.

**Usage:**
Running the Code
Clone the repository: git clone https://github.com/your-username/neural-network-binary-classifier.git
Navigate to the repository directory: cd neural-network-binary-classifier
Run the trainer: python trainer.py
Evaluate the model: python evaluator.py

**Input Data:**
The input data should be a 2D NumPy array, where each row represents a sample and each column represents a feature. The input data should be normalized before feeding it into the classifier.

**Output:**
The output of the classifier is a binary classification (0 or 1) for each input sample.

Model Architecture:
The neural network model consists of the following layers:
Input Layer: 784 neurons
Hidden Layer: 256 neurons, with ReLU activation
Output Layer: 1 neuron, with Sigmoid activation
Hyperparameters
Learning Rate: 0.01
Batch Size: 128
Number of Epochs: 10

**Dependencies:**
NumPy: The code uses NumPy for numerical computations.
TensorFlow: The code uses TensorFlow as the backend for building and training the neural network model.


Author
M R DRUSHYA
