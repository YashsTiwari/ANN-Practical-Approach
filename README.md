# **Beginner's Guide to Understanding Artificial Neural Networks (ANNs)**

Welcome to this comprehensive guide designed to help beginners understand Artificial Neural Networks (ANNs) with practical implementation examples. This README complements the provided Jupyter Notebook by explaining core concepts in detail and providing visual representations to make the learning experience enjoyable and intuitive.

---

## **Table of Contents**
1. [Introduction to Neural Networks](#introduction-to-neural-networks)
2. [Core Concepts of ANN](#Core-Concepts-of-ANN)
   - Layers in ANN
   - Activation Functions
   - Forward Propagation
   - Backpropagation
3. [Dataset Preprocessing](#dataset-preprocessing)
4. [Building the ANN](#building-the-ann)
5. [Training the ANN](#training-the-ann)
6. [Evaluating the Model](#evaluating-the-model)
7. [Practical Implementation](#practical-implementation)
8. [Visual Representation of Concepts](#visual-representation-of-concepts)
9. [Further Reading and Resources](#further-reading-and-resources)

---

## **1. Introduction to Neural Networks**

Artificial Neural Networks are computational models inspired by the human brain. They consist of layers of artificial neurons that mimic biological neurons to process information and make decisions.

### **What is a Neural Network?**
A Neural Network consists of:
- **Input Layer**: Where data enters the network.
- **Hidden Layers**: Intermediate layers that process the data.
- **Output Layer**: Provides the final result or prediction.

**Example Use Cases**:
- Spam email detection.
- Predicting customer behavior.
- Image recognition.

---

![Image of Neural Network](https://github.com/YashsTiwari/Building-Basic-ANN/blob/19104d4b92766ce48bf0b4c30a30929d2256e7f4/images/ANN.png)

In this image:
- The **blue circles** represent neurons.
- **Arrows** represent weights connecting neurons between layers.

---

## **2. Core Concepts of ANN**

### **Layers in ANN**
- **Input Layer**: Accepts features from the dataset.
- **Hidden Layers**: Perform transformations to learn patterns.
- **Output Layer**: Outputs predictions or classifications.

---

### **Activation Functions**
Activation functions determine the output of a neuron based on its input. They introduce non-linearity, enabling the network to learn complex patterns.

#### **ReLU (Rectified Linear Unit)**:
- Formula: \( f(x) = max(0, x) \)
- Allows positive values to pass while setting negative values to 0.

**Example**: Imagine water flowing through a pipe. If the valve (input) is negative, no water flows.

![ReLU Activation Function](https://github.com/YashsTiwari/Building-Basic-ANN/blob/19104d4b92766ce48bf0b4c30a30929d2256e7f4/images/ReLU%20Activation%20Function.png)

#### **Sigmoid**:
- Formula: \( f(x) = \frac{1}{1 + e^{-x}} \)
- Maps inputs to a range between 0 and 1, making it suitable for probabilities.

**Example**: Think of it as a probability dial. Any input is converted to a probability score.

![Sigmoid Activation Function](https://github.com/YashsTiwari/Building-Basic-ANN/blob/19104d4b92766ce48bf0b4c30a30929d2256e7f4/images/Sigmoid%20Activation%20Function.png)

---

### **Forward Propagation**
Forward propagation is the process of passing inputs through the network to compute the output.

**Example**: It’s like a conveyor belt in a factory. Raw materials (inputs) move through different stages (layers) to produce a finished product (output).

---

### **Backpropagation**
Backpropagation adjusts the weights in the network to reduce error using the concept of gradients.

**Analogy**: Think of tuning a recipe. If a dish is too salty (high error), adjust the salt (weights) in the next attempt.

![Backpropagation Illustration](https://github.com/YashsTiwari/Building-Basic-ANN/blob/19104d4b92766ce48bf0b4c30a30929d2256e7f4/images/Backpropagation.png)

---

## **3. Dataset Preprocessing**

Before feeding data to the network, it must be cleaned and prepared.

### **Feature Scaling**
Feature scaling ensures all input features have the same scale, preventing larger values from dominating smaller ones.

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### **Train-Test Split**

To evaluate the performance of our ANN, we split the dataset into training and test sets. This ensures that the model is trained on one portion of the data (training set) and tested on another (test set) to simulate unseen scenarios.

#### **Why Split Data?**

- Training Set: Used to train the model by learning patterns and relationships.
- Test Set: Used to evaluate the model’s ability to generalize to new, unseen data.

```
Code Example:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Here:
- test_size=0.2: Allocates 20% of the data to the test set.
- random_state=42: Ensures reproducibility by fixing the random seed.

For practical implementation, refer to the provided notebook.

---

## **4. Building the ANN**

To build the ANN, we define layers using keras.Sequential. Each layer processes data and passes it to the next layer.

```
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units=6, activation='relu', input_dim=10))
classifier.add(Dense(units=6, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
```

Refer to the notebook for the complete implementation.

---

## **5. Training the ANN**

#### **Loss Function**

The loss function measures the difference between predicted and actual values. For binary classification, we use binary cross-entropy.

#### **Optimizer**

The optimizer updates weights to minimize the loss function. We use the Adam optimizer for its efficiency.

#### **Epochs and Batch Size**

- Epochs: Number of times the model sees the entire dataset.
- Batch Size: Number of samples processed at once.

```
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=32, epochs=50)
```
Refer to the notebook for full training details.

---

## **6. Evaluating the Model**

#### **Confusion Matrix**

The confusion matrix compares predicted labels with actual labels.

```
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

Visual:

#### **Accuracy**

Accuracy measures the percentage of correct predictions.

```
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
```

---

## **7. Practical Implementation**

The provided Jupyter Notebook contains the practical implementation of all the concepts discussed here. Key sections include:
- Data preprocessing.
- Model architecture and training.
- Evaluation metrics like confusion matrix and accuracy.

To run the notebook:
	1.	Install required libraries (TensorFlow, Keras, Scikit-learn).
	2.	Load the notebook and follow the step-by-step code.

---

## **8. Further Reading and Resources**

- Keras Documentation
- Scikit-learn Documentation
- Deep Learning by Ian Goodfellow



---

This README serves as a guide to both the theory and practical implementation of Artificial Neural Networks. Use the provided notebook to practice and explore further!

