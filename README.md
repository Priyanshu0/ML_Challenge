# 🚀 ML Engineer @ Emissary 🚀

## 🥁 Background

An AI infrastructure platform that enables companies build extraordinary proprietary AI capabilities that withstand the test of time. They're a well-funded team with extensive production-grade AI development experience focused on making the development and dissemination of AI as smooth and rapid as humanly possible and are passionate about bringing the benefits of AI to the 99%.

### 🛠️ Setup

- Laptop or machine or cloud instance with GPU and a stable internet connection.
- Setup deep learning required environment (packages like python, tensorflow, keras, etc).
- Download the dataset ([link](https://drive.google.com/file/d/1ZZ2TvL-UhdLbUOyQgYPMRiAG3APxPRp_/view?usp=sharing)) mentioned below and the data loader script handy to avoid delay.

## ⛔ Thing to keep in mind

- Chatgpt use is not allowed
- The maximum allotted time for this challenge is 2.5 hrs. Pls manage your time accordingly.
- keep your camera on throughout the challenge
- share your entire screen throughout the challenge
- once you complete the challenge, please ring us at 9560444655 to assist you with your response & demo recording
- once you are done, push your project on a github link and WhatsApp it to the above number

## **🧠 What's the Mission?**

Welcome to this klimbB challenge! This challenge is designed for ML Engineers who are keen to demonstrate their expertise in deep learning and large language models models. End goal of this challenge is to build a model for classification/prediction and which can be optimized to be able to deploy/use in devices with low memory requirements without sacrificing the model performance.

### 📜 Asks

Here are the steps to be followed for creating solution for the given challenge:

1. Using provided pre-trained model to train/learn a large complex model on the task of interest. This model will serve as the *“master”* model *(Starter code is provided below for training master model, you are **not** expected to improve the performance of this model and will serve as base).*
2. Train a smaller and simpler *“student”* model using *“master”* model, with an objective to reduce the complexity, latency and memory size of the model without losing much of a performance. *(Building the “student” model from the “master” model is your main objective of this challenge).*
3. Perform optimization of *“student”* model by minimizing difference between the output of the *“master”* model and the *“student”* model (choice of optimization technique is up to you).
4. Evaluate the performance of both the models on the set of test data to determine how well it performs on the task of interest (refer “Success Metric” section).

***Important Note:*** A code snippet is provided below for training “master” model (*Asks Step-1*).

*It's worth noting that the exact steps for creating a model and optimization techniques can vary depending on the specific task and models being used.* 

### 🌱 Seed Data

Data can be downloaded from the following link: 

This data contains around 25k images of size 150x150 distributed under 6 categories.
*[ buildings,  forest,  glacier,  mountain,  sea,  street ].*

The Training and Testing data is already separated in the zip file provided. 
There are around 14k images in the Training and 3k in the Testing data.

***Important Note:*** A code snippet is provided below for loading the training and testing data.

### **💡 Starter code for loading train-test data and training “master” model**

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Path of train and test data
train_dir = "../path_to_directory/seg_train/"
test_dir = "../path_to_directory/seg_test/"

# Data configs
batch_size = 32
img_height = 150
img_width = 150

# Load train data
train_ds = tf.keras.utils.image_dataset_from_directory(
	train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Load test data
test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Training the MASTER Model - using Transfer Learning
# Here we are using ImageNet pre-trained model weights
base_model = keras.applications.ResNet152(
		weights='imagenet',  # Load weights pre-trained on ImageNet.
		input_shape=(img_height, img_width, 3),
		include_top=False)  # Do not include the ImageNet classifier at the top.
base_model.trainable = False
inputs = keras.Input(shape=(img_height, img_width, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning.
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(6)(x)
model = keras.Model(inputs, outputs)
model.summary()
model.compile(
		optimizer=keras.optimizers.Adam(),
		loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

epochs = 20
model.fit(train_ds, epochs=epochs)

# Generate results on test data
results = model.evaluate(test_ds)
print(f"Test accuracy with trained teacher model:{results[1]*100 :.2f} %")
```

### 🏁 Evaluation and Success Metrics

1. **Model Size Ratio:** Ratio = master model size / student model size.
2. **Parameter Ratio:** Ratio = master model parameter / student model parameter.
3. **Accuracy:** Both of master and student model should provide accurate responses.
4. **Latency:** Time taken from feeding an input (single image) to receiving an output (prediction).
5. **End-to-End Functionality:** The entire pipeline, from the model building to the final prediction from both master and student models should be operational without any errors.

### ⏱️ Estimates

The estimated time to finish this challenge is 2 hours, breakup is as follows:

1. Loading the train and test data setup - 30 mins.
2. Building and training the *“master”* model - 1 hrs.
3. Building and training the *“student”* model - 1 hrs.
4. Evaluating and comparing performances - 30 mins.

## **💡 Quick Tips**

Before submitting, run your solution through the success metrics to ensure it meets the objectives of the challenge. Create a compelling demo to share your work with impact.

##
