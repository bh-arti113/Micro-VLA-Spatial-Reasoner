# Micro-VLA-Spatial-Reasoner

A small PyTorch CNN that detects the **position and colour of squares** in synthetic 32×32 RGB images.

## 🧠 What this project does

This project generates simple synthetic images containing coloured squares and trains a convolutional neural network (CNN) to answer two questions:

1. **Where is the square?** → predicts its `(x, y)` position
2. **What colour is it?** → predicts red, green, or blue

The model is trained entirely on generated data, so no external dataset is required.

The project was built as a hands-on introduction to computer vision and deep learning using PyTorch.

## 🏗️ Project Structure

```text
Micro-VLA-Spatial-Reasoner/
├── world.py          # Generates synthetic images
├── dataset.py        # PyTorch dataset
├── model.py          # CNN architecture
├── train.py          # Model training
├── predict.py        # Interactive prediction
├── requirements.txt  # Python dependencies
└── .gitignore        # Files ignored by Git
```

## 🔬 How it works

The overall pipeline is:

```text
Synthetic image
      ↓
CNN convolution layer
      ↓
Feature extraction
      ↓
Flatten
      ↓
Fully connected layers
      ↓
    128 features
      ↓
 ┌───────────────┐
 ↓               ↓
Position head   Colour head
 ↓               ↓
(x, y)         Red/Green/Blue
```

The CNN receives a **32×32 RGB image** represented as a PyTorch tensor with shape:

```text
[3, 32, 32]
```

The `3` represents the RGB colour channels.

The convolution layer learns multiple feature detectors that respond to useful visual patterns in the image.

The extracted features are then flattened and passed through fully connected layers.

Finally, the network produces two outputs:

- A position prediction
- A colour prediction

## 📍 Position Prediction

The position task is treated as a **regression problem**.

The model predicts normalized `(x, y)` coordinates.

The target coordinates are normalized between `0` and `1` so that the neural network can learn the position more easily.

The prediction can then be converted back into pixel coordinates.

## 🎨 Colour Prediction

The colour task is treated as a **classification problem**.

The model learns to distinguish between:

```text
0 → Red
1 → Green
2 → Blue
```

The user can select which colour to find, and the program displays the model's predicted position for that colour.

## 🧪 Synthetic Data

Instead of downloading a large computer-vision dataset, this project generates its own training data.

Each image contains coloured squares placed at different positions.

This makes it possible to experiment with:

- Image tensors
- RGB channels
- CNNs
- Training
- Regression
- Classification
- Loss functions
- Backpropagation

without needing an external dataset.

## 🚀 Running the Project

### 1. Install dependencies

Create and activate a Python virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Train the model

Run:

```bash
python train.py
```

The training script generates synthetic training examples and trains the CNN.

The learned model weights are saved as:

```text
microcnn.pth
```

This file is intentionally excluded from the GitHub repository using `.gitignore`.

### 3. Run predictions

After training, run:

```bash
python predict.py
```

The program generates a synthetic image and allows the user to select a colour to find.

The model then predicts the approximate position of that colour.

## 🛠️ Technologies

- **Python**
- **PyTorch**
- **Convolutional Neural Networks (CNNs)**
- **Matplotlib**
- **Synthetic data generation**

## 📚 Learning Goals

This project was built to understand the fundamentals of:

- Image tensors
- RGB colour channels
- Convolution layers
- Feature extraction
- ReLU activation
- Flattening tensors
- Fully connected layers
- Regression vs classification
- Loss functions
- Backpropagation
- Optimizers
- Model training
- Model inference
- Saving and loading PyTorch models

## 🧠 Key Concepts Learned

### Image tensor

A single RGB image is represented as:

```text
[3, 32, 32]
```

where:

```text
3  → RGB channels
32 → image height
32 → image width
```

### CNN feature extraction

The convolution layer learns visual patterns from the image.

For example, different learned filters can become sensitive to patterns such as edges, colours, or square-like structures.

### Loss

The loss measures how different the model's prediction is from the correct target.

During training, the goal is to reduce the loss.

### Backpropagation

Backpropagation calculates how the model's parameters contributed to the error.

The optimizer then uses this information to update those parameters.

### Inference

After training, the model can be used to make predictions on new synthetic images.

## 📊 Example Result

The trained model successfully learned to identify coloured squares and predict their approximate spatial positions.

For example, a prediction may look conceptually like:

```text
Requested colour: Blue

Actual position:     (15.5, 21.5)
Predicted position:  (15.4, 21.2)
```

The exact values vary because the images are generated randomly.

## 📁 Repository Files

| File | Purpose |
|------|---------|
| `world.py` | Generates synthetic images |
| `dataset.py` | Creates the PyTorch dataset |
| `model.py` | Defines the CNN |
| `train.py` | Trains the model |
| `predict.py` | Runs interactive predictions |
| `requirements.txt` | Lists required Python packages |
| `.gitignore` | Prevents unnecessary files from being uploaded |

## 🎯 Project Status

**Completed — beginner computer vision / CNN project**

The project successfully demonstrates an end-to-end machine-learning workflow:

```text
Generate data
     ↓
Create dataset
     ↓
Build CNN
     ↓
Train model
     ↓
Save model
     ↓
Load model
     ↓
Make predictions
     ↓
Visualize results
```

## 🔮 Possible Future Improvements

Some possible next steps for the project are:

- Add object-size prediction
- Improve the visualization
- Add validation and test datasets
- Track training loss with plots
- Add a confusion matrix for colour classification
- Experiment with deeper CNN architectures
- Train on real images instead of only synthetic data

## 👨‍💻 About

This project was created as a hands-on learning project to understand how a CNN can learn visual information and make spatial predictions using PyTorch.
