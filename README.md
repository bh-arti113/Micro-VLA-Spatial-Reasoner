# Micro-VLA-Spatial-Reasoner

🚀 **Live Demo:** https://micro-vla-spatial-reasoner-ferekcgzphv5tj7q9aaehz.streamlit.app/

A small PyTorch CNN that learns to locate coloured squares in synthetic **32×32 RGB images**.

The project is designed as a beginner-friendly introduction to **Computer Vision, CNNs, PyTorch, image tensors, regression, model training, and deployment with Streamlit**.

---

## 🧠 What This Project Does

This project creates simple synthetic images containing coloured squares and trains a small convolutional neural network to predict **where a selected object is located**.

The workflow is:

```text
Synthetic World
      ↓
Generate RGB Image
      ↓
Create Target Coordinates
      ↓
Train CNN
      ↓
Save Model
      ↓
Load Model
      ↓
Select Object Colour
      ↓
Predict Object Position
      ↓
Visualize Prediction
```

The current application allows the user to select:

- 🔴 Red
- 🟢 Green
- 🔵 Blue

The CNN then predicts the position of the selected object.

---

## 🚀 Live Demo

Try the deployed application:

👉 https://micro-vla-spatial-reasoner-ferekcgzphv5tj7q9aaehz.streamlit.app/

The demo allows you to:

1. Generate a synthetic world.
2. Select an object colour.
3. Ask the CNN to locate the selected object.
4. Compare the actual position with the CNN prediction.
5. View the prediction directly on the image.

Example:

```text
Actual Centre:   (5.5, 20.5)
CNN Prediction:  (5.3, 20.3)
Pixel Error:     0.30 px
```

---

# 📁 Project Structure

```text
Micro-VLA-Spatial-Reasoner/
│
├── app.py
├── dataset.py
├── model.py
├── predict.py
├── train.py
├── world.py
│
├── microcnn.pth
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

### File Overview

| File | Purpose |
|---|---|
| `world.py` | Generates synthetic RGB worlds containing coloured squares |
| `dataset.py` | Creates PyTorch training examples |
| `model.py` | Defines the CNN architecture |
| `train.py` | Trains the CNN |
| `predict.py` | Runs predictions interactively |
| `app.py` | Streamlit web application |
| `microcnn.pth` | Saved trained model weights |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `LICENSE` | MIT License |
| `.gitignore` | Files ignored by Git |

---

# 🔍 How It Works

The project is based on a simple idea:

> Give a CNN an image containing a coloured square and teach it to predict the square's centre position.

Each synthetic image is only:

```text
32 × 32 pixels
```

and contains:

```text
3 colour channels
```

which means the image tensor has the shape:

```text
[3, 32, 32]
```

The three channels represent:

```text
Channel 0 → Red
Channel 1 → Green
Channel 2 → Blue
```

During training, the target position is represented using normalized coordinates.

For a 32×32 image:

```text
x_normalized = x / 31
y_normalized = y / 31
```

The model learns to output values between approximately:

```text
0 → left / top
1 → right / bottom
```

The predicted normalized coordinates are converted back into pixel coordinates for visualization.

---

# 🌍 Synthetic World

Instead of using a real-world dataset, this project generates its own images.

A typical world contains coloured squares such as:

```text
🔴 Red square
🟢 Green square
🔵 Blue square
```

The squares are placed at random positions.

For example:

```text
32 × 32 Image

┌──────────────────────────────┐
│                              │
│       🔵                     │
│                              │
│                    🟢        │
│                              │
│   🔴                         │
│                              │
└──────────────────────────────┘
```

Because the positions are generated automatically, thousands of training examples can be created without manually labelling images.

---

# 🧪 Dataset Generation

The dataset creates examples containing:

```text
Image
+
Target Position
```

A simplified example is:

```python
image = torch.zeros(3, 32, 32)

x = random_x
y = random_y

image[colour, y:y+4, x:x+4] = 1.0

center_x = x + 1.5
center_y = y + 1.5
```

The square is 4×4 pixels, so its centre is calculated from its starting position.

The target is then normalized before being given to the neural network.

---

# 🧠 CNN Architecture

The project uses a small convolutional neural network called:

```text
MicroCNN
```

The basic architecture contains:

```text
Input Image
[3 × 32 × 32]
      ↓
Conv2D
3 → 8 channels
      ↓
ReLU
      ↓
Flatten
      ↓
Fully Connected Layer
8192 → 128
      ↓
Position Prediction
```

The convolution layer acts like a collection of small visual detectors.

Each detector can learn to respond to patterns in the image.

For example, some filters may become useful for detecting:

```text
edges
corners
bright regions
square-like patterns
colour patterns
```

---

# 👀 Understanding the Tensor Shapes

The input image has:

```text
[3, 32, 32]
```

After batching with a PyTorch `DataLoader`, the shape becomes:

```text
[batch_size, 3, 32, 32]
```

For example:

```text
[32, 3, 32, 32]
```

means:

```text
32 images
3 colour channels
32 pixel height
32 pixel width
```

After the convolution layer:

```text
[32, 8, 32, 32]
```

The CNN now has:

```text
8 learned feature maps
```

These are flattened before being passed to the fully connected layers.

---

# 🎯 Position Prediction

The main task is **regression**.

Instead of predicting a class such as:

```text
cat
dog
car
```

the network predicts numerical coordinates:

```text
x
y
```

For example:

```text
Prediction:
[0.177, 0.655]
```

These normalized values can be converted back into pixel coordinates:

```text
x_pixel = x_normalized × 31
y_pixel = y_normalized × 31
```

For example:

```text
Actual Centre:
(5.5, 20.5)

CNN Prediction:
(5.3, 20.3)
```

The difference between these values can be used to calculate the prediction error in pixels.

---

# 🎨 Selecting an Object

The Streamlit application allows the user to choose the object colour:

```text
🔴 Red
🟢 Green
🔵 Blue
```

After selecting a colour, the application runs the trained model and displays the predicted position.

The selected colour determines which object position is being evaluated in the generated world.

The current application focuses on **spatial localization** rather than using a separate colour-classification output head.

---

# 🏋️ Training

The model is trained using:

```text
PyTorch
+
DataLoader
+
Mean Squared Error
+
Adam Optimizer
```

The basic training loop follows:

```python
optimizer.zero_grad()

prediction = model(images)

loss = criterion(prediction, targets)

loss.backward()

optimizer.step()
```

### What happens here?

### 1. Forward Pass

The image is passed through the CNN.

```text
Image
 ↓
CNN
 ↓
Prediction
```

### 2. Calculate Loss

The prediction is compared with the correct target.

```text
Prediction
    ↓
Compare
    ↓
Actual Target
    ↓
Loss
```

### 3. Backpropagation

The loss is used to calculate how the model's parameters should change.

```python
loss.backward()
```

### 4. Optimizer Step

The Adam optimizer updates the model parameters.

```python
optimizer.step()
```

The goal is to gradually reduce the loss.

---

# 📉 Understanding Loss

The project uses **Mean Squared Error (MSE)** for the position prediction.

A simplified idea is:

```text
Loss = average(prediction - target)²
```

A smaller loss generally means the predicted coordinates are closer to the target coordinates.

For example:

```text
3.6e-05
```

means:

```text
0.000036
```

Scientific notation is commonly used because neural-network losses can become very small.

---

# ⚙️ Optimizer

The optimizer is responsible for updating the neural network's learnable parameters.

This project uses:

```text
Adam
```

Think of the model parameters as thousands of small adjustable knobs.

Training repeatedly adjusts those knobs so that:

```text
Prediction → closer to Target
```

and therefore:

```text
Loss → smaller
```

---

# 💾 Saving the Model

After training, the learned parameters are saved using:

```python
torch.save(model.state_dict(), "microcnn.pth")
```

The file:

```text
microcnn.pth
```

contains the trained model weights.

The model architecture is defined in:

```text
model.py
```

while the learned parameters are stored in:

```text
microcnn.pth
```

---

# 📥 Loading the Model

The saved model can later be loaded without training again.

```python
model.load_state_dict(
    torch.load("microcnn.pth")
)
```

This allows the trained model to be used for prediction and deployment.

---

# 🔮 Prediction

The prediction script loads the trained model and evaluates a generated image.

The model is placed into evaluation mode:

```python
model.eval()
```

A single image is given a batch dimension using:

```python
image.unsqueeze(0)
```

So:

```text
[3, 32, 32]
```

becomes:

```text
[1, 3, 32, 32]
```

The model then produces the predicted coordinates.

---

# 🎨 Visualization

The project uses Matplotlib to visualize the generated world.

The image tensor is converted from PyTorch's format:

```text
[C, H, W]
```

to Matplotlib's expected format:

```text
[H, W, C]
```

using:

```python
image.permute(1, 2, 0)
```

The application then displays:

```text
Actual Position
       ↓
      ●

CNN Prediction
       ↓
      ●
```

and connects them visually so the prediction error can be seen directly.

---

# 🌐 Streamlit Application

The project includes a web interface built with:

```text
Streamlit
```

The application provides:

- Object colour selection
- Synthetic world generation
- CNN prediction
- Actual centre coordinates
- CNN predicted coordinates
- Pixel error
- Visual prediction overlay

The interface is designed to make the CNN's behaviour easy to understand visually.

---

# ▶️ Running Locally

Clone the repository:

```bash
git clone https://github.com/bh-arti113/Micro-VLA-Spatial-Reasoner.git
cd Micro-VLA-Spatial-Reasoner
```

Create a virtual environment:

```bash
python -m venv myenv
```

Activate it.

### Windows

```bash
myenv\Scripts\activate
```

### macOS / Linux

```bash
source myenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🏋️ Train the Model

Run:

```bash
python train.py
```

This generates training data, trains the CNN, and saves the learned parameters.

The trained model is saved as:

```text
microcnn.pth
```

---

# 🔮 Run Prediction

Run:

```bash
python predict.py
```

The script loads the trained model and allows interactive prediction.

---

# 🌐 Run the Streamlit App

Run:

```bash
streamlit run app.py
```

Streamlit will start a local web server and provide a URL where the application can be opened in a browser.

---

# 🛠️ Technologies Used

- 🐍 Python
- 🔥 PyTorch
- 🧠 Convolutional Neural Networks
- 📊 NumPy / Tensor operations
- 📈 Matplotlib
- 🌐 Streamlit
- 🗂️ GitHub

---

# 📚 Learning Goals

This project was built to understand the fundamentals of neural networks through a small and visual problem.

The main concepts explored are:

### PyTorch

- Tensors
- `Dataset`
- `DataLoader`
- Neural network modules
- Forward pass
- Backpropagation
- Optimizers
- Model saving and loading

### Computer Vision

- RGB images
- Image tensors
- Convolution
- Feature maps
- Spatial localization
- Coordinate prediction

### Machine Learning

- Training data
- Targets
- Loss functions
- Regression
- Optimization
- Model evaluation

### Deployment

- Streamlit applications
- GitHub repositories
- Streamlit Community Cloud
- Loading trained model weights in a deployed application

---

# 🧩 Key Concepts Learned

## 1. Images Are Tensors

A colour image can be represented as:

```text
[Channels, Height, Width]
```

For this project:

```text
[3, 32, 32]
```

---

## 2. CNNs Learn Visual Features

Convolutional layers allow the network to learn useful patterns from images.

Instead of manually programming:

```text
"look for a square here"
```

the CNN learns useful features from training examples.

---

## 3. Regression Predicts Numbers

The network predicts coordinates rather than a simple class label.

```text
(x, y)
```

This makes the project a small example of **visual localization**.

---

## 4. Training Changes Model Parameters

The optimizer repeatedly updates the model's parameters based on the loss.

```text
Input
 ↓
Prediction
 ↓
Loss
 ↓
Backpropagation
 ↓
Parameter Update
 ↓
Better Prediction
```

---

## 5. Normalization Helps Neural Networks

Coordinates are normalized from pixel values into approximately:

```text
0 → 1
```

This gives the network a consistent numerical range for the target values.

---

# 📊 Example Result

One example prediction from the application:

```text
Actual Centre:   (5.5, 20.5)

CNN Prediction:  (5.3, 20.3)

Pixel Error:     0.30 px
```

This shows that the CNN can learn the spatial relationship between the image and the object's position on this synthetic task.

---

# 🚧 Project Status

### ✅ Completed

- Synthetic RGB world generation
- Random object placement
- PyTorch dataset creation
- CNN architecture
- Position regression
- Model training
- Model saving
- Model loading
- Interactive prediction
- Matplotlib visualization
- Streamlit interface
- GitHub repository
- Streamlit Community Cloud deployment

### 🔜 Possible Future Improvements

- Add a dedicated colour-classification head
- Train on more complex scenes
- Add multiple objects of the same colour
- Use larger and more varied images
- Add data augmentation
- Improve CNN architecture
- Add confidence estimates
- Experiment with larger datasets
- Extend the project toward more general visual reasoning

---

# 💡 Why This Project?

The goal was not to build a huge production-scale computer vision system.

The goal was to understand the complete machine-learning pipeline:

```text
Generate Data
      ↓
Prepare Dataset
      ↓
Build Neural Network
      ↓
Train Model
      ↓
Measure Loss
      ↓
Save Weights
      ↓
Load Model
      ↓
Make Predictions
      ↓
Visualize Results
      ↓
Deploy Application
```

Building this small project makes each stage of the pipeline easier to inspect and understand.

---

# 👨‍💻 About

This project was created as a hands-on learning project for understanding **PyTorch, CNNs, computer vision, spatial localization, and model deployment**.

The emphasis is on learning by building a complete working system from synthetic data to a live web application.

---

# 📜 License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for the full license text.

---

⭐ If you find this project useful for learning, feel free to explore the code and experiment with the model.
