import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from model import MicroCNN
from world import make_world



# Load trained model


model = MicroCNN()

model.load_state_dict(
    torch.load("microcnn.pth")
)

model.eval()


# Generate new world


image, target = make_world()

image_batch = image.unsqueeze(0)

# Ask user what to find

colour_names = [
    "Red",
    "Green",
    "Blue"
]

print()
print("Which colour do you want to find?")
print("0 = Red")
print("1 = Green")
print("2 = Blue")

choice = int(input("Enter 0, 1 or 2: "))

# Prediction

with torch.no_grad():

    prediction = model(image_batch)[0]


# Select requested colour
actual_position = target[choice]
predicted_position = prediction[choice]


# Convert normalised coordinates back to pixels
actual_x = actual_position[0].item() * 31
actual_y = actual_position[1].item() * 31

predicted_x = predicted_position[0].item() * 31
predicted_y = predicted_position[1].item() * 31

# Print results

print()
print("You asked for:", colour_names[choice])

print(
    "Actual position:",
    actual_position
)

print(
    "Predicted position:",
    predicted_position
)

print()
print(
    "Actual pixel position:",
    actual_x,
    actual_y
)

print(
    "Predicted pixel position:",
    predicted_x,
    predicted_y
)

# Display image

image_display = image.permute(1, 2, 0)

fig, ax = plt.subplots()

ax.imshow(image_display)

ax.set_title(
    f"CNN asked to find: {colour_names[choice]}"
)

predicted_box = Rectangle(
    (predicted_x - 2, predicted_y - 2),
    4,
    4,
    fill=False,
    linewidth=3
)

ax.add_patch(predicted_box)

# Put a marker at CNN's predicted centre
ax.scatter(
    predicted_x,
    predicted_y,
    marker="+",
    s=150,
    linewidths=3
)

# Text below image
fig.text(
    0.5,
    0.02,
    f"Found {colour_names[choice]} at "
    f"({predicted_x:.1f}, {predicted_y:.1f})",
    ha="center"
)

plt.show()