import torch


def make_world():

    # Empty RGB image: 3 channels, 32x32
    image = torch.zeros(3, 32, 32)

    positions = []

    # Create Red, Green and Blue squares
    for colour in range(3):

        # Keep generating until the square doesn't overlap
        while True:

            x = torch.randint(0, 29, ()).item()
            y = torch.randint(0, 29, ()).item()

            new_position = (x, y)

            good_position = True

            for old_x, old_y in positions:

                if abs(x - old_x) < 5 and abs(y - old_y) < 5:
                    good_position = False

            if good_position:
                break

        positions.append(new_position)

        # Draw 4x4 square
        image[colour, y:y+4, x:x+4] = 1.0

    # Convert square positions to centre positions
    targets = []

    for x, y in positions:

        centre_x = x + 1.5
        centre_y = y + 1.5

        # Normalise 0-31 → 0-1
        target_x = centre_x / 31
        target_y = centre_y / 31

        targets.append([target_x, target_y])

    target = torch.tensor(targets, dtype=torch.float32)

    return image, target 