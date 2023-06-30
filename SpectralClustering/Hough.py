import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('images/train_11.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Perform probabilistic Hough transform for line detection
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=80, maxLineGap=10)

# Filter and connect horizontal lines on the same level
if lines is not None:
    lines = sorted(lines, key=lambda line: line[0][1])  # Sort lines by y-coordinate
    connected_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = abs((y2 - y1) / (x2 - x1 + 1e-5))  # Calculate the slope
        if slope < 0.1:  # Adjust this threshold to filter horizontal lines
            if connected_lines:
                last_line = connected_lines[-1]
                if abs(y1 - last_line[1]) < 10:
                    # Connect with the previous line on the same level
                    last_line[2] = x2
                else:
                    # Check if there are close lines above or below
                    for prev_line in connected_lines[::-1]:
                        if abs(y1 - prev_line[1]) < 10:
                            # Connect with the close line above
                            prev_line[2] = x2
                            break
                        elif abs(y1 - prev_line[3]) < 10:
                            # Connect with the close line below
                            prev_line[2] = x2
                            prev_line[3] = y2
                            break
                    else:
                        # Add a new line to the list
                        connected_lines.append([x1, y1, x2, y2])
            else:
                # Add the first line to the list
                connected_lines.append([x1, y1, x2, y2])

    # Draw the connected lines
    thickness = 25  # Adjust this value to increase the line thickness
    for line in connected_lines:
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)

# Display the image with detected lines using matplotlib
plt.imshow(edges)
plt.axis('off')
plt.show()

#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.imshow(image)
plt.axis('off')
plt.show()
