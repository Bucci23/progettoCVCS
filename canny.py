import cv2
import numpy as np
import os
# Read the image
input_dir = 'robaccia/in'
output_dir = 'robaccia/out'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over all the files in the input directory
for filename in os.listdir(input_dir):
    # Check if the file is a JPEG image
    if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        # Load the image
        img = cv2.imread(os.path.join(input_dir, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply Otsu's thresholding
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Apply Canny edge detection
        edges = cv2.Canny(img, 70, 150)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Iterate over all the contours found
        for contour in contours:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            # Ignore small contours
            if area < 500:
                continue
            # Draw a bounding box around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, filename), img)
# Display the original image and the edge-detected image
#cv2.imshow('Original', blur)
#cv2.imshow('Edges', edges)

# Wait for a key press and then exit
#cv2.waitKey(0)
#cv2.destroyAllWindows()
