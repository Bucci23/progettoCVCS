import cv2
import numpy as np
import os

# Read the image
input_dir = 'robaccia/input2/'
output_dir = 'robaccia/out2/'
mask_dir = 'robaccia/masks2/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)
# Iterate over all the files in the input directory
for filename in os.listdir(input_dir):
    # Check if the file is a JPEG image
    if (filename.endswith('.jpg') or filename.endswith('.jpeg')) or filename.endswith('.JPG'):
        # Load the image
        img = cv2.imread(os.path.join(input_dir, filename))
        blur = cv2.GaussianBlur(img, (21, 21), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(blur)
        edgesB = cv2.Canny(b, 25, 100)
        edgesG = cv2.Canny(g, 25, 100)
        edgesR = cv2.Canny(r, 25, 100)
        grayEdges = cv2.cvtColor(cv2.merge((edgesB, edgesG, edgesR)), cv2.COLOR_BGR2GRAY)
        _, edges = cv2.threshold(grayEdges, 1, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxArea = 0
        best = None
        # Iterate over all the contours found
        for contour in contours:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            # find the biggest contour
            if area < 100:
                continue
            if area > maxArea:
                best = contour
                maxArea = area
            # Draw a bounding box around the contour
        if best is not None:
            x, y, w, h = cv2.boundingRect(best)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, filename), img)
            cv2.imwrite(os.path.join(mask_dir, filename), eroded)


# Display the original image and the edge-detected image
# cv2.imshow('Original', blur)
# cv2.imshow('Edges', edges)

# Wait for a key press and then exit
# cv2.waitKey(0)
# cv2.destroyAllWindows()
