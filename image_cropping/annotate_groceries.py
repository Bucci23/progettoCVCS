import cv2
import numpy as np
import os
import json


def single_object_detect(image_names, output_dir, thl=75, thh=150):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Iterate over all the files in the in directory
    predictions = {}
    for filename in image_names:
        # Check if the file is a JPEG image
        if (filename.endswith('.jpg') or filename.endswith('.jpeg')) or filename.endswith('.JPG'):
            if not os.path.isfile(filename):
                continue
            print(f"elaborating{filename}")
            # Load the image
            img = cv2.imread(filename)
            # Apply gaussian blur
            blur = cv2.GaussianBlur(img, (21, 21), 0)
            # Extract gray-level image
            # Split the channels
            b, g, r = cv2.split(blur)
            # Find the edges in each channel

            # Multiresolution version. To find fatter and stronger edges. To use the classical one comment from here...
            pyramidB = [b]
            pyramidG = [g]
            pyramidR = [r]
            for i in range(4):
                pyramidB.append(cv2.pyrDown(pyramidB[-1]))
                pyramidG.append(cv2.pyrDown(pyramidG[-1]))
                pyramidR.append(cv2.pyrDown(pyramidR[-1]))
            edgesB = np.zeros(b.shape, dtype=np.uint8)
            edgesG = np.zeros(g.shape, dtype=np.uint8)
            edgesR = np.zeros(r.shape, dtype=np.uint8)
            # Apply edge detection to each level of the pyramid
            for i in range(len(pyramidB)):
                edges = cv2.Canny(pyramidB[i], thl, thh)
                if edgesB.shape != edges.shape:
                    edges = cv2.resize(edges, (edgesB.shape[1], edgesB.shape[0]))
                edgesB += edges
                edges = cv2.Canny(pyramidG[i], thl, thh)
                if edgesG.shape != edges.shape:
                    edges = cv2.resize(edges, (edgesG.shape[1], edgesG.shape[0]))
                edgesG += edges
                edges = cv2.Canny(pyramidR[i], thl, thh)
                if edgesR.shape != edges.shape:
                    edges = cv2.resize(edges, (edgesR.shape[1], edgesR.shape[0]))
                edgesR += edges
            # Merge the edges from each channel
            grayEdges = cv2.cvtColor(cv2.merge((edgesB, edgesG, edgesR)), cv2.COLOR_BGR2GRAY)
            # Thresholding to get a binary image
            _, edges = cv2.threshold(grayEdges, 1, 255, cv2.THRESH_BINARY)
            # Erosion and dilation to connect non-closed edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(edges, kernel, iterations=5)
            eroded = cv2.erode(dilated, kernel, iterations=5)
            # Detect the contours from the closed edges
            contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            maxArea = 0
            best = None
            # Iterate over all the contours found to get the biggest one
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

                predictions[filename] = [x, y, x + w, y + h]
            # save both the output image with the bb and the edge map
            cv2.imwrite(os.path.join(output_dir, os.path.basename(filename)), img)
    with open('grocery_output.json', 'w') as f:
        json.dump(predictions, f)
    return predictions
