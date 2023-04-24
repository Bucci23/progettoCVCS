import cv2
import numpy as np
import os


# Read the image


def single_object_detect(input_dir, output_dir, mask_dir, intermediate_result, grabcut_images, thl=75, thh=150):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    if not os.path.exists(intermediate_result):
        os.makedirs(intermediate_result)
    if not os.path.exists(grabcut_images):
        os.makedirs(grabcut_images)
    # Iterate over all the files in the in directory
    predictions = {}
    for filename in os.listdir(input_dir):
        # Check if the file is a JPEG image
        if (filename.endswith('.jpg') or filename.endswith('.jpeg')) or filename.endswith('.JPG'):
            # Load the image
            img = cv2.imread(os.path.join(input_dir, filename))
            # Apply gaussian blur
            blur = cv2.GaussianBlur(img, (21, 21), 0)
            # Extract gray-level image
            # Split the channels
            b, g, r = cv2.split(blur)
            '''Prova con histogram equalization (va peggio)
            min_val, max_val, _, _ = cv2.minMaxLoc(b)
            outB = np.uint8((b - min_val) * 255.0 / (max_val - min_val))
            min_val, max_val, _, _ = cv2.minMaxLoc(g)
            outG = np.uint8((g - min_val) * 255.0 / (max_val - min_val))
            min_val, max_val, _, _ = cv2.minMaxLoc(r)
            outR = np.uint8((r - min_val) * 255.0 / (max_val - min_val))
            equB = cv2.equalizeHist(outB)
            equG = cv2.equalizeHist(outG)
            equR = cv2.equalizeHist(outR)
            '''
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
            # to here. Uncomment the next 3 lines
            '''
            edgesB = cv2.Canny(b, 10, 50)
            edgesG = cv2.Canny(g, 10, 50)
            edgesR = cv2.Canny(r, 10, 50)
            '''
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
                # apply grabCut on the region we have just found.
                '''
                mask = np.zeros(img.shape[:2], np.uint8)
                mask[y:y + h, x:x + w] = cv2.GC_PR_FGD
                mask[0:100, 0:100] = cv2.GC_PR_BGD
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                iterations = 5
                cv2.grabCut(img, mask, None, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_MASK)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                segmented_img = img * mask2[:, :, np.newaxis]
                '''
                predictions[filename] = [x, y, w, h]
            # save both the output image with the bb and the edge map
            cv2.imwrite(os.path.join(output_dir, filename), img)
            cv2.imwrite(os.path.join(mask_dir, filename), eroded)
            # cv2.imwrite(os.path.join(grabCut_images, filename), segmented_img)
            # cv2.imwrite(os.path.join(intermediate_result, filename), cv2.merge((equB, equG, equR)))
    return predictions
# Display the original image and the edge-detected image
# cv2.imshow('Original', blur)
# cv2.imshow('Edges', edges)

# Wait for a key press and then exit
# cv2.waitKey(0)
# cv2.destroyAllWindows()
