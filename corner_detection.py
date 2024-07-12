import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten, kmeans, vq

# Read in the Image
im = cv.imread('.\\onnx_outputs\\resized d3uam4kdcrop1.png')
assert im is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# Detect HoughLines within the Image
edges = cv.Canny(imgray, 0, 175, apertureSize=3)
lines = cv.HoughLines(edges, 2, np.pi / 180, 190)
lines = np.array(lines)
lines = np.squeeze(lines)

# Iterate over each HoughLine to visualize them on the Image
for i in range(len(lines)):
    r, theta = lines[i]

    # Fix data within our HoughLines
    # If any radius is below 0, then correct it to make sure our data doesn't dip below zero
    if r < 0:
        theta = -(np.pi - theta)
        r = -r
        lines[i] = (r, theta)
    
    ## VISUALIZE THE HOUGHLINES ##

    # Stores the value of cos(theta) in a
    a = np.cos(theta)
 
    # Stores the value of sin(theta) in b
    b = np.sin(theta)
 
    # x0 stores the value rcos(theta)
    x0 = a*r
 
    # y0 stores the value rsin(theta)
    y0 = b*r
 
    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000*(-b))
 
    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000*(a))
 
    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000*(-b))
 
    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000*(a))
 
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    cv.line(im, (x1, y1), (x2, y2), (255, 0, 0), 2)

    ## END OF HOUGHLINE VISUALIZATION ##

    # Convert all thetas to be in degrees instead of Radians
    lines[i] = (r, theta*180/np.pi)

# normalize HoughLines
dataset = whiten(lines)

# generate center lines from the clusters of houghlines
centroids, mean_dist = kmeans(lines, 4)
print("Centeroids :\n", centroids, "\n")

# Store the coefficients for the center lines that follow Standard Form: Ax + By + C = 0
coefficients = [] # [A,B,C]

# Iterate over each Center lines
for centroid in centroids:
    ## VISUALIZE CENTER LINES ##

    r, theta = centroid
    theta = theta*(np.pi/180)
    # Stores the value of cos(theta) in a
    a = np.cos(theta)
 
    # Stores the value of sin(theta) in b
    b = np.sin(theta)
 
    # x0 stores the value rcos(theta)
    x0 = a*r
 
    # y0 stores the value rsin(theta)
    y0 = b*r
 
    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000*(-b))
 
    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000*(a))
 
    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000*(-b))
 
    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000*(a))
 
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    cv.line(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

    ## END OF VISUALIZATION ##

    # Convert lines to be in Standard form: Ax + By + C = 0
    A = (y2 - y1)
    B = (x1 - x2)
    C = y1 * (x2 - x1) - (y2 - y1) * x1

    # Store the coefficients
    coefficients.append((A,B,C))

## CALCULATE RECTANGLE CORNERS ##

# Iterate to find the intersections between all pairs of lines
for i in range(len(coefficients)):
    # Do a Cross Product between [a1,b1,c1] and [a2,b2,c2]
    line1 = coefficients[i-1]
    line2 = coefficients[i]
    intersectionMatrix = np.cross(np.array(line1), np.array(line2))
    divisor = intersectionMatrix[2]

    # Intersection position array as [x, y, 1] 
    intersectionPos = intersectionMatrix/divisor

    # Draw the Intersection point in red
    cv.circle(im, (int(intersectionPos[0]), int(intersectionPos[1])), 3, (0,0,255), -1)

## END OF RECTANGLE CORNER DETECTION ##

# Write all transforms to a file
cv.imwrite('linesDetected.jpg', im)

# Draw the scatter plot for the Houghlines and Centerlines
plt.scatter(lines[:,0], lines[:,1])
plt.scatter(centroids[:,0], centroids[:,1])

#plt.show()