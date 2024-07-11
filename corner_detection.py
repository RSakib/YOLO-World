import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

im = cv.imread('.\\onnx_outputs\\resized d3uam4kdcrop1.png')
assert im is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
edges = cv.Canny(imgray, 0, 175, apertureSize=3)
lines = cv.HoughLines(edges, 2, np.pi / 180, 190)
lines = np.array(lines)
lines = np.squeeze(lines)
for i in range(len(lines)):
    r, theta = lines[i]
    if r < 0:
        print(theta)
        theta = -(np.pi - theta)
        r = -r
        print(theta)
        lines[i] = (r, theta)

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
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.
    cv.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
 
# All the changes made in the input image are finally
# written on a new image houghlines.jpg
cv.imwrite('linesDetected.jpg', im)

plt.scatter(lines[:,0], lines[:,1]*180/np.pi)

plt.show()
""" if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv.imshow("Rectangles Detected", im)
cv.waitKey(0)
cv.destroyAllWindows() """