# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2
a = 0
def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	cv2.imshow("Edges",edged)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
	# print(c)
	# compute the bounding box of the of the paper region and return it
	return cnts, cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# READ IMAGE
image = cv2.imread("distImage.jpg")

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return ((knownWidth * focalLength) / perWidth)

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 12.0*25.4
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 8.375*25.4
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length

contours, marker = find_marker(image)
print("CONTOURS", contours)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

print(sorted(paths.list_images("images")))
a=0
# loop over the images
for imagePath in sorted(paths.list_images("images")):
	# load the image, find the marker in the image, then compute the
	# distance to the marker from the camera
	a = a+1
	image = cv2.imread(imagePath)
	contours, marker = find_marker(image)
	millimeters = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
	# draw a bounding box around the image and display it
	box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
	box = np.int0(box)
	cv2.drawContours(image,contours , -1, (0, 255, 0), 2)
	cv2.putText(image, "%.2fmm" % (millimeters),
		(image.shape[1] - 350, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
	cv2.imshow("image", image)
	cv2.waitKey(0)

print(a)
cv2.waitKey()
cv2.destroyAllWindows()
