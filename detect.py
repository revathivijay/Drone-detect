import cv2
import numpy as np
from imutils import paths
import imutils


def find_marker(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	cv2.imshow("Edges",edged)
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
	return cnts, cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	return (knownWidth * focalLength) / perWidth

KNOWN_DISTANCE = 12.0*25.4
KNOWN_WIDTH = 8.375*25.4

image = cv2.imread("images/distImage.jpg")
contours, marker = find_marker(image)
# print("CONTOURS", contours)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

print("Focal length: ", focalLength)

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# print(classes[:5])
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes), 3))

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence>0.5:
                centre_x = int(detection[0]*width)
                centre_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(centre_x-w/2)
                y = int(centre_y-h/2)
                print("xywh: ", x, " ", y, " ", w, " ", h)
                # print(detection)
                color = colors[0]
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    millimeters = 0
    for i in range(len(boxes)):
        x,y,w,h = boxes[i]
        im = frame[y:y+h,x:x+w]
        if np.shape(im) == ():
            continue
        c, m = find_marker(im)
        millimeters = distance_to_camera(KNOWN_WIDTH, focalLength, m[1][0])
        print("DISTANCE IN mm: ", millimeters)
        box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
        box = np.int0(box)
        cv2.drawContours(im,c , -1, (0, 255, 0), 2)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            # print(label)
            confidence_label = confidences[i]
            new_label = "(" + str(confidence_label) + ")"
            label = label + new_label
            # print(label)
            color = colors[i]
            cv2.rectangle(frame, (x,y), (x*w, y*h), color, 2)
            cv2.putText(frame, label, (x,y), font, 3, color, 3)
            # cv2.putText(frame, "%.2fmm"(millimeters), (50,50), font, 3, color, 3)

    cv2.imshow("image", frame)


    key = cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()
