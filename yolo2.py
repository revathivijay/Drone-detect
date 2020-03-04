import cv2
import numpy as np

#load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# print(classes)

# cell_phone = ['cell phone', 'person', 'laptop']

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0,255,size=(len(classes),3))

img = cv2.imread("images/bottle5.jpg") 
img = cv2.resize(img, ( int(img.shape[1]/6),int(img.shape[0]/6) ) )
height, width, channels = img.shape
print("height: ", height)
blob = cv2.dnn.blobFromImage(img, 0.00392,(416, 416), (0, 0, 0), True, crop=False)

# for b in blob:
#     for n, img_blob in enumerate(b):


net.setInput(blob)
outs = net.forward(outputlayers)

confidences = []
classids = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        classid = np.argmax(scores)
        # print("classid", classid)
        confidence = scores[classid]
        if confidence > 0.5:
            centerX = int(detection[0]*width)
            centerY = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            #cv2.circle(img, (centerX,centerY), 10, (0,255,0), 2)
            #Rect
            x = int(centerX - w/2)
            y = int(centerY - h/2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            classids.append(classid)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#print(indexes)

font  = cv2.FONT_HERSHEY_PLAIN

numberObjectsDetected = len(boxes)
for i in range(numberObjectsDetected):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[classids[i]])
        if(label=='bottle'):
            print("Object detected: ", label, " , height: ", h)
            distance = 4*260*height/(3.5*h)
            print("Distance: ", distance/10, " cm")
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+h,y+h),color,2)
        cv2.putText(img, label, (x,y+30), font, 2, color, 2)





cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
