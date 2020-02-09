import cv2, time
import numpy as np

#load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
font  = cv2.FONT_HERSHEY_PLAIN
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0,255,size=(len(classes),3))

setFPS = 250
accuracy = 0.55
cap = cv2.VideoCapture(0)
timeStart = time.time()
frameID = 0
while True:
    _, frame = cap.read()
    frameID = frameID+1
    #frame = cv2.resize(frame, ( int(frame.shape[1]/6),int(frame.shape[0]/6) ) )
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.005,(setFPS, setFPS), (0, 0, 0), True, crop=False)

    # for b in blob:
    #     for n, frame_blob in enumerate(b):


    net.setInput(blob)
    outs = net.forward(outputlayers)

    confidences = []
    classids = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            if confidence > accuracy:
                centerX = int(detection[0]*width)
                centerY = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #cv2.circle(frame, (centerX,centerY), 10, (0,255,0), 2)
                #Rect
                x = int(centerX - w/2)
                y = int(centerY - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                classids.append(classid)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    #print(indexes)

    numberObjectsDetected = len(boxes)
    for i in range(numberObjectsDetected):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[classids[i]])
            con = str(float("{0:0.2f}".format(confidences[i])))
            color = colors[i]
            cv2.rectangle(frame, (x,y), (x+h,y+h),color,2)
            cv2.putText(frame, label, (x,y+30), font, 2, color, 2)
            cv2.putText(frame, con, (x,y+60), font, 2, color, 2)

    elapsedTime = time.time() - timeStart
    fps = frameID/elapsedTime
    fps = str(float("{0:0.2f}".format(fps)))

    cv2.putText(frame, "FPS = "+ fps, (10,30),font, 3, (0,0,0), 1)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(100)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
