import cv2
#C:\Users\aadiv.DESKTOP-TEH73VP\OneDrive\Desktop
img = cv2.imread ("hari.jpg")
resize = cv2.resize(img, ( int(img.shape[1]/6),int(img.shape[0]/6) ) )
face_cascade = cv2.CascadeClassifier("C:\\Users\\aadiv.DESKTOP-TEH73VP\\OneDrive\\Documents\\Python\\haarcascade_frontalface.xml")
#face_cascade.load("C:\\Users\\aadiv.DESKTOP-TEH73VP\\AppData\\Local\\Programs\\Python\\Python38-32\\XML\\haarcascade_frontalcatface.xml")
print(face_cascade)
gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.31, minNeighbors=3)
print(type(faces))
print(faces)

for x,y,w,h in faces:
    resize = cv2.rectangle(resize, (x,y), (x+w,y+h), (255,0,0), 3)


cv2.imshow("hari",resize)
cv2.waitKey()
cv2.destroyAllWindows()
