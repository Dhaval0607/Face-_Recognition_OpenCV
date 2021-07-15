import cv2
face_cascade = cv2.CascadeClassifier("C:\\Users\\Rathod Dhaval\\PycharmProjects\\Facerecognition\\haarcascade_frontalface_default.xml")

img = cv2.imread("C:\\Users\\Rathod Dhaval\\PycharmProjects\\Facerecognition\\dhaval.jpg",1)

gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img,scaleFactor = 1.05,minNeighbors =5)

for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

resized = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
cv2.imshow("Legend",resized)


cv2.waitKey()
