import cv2
import tensorflow as tf
import numpy as np
import sys

def main():
    # Load the cascade
    front_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')

    # To capture video from webcam.
    if len(sys.argv) == 1:
        video = cv2.VideoCapture(0)

    #If command line argument is given thne read the file
    else:
        video = cv2.VideoCapture(sys.argv[1])

    #Check if video is properly opened
    if video.isOpened() == False:
        print("Error reading video from camera or file does not exist")
        return

    # To use a video file as input
    model =  load_model()
    while True :
        # Read the frame
        ret, img = video.read()
        # If frame is not returned then break the loop
        if not ret:
            print("Error reading frame")
            break

        #Detect faces
        faces = front_cascade.detectMultiScale(img, 1.1, 4)

        # Make inference and draw the rectangle around each face
        for x,y,w,h in faces:
            roi = img[y:y+h, x:x+w]
            prediction = predict(model, roi)
            if prediction == 1:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
                cv2.putText(img, "NO mask Detected", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            elif prediction == 2:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(img, "mask Detected", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Display
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
        

def load_model():
    # Load the model
    model = tf.keras.models.load_model('model.h5')
    return model

def predict(model, img):
    # Preprocess the image
    img = cv2.resize(img, (244, 244))
    img = img.reshape(1, 244, 244, 3)
    img = img / 255.0
    # Make prediction
    pred = model.predict(img)
    return np.argmax(pred)

if __name__ == "__main__":
    main()