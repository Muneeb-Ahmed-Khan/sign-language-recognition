from __future__ import print_function
import requests
import json
import cv2

addr = 'http://192.168.1.108:5000'
test_url = addr + '/api/recieveData'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    
    if ret:
        height = img.shape[0]
        width = img.shape[1]

        x1, y1, x2, y2 = int((width / 2)  - 100), int((height / 2) - 100), int((width / 2)  + 100), int((height / 2) + 100)
        img_cropped = img[y1:y2, x1:x2]

        #cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        
        cv2.imshow("Cropped...", img_cropped)
        cv2.imshow("Saving...", img)

        a = cv2.waitKey(1)
        if a & 0xFF == ord("q"):
            
            #rotated because server will also rotate it so it will become normal.
            img = cv2.rotate(img, cv2.ROTATE_180)
            # encode image as jpeg
            _, img_encoded = cv2.imencode('.jpg', img)
            print(type(img_encoded.tostring()))
            print(len(img_encoded.tostring()))
            # send http request with image and receive response
            response = requests.post(test_url, data=img_encoded.tostring())

            # decode response
            print(response.text)

        if a == 27: # when `esc` is pressed
            break

