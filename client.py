from __future__ import print_function
import requests
import json
import cv2

addr = 'http://159.65.157.105:5000'
test_url = addr + '/api/recieveData'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('image.jpg')

# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
print(type(img_encoded.tostring()))
print(len(img_encoded.tostring()))
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring())

# decode response
print(response.text)