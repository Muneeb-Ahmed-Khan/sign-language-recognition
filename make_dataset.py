import cv2
import argparse
import os

print("\nWelcome to Dataset Maker\nPress \'Crtl + C\' to eixt")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="path to input dataset")
parser.add_argument("-i", "--images", type=int, required=True, help="No. of Images per Label")
args = vars(parser.parse_args())


cap = cv2.VideoCapture(0)

while True:
    
    label = input("\n\nLabel Name : ")
    path = os.path.join(args["dataset"], label)
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    print("Directory '%s' created" %path)

    # Loop so in that time user will adjust his hand
    for i in range(100):
        ret, img = cap.read()
        if ret:
            height = img.shape[0]
            width = img.shape[1]
            x1, y1, x2, y2 = int((width / 2)  - 100), int((height / 2) - 100), int((width / 2)  + 100), int((height / 2) + 100)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.imshow("Adjust your Hand in middle center", img)
            a = cv2.waitKey(1)
            if a == 27: # when `esc` is pressed
                break
    
    
    for i in range(args["images"]):
        ret, img = cap.read()
        if ret:
            height = img.shape[0]
            width = img.shape[1]

            x1, y1, x2, y2 = int((width / 2)  - 100), int((height / 2) - 100), int((width / 2)  + 100), int((height / 2) + 100)
            img_cropped = img[y1:y2, x1:x2]

            imgPath = os.path.join(path, str(i) + ".jpg")
            cv2.imwrite(imgPath, img_cropped)

            cv2.putText(img, str(i) + ".jpg" , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            
            cv2.imshow("Saving...", img)
            
            a = cv2.waitKey(1)
            if a == 27: # when `esc` is pressed
                break

cv2.VideoCapture(0).release()
cv2.destroyAllWindows()