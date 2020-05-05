import sys
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def predict(image_data):

    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string
    return res, max_score

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        
        if ret:
            height = img.shape[0]
            width = img.shape[1]

            x1, y1, x2, y2 = int((width / 2)  - 100), int((height / 2) - 100), int((width / 2)  + 100), int((height / 2) + 100)
            img_cropped = img[y1:y2, x1:x2]
            
            image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
            res_tmp, score = predict(image_data)

            cv2.putText(img, '%s' % (res_tmp.upper()), (100,500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
            cv2.putText(img, '(score = %.5f)' % (float(score)), (100,550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.imshow("img", img)

            a = cv2.waitKey(1) # waits to see if `esc` is pressed
            if a == 27: # when `esc` is pressed
                break

cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()
