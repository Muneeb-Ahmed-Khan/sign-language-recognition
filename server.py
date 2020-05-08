
from flask import Flask, request, Response
import numpy as np
import cv2
import json
import sys
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]
with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

global_session = tf.Session()
softmax_tensor = global_session.graph.get_tensor_by_name('final_result:0')


# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/api/recieveData', methods=['POST'])
def recieveData():
    
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.rotate(img, cv2.ROTATE_180)
    #img = cv2.flip(img, 1)

    #Croping the center 200px square
    height = img.shape[0]
    width = img.shape[1]
    x1, y1, x2, y2 = int((width / 2)  - 100), int((height / 2) - 100), int((width / 2)  + 100), int((height / 2) + 100)
    img_cropped = img[y1:y2, x1:x2]

    # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
    # cv2.imshow("IMAGE1", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
    
    predictions = global_session.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        #print('%s (score = %.5f)' % (human_string, score))
        if score > max_score:
            max_score = score
            res = human_string
        
    return Response(response= res, status=200, mimetype="application/json")

# start flask app
app.run(host="192.168.1.108", port=5000)