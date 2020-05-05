
from flask import Flask, request, Response
import numpy as np
import cv2
import json

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/api/recieveData', methods=['POST'])
def recieveData():
    r = request
    
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    cv2.imshow("img", img)
    cv2.waitKey(0)
    
    return Response(response= json.dumps({'message': 'OK'}), status=200, mimetype="application/json")

# start flask app
app.run(host="0.0.0.0", port=5000)