# -*- coding: utf-8 -*-#
#Import flask class library, render_template template,
from flask import Flask, render_template, request, jsonify, make_response
 #Safe file name
from werkzeug.utils import secure_filename
import os
import cv2
import time
import json
from PIL import Image
from io import BytesIO
import json
import numpy as np
from datetime import timedelta
import yolov4

set_upload_path = 'images'
set_result_path = 'out'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)
 #URLAddress
@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "File type exception !"})
                 #t is the name of the obtained picture
        t = f.filename
                 #Division, take the file name without .jpg
        filename_ = t.split('.')[0]
        user_input = request.form.get("name")
                 #Project Main Directory
        basepath = os.path.dirname(__file__)
                 # File upload directory address
        upload_path = os.path.join(basepath, set_upload_path, secure_filename(f.filename))
        f.save(upload_path)
        lab, img, loc, res = yolov4.yolo_detect(pathIn=upload_path)
        
                 #The directory where the test results are written
        
        cv2.imwrite(os.path.join(basepath, set_result_path, filename_+'_res.jpg'), img)
        return res
    return render_template('upload.html')
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")