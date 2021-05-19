from flask import Flask, render_template, request, jsonify, make_response, jsonify
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
import yaml
import shutil
from azure.storage.blob import ContainerClient, BlobClient, BlobLeaseClient, BlobServiceClient

#Delete befor deployment  !!!!!!
from flask_cors import CORS #comment this on deployment

#Path to input data
set_upload_path = 'input_images'
set_video_path = 'input_videos'

#Path to output images with Bounding-Box
set_result_path = 'output_images'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp','jpeg'])
VIDEO_EXTENSIONS = set(['mp4', 'avi'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
def allowed_vfile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in VIDEO_EXTENSIONS

def load_config():
    dir_root = os.path.dirname(os.path.abspath(__file__))
    with open(dir_root + "/config.yaml", "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.Loader )
    
app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)

#Delete befor deployment!!!!!!
CORS(app) #comment this on deployment


# #%% app route - Upload singal image for species detection component
# @app.route('/', methods=['POST'])
# def upload():
#     f = request.files['image']
#     if not (f and allowed_file(f.filename)):
#         return jsonify({"error": 1001, "msg": "File type exception !"})
#     #t is the name of the obtained picture
#     t = f.filename
#     #Division, take the file name without .jpg
#     filename_ = t.split('.')[0]
#     user_input = request.form.get("name")
#     #Project Main Directory
#     basepath = os.path.dirname(__file__)
    
#     #Create the uplode path if not exist
#     try:
#         if not os.path.exists(set_upload_path):
#             os.makedirs('input_images')
#     except OSError:
#         print ('Error: Creating directory failed')                    
#     #File upload directory address
#     upload_path = os.path.join(basepath, set_upload_path, secure_filename(f.filename))
#     f.save(upload_path)
#     lab, img, loc, res = yolov4.yolo_detect(pathIn=upload_path)

#     #Create the result path if not exist
#     try:
#         if not os.path.exists(set_result_path):
#             os.makedirs('output_images')
#     except OSError:
#         print ('Error: Creating directory failed')
#     #Save the output image
#     cv2.imwrite(os.path.join(basepath, set_result_path, filename_+'_res.jpg'), img)
#     return res

#%% app route - Uploading multiple images
@app.route('/uploadimages', methods=['POST'])
def uploadMult():
    files = request.files.getlist('image')
    print("#########################>>>>")
    print(request.files)
    data = []
    for f in files:
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "File type exception !"})
        #t is the name of the obtained picture
        t = f.filename
        #Division, take the file name without .jpg
        filename_ = t.split('.')[0]
        user_input = request.form.get("name")
        #Project Main Directory
        basepath = os.path.dirname(__file__)
        #File upload directory address
        upload_path = os.path.join(basepath, set_upload_path, secure_filename(f.filename))
        f.save(upload_path)
        lab, img, loc, res = yolov4.yolo_detect(pathIn=upload_path)
        
        if res['counts'] > 0:
            #Save the output image
            cv2.imwrite(os.path.join(basepath, set_result_path, filename_+'_res.jpg'), img)
            #append the json files of all images
            data.append(res)

    final_result = jsonify(data)
    return final_result


#%% app route - Copy blob image
@app.route('/copyBlobImage', methods=['POST'])
def blob_copy():
    c = load_config()
    json_info = request.get_json(force=True)
    url =  json_info['url']
    fileName = json_info['fileName']
    directoryPath = json_info['individual_ID']

    destination_blob_name = directoryPath + '/' + fileName

    client = BlobServiceClient.from_connection_string(c["azure_storage_connectionstring"])
    new_blob = client.get_blob_client(c["ident_container_name"],destination_blob_name)
    res = new_blob.start_copy_from_url(url)
            
    final_result = jsonify(res)
    return final_result

    
#%% app route - Upload video for species detection component
@app.route("/uploadVideo", methods=["POST"])
def uploadVideo():
    id = request.form["id"]
    #id = 12323232898989
    f = request.files['file']
    if not (f and allowed_vfile(f.filename)):
        return jsonify({"error": 1001, "msg": "File type exception !"})
    #t is the name of the obtained picture
    t = f.filename
    #Division, take the file name without .jpg
    filename_ = t.split('.')[0]
    user_input = request.form.get("name")
    #Project Main Directory
    basepath = os.path.dirname(__file__)
    #File upload directory address
    upload_path = os.path.join(basepath, set_video_path, secure_filename(f.filename))
    f.save(upload_path)
    
    # Playing video from file:
    #cap = cv2.VideoCapture('./input_videos/E497.mp4'  )
    cap = cv2.VideoCapture('./input_videos/' + t )
    
    def load_config():
        dir_root = os.path.dirname(os.path.abspath(__file__))
        with open(dir_root + "/config.yaml", "r") as yamlfile:
            return yaml.load(yamlfile, Loader=yaml.Loader )
    
    def get_file(dir):
        print('#################################')
        print(dir)
        for entry in os.scandir(dir):
                print(entry)
                if entry.is_file() and not entry.name.startswith('.'):
                    yield entry
                    
    def upload(files, connection_string, container_name):
        container_client = ContainerClient.from_connection_string(connection_string, container_name)
        print("Uploading files to blob storage...")
        
        for file in files:
            blob_client = container_client.get_blob_client(file.name)
            with open(file.path, "rb") as data:
                blob_client.upload_blob(data)
#                 print('###')
#                 print(data)
#                 blob_list = container_client.list_blobs()
#                 print("\n----------List of blobs in the container ----------")
#                 for file in blob_list:
#                     print("----> " + file.name)
                    
    config = load_config()
    video = get_file("./input_videos")
    print(video)
    upload_res = upload(video, config["azure_storage_connectionstring"], config["videos_container_name"])
#     print("######Uploaded video res##### ")
#     print(jsonify(upload_res))
    
    try:
        if not os.path.exists('videoData'):
            os.makedirs('videoData')
    except OSError:
        print ('Error: Creating directory of data')
    
    currentFrame = 0
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i%58==0:
            name = './videoData/{}'.format(filename_) + "_" + str(currentFrame) + '.jpg'
            print ('Creating...' + name)
            cv2.imwrite(name, frame)
            currentFrame += 1 
        i+=1
        
    final_res = 'Uploaded total:{}'.format(currentFrame)
    data = []
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    res=0;    
    
    input_path = './videoData/'
    for filename in os.listdir(input_path):
        if filename.endswith('.jpg'):
            name = './videoData/' + filename
            print(name)
            lab, img, loc, res = yolov4.yolo_detect(pathIn=name)
            print(res["counts"])
            if res["counts"]==0:
                print(filename)
                os.remove(input_path + filename)
            else: data.append(res)
    
    picture = get_file("./videoData")
    upload(picture, config["azure_storage_connectionstring"], "encountersraw/{}".format(id))
    
#     container_client = ContainerClient.from_connection_string(connection_string, container_name)
#     blob_list = container_client.list_blobs()
#     print("\n----------List of blobs in the container ----------")
#     for blob in blob_list:
#         print("----> " + blob.name)
        
    shutil.rmtree(input_path) 
    os.remove('./input_videos/' + t)   #Test required
    data.append(upload_res)
    print(data)
    final_result = jsonify(data)
    return final_result      

# @app.route('/cutVideo', methods=['POST'])
# def cutVideo():
#     # Playing video from file:
#     #cap = cv2.VideoCapture('E497.mp4')
#     f = request.files['video']
#     print (request)
#     cap = cv2.VideoCapture(f)
#     try:
#         if not os.path.exists('videoData'):
#             os.makedirs('videoData')
#     except OSError:
#         print ('Error: Creating directory of data')

#     currentFrame = 0
#     while(True):
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         # Saves image of the current frame in jpg file
#         name = './videoData/frame' + str(currentFrame) + '.jpg'
#         print ('Creating...' + name)

#         # To stop duplicate images
#         currentFrame += 1

#     # When everything done, release the capture
#     cap.release()
#     cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")