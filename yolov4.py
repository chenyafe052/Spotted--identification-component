# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import time
import json
 
def yolo_detect(im=None,
                pathIn=None,
                label_path='./cfg/obj.names',
                config_path='./cfg/yolov4-obj.cfg',
                weights_path='./cfg/yolov4-obj_best.weights',
                confidence_thre=0.5,
                nms_thre=0.3):
    labels = open(label_path).read().strip().split("\n")
    nclass = len(labels)
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(nclass, 3), dtype='uint8')
    if pathIn == None:
        img = im
    else:
        img = cv2.imread(pathIn)
    # print(pathIn)
    filename = pathIn.split('/')[-1]
    name = filename.split('.')[0]
    (H, W) = img.shape[:2]
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_thre:
                 # Restore the coordinates of the bounding box to match the original picture, remember that YOLO returns
                 # The center coordinates of the bounding box and the width and height of the bounding box
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                 # Calculate the position of the upper left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)
    lab = []
    loc = []
    resultdata=[]
    data = {}
    data["filename"]=filename
    data["counts"]=len(idxs)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.3f}'.format(labels[classIDs[i]], confidences[i])
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x, y-text_h-baseline), (x + text_w, y), color, -1)
            cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            text_inf = text + '' +'(' + str(x) +',' + str(y) +')' + '' + 'width:' + str(w) + 'height:' + str(h)
            info = {'label':labels[classIDs[i]],"confidences":confidences[i],"x":str(x),"y":str(y),"w":str(w),"h":str(h)}
            resultdata.append([info])
            
            data['data']=resultdata
            # print(filename,labels[classIDs[i]],confidences[i],str(x),str(y),str(w),str(h))
            loc.append([x, y, w, h])
            lab.append(text_inf)
    #res = jsonify(data)
    res = data
    return lab, img, loc, res
 
# if __name__ == '__main__':
#     pathIn = './static/images/test1.jpg'
#     im = cv2.imread('./static/images/test2.jpg')
#     lab, img, loc = yolo_detect(pathIn=pathIn)
#     print(lab)