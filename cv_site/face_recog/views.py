from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import cv2
from django.shortcuts import render
import os
import pdb

HAARCASCADE_EYE_PATH = "{base_path}/haar_cascades/haarcascade_eye.xml".format(base_path=os.path.abspath(os.path.dirname(__file__)))
HAARCASCADE_FRONTALFACE_PATH = "{base_path}/haar_cascades/haarcascade_frontalface_default.xml".format(base_path=os.path.abspath(os.path.dirname(__file__)))
 
@csrf_exempt
def detect(request):
    data = {"success": False}

    if request.method == "POST":
        if request.FILES.get("image", None) is not None:
            image = _grab_image(stream=request.FILES["image"])
 
        else:
            url = request.POST.get("url", None)
 
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)
 
            image = _grab_image(url=url)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eye_recog = cv2.CascadeClassifier(HAARCASCADE_EYE_PATH)
        face_recog = cv2.CascadeClassifier(HAARCASCADE_FRONTALFACE_PATH)
        eye_rects = eye_recog.detectMultiScale(image, 1.3, 5)
        
        face_rects = face_recog.detectMultiScale(image, 1.3, 5)
        eye_measures = [(int(x), int(y), int(x+w), int(y+h)) for (x,y,w,h) in eye_rects]
        face_measures = [(int(x), int(y), int(x+w), int(y+h)) for (x,y,w,h) in face_rects]

        data.update({"num_faces": len(face_measures), "eye": eye_measures, "faces": face_measures, "success": True})
        print(JsonResponse(data))
 
    return JsonResponse(data)
 
def _grab_image(path=None, stream=None, url=None):

    if path is not None:
        image = cv2.imread(path)
 
    else:	
        if url is not None:
            resp = urllib.request.urlopen(url)
            data = resp.read()
 
        elif stream is not None:
            data = stream.read()
 
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

 
    return image
