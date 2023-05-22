from flask import Flask, request, redirect, json
from PIL import Image
import io
import torch
from models.yolo import Model
from utils.torch_utils import select_device
from hubconf import custom
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import easyocr
import subprocess
from subprocess import Popen
import os
import glob
import shutil
import collections.abc

app = Flask(__name__)

reader = easyocr.Reader(['en']) 

#get targer images
def plot_bounding_box(image_file, image, annotation_list, i, save_file_directory, filename):
   
    annotations = np.array(annotation_list) 
    w, h = image.size 
    directory = image_file[0:len(image_file)-3]
    if not os.path.exists(os.path.join(save_file_directory, filename)):
        os.makedirs(os.path.join(save_file_directory, filename))
    transformed_annotations = np.copy(annotations) 
    transformed_annotations[:,[0,2]] = annotations[:,[0,2]] * w 
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * h  
    
    transformed_annotations[:,0] = transformed_annotations[:,0] - (transformed_annotations[:,2] / 2) - 3
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2) - 3
    transformed_annotations[:,2] = transformed_annotations[:,0] + transformed_annotations[:,2] + 3
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3] + 3
    
    for ann in transformed_annotations: 
        x0, y0, x1, y1 = ann 
        cropped_image = image.crop((x0,y0, x1, y1))
        filepath = os.path.join(save_file_directory, filename)
        cropped_image.save(os.path.join(filepath,  str(i) + '.jpg'))
        
def resavedImageToTargetsImages(detect_path, k_global):
    k_temp = k_global
    exp_detect_path = os.path.join(detect_path, 'exp')
    labels_path = os.path.join(exp_detect_path, 'labels')
    labels = glob.glob(os.path.join(labels_path, '*.txt')) 
    for label in labels:
        with open(label, 'r') as f:
            result = list()
            for line in f: 
                annotation_list = line.split("\n")[:-1] 
                annotation_list = [x.split(" ") for x in annotation_list] 
                annotation_list = [[float(y) for y in x if y != ''] for x in annotation_list]
                if len(annotation_list[0]) != 0:
                    result.append(annotation_list)
        image_file = label.replace("labels\\", "")
        image_file = image_file[0:len(image_file)-3] 
        image_file += "jpg" 
        assert os.path.exists(image_file) 
        
        #Load the image 
        image = Image.open(image_file) 
        
        #Plot the Bounding Box 
        i = 1
        for bbox in result:
            plot_bounding_box(image_file, image, [bbox[::][0][1::]], i, detect_path, "temp" + str(k_temp))
            i += 1
        k_temp += 1
    try:
        shutil.rmtree(exp_detect_path)
    except:
        print("An exception occurred")
    return k_temp

k_global = 0
def get_prediction(img_bytes):
    filename = img_bytes.filename
    img = Image.open(img_bytes)
    #basepath = "C:\Curspaper" 
    basepath = os.path.dirname(__file__)
    filepath = os.path.join(basepath, 'uploads', filename)
    print("upload folder is ", filepath)
    img.save(filepath)
    process = Popen(["python", "detect.py", '--save-txt', '--source', filepath, "--weights", "yolov7.pt"], shell=True)
    process.wait()
    
    folder_path = 'runs/detect'
    k_temp = resavedImageToTargetsImages(folder_path, k_global)
    
    result = list()
    for i in range(k_global, k_temp):
        files_to_detect_dir = os.path.join(folder_path, "temp" + str(i))
        files_to_detect = glob.glob(os.path.join(files_to_detect_dir, '*.jpg'))
        local_result = list()
        for file in files_to_detect:
            img = cv2.imread(file)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            msk = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([179, 255, 255]))
            krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            dlt = cv2.dilate(msk, krn, iterations=1)
            thr = 255 - cv2.bitwise_and(dlt, msk)
            plt.imshow(thr)
            text = pytesseract.image_to_string(thr, lang='eng', config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789")
            mass_text = reader.readtext(thr, allowlist='0123456789')
            local_result.append(text)
            local_result.append(mass_text)
        result.append(local_result)
    return result

def get_strongest(files_result):
    for file_result in files_result:
        exists_result_strong = [x for x in file_result if x != '' and len(x) != 0]                
        max_strong = [x for x in exists_result_strong if isinstance(x, (list, tuple))]
        if len(max_strong) != 0:
            max_strong = max(x[-1][-1] for x in max_strong)
        elif len(exists_result_strong) != 0:
            return exists_result_strong[0]
        for x in exists_result_strong:
            if isinstance(x, (list, tuple)): 
                if (x[-1][-1] == max_strong):
                    return x

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        results = get_prediction(f)
        print(results)
        try:
            result = get_strongest(results)
            if isinstance(result, (list, tuple)):
                points, number, strong = result[0]
            else:
                number = result
            response = app.response_class(
                response = json.dumps(number),
                status=200,
                mimetype='application/json'
            )
            return response
        except:
            response = app.response_class(
                response=json.dumps('smth went wrong'),
                status=201,
                mimetype='application/json'
            )
            return response
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6666)