import cv2
import torch
import glob
import os
from helper import get_color
import xml.etree.cElementTree as tree
from pascal_voc_writer import Writer

model = torch.hub.load('../yolov5','custom', path="best.pt", source='local')
classes = ['helmet','head','person']
root_dir = './Images/'
images = glob.glob(os.path.join(root_dir,'*.png'))

for img in images:
    basename = os.path.basename(img)
    filename = os.path.splitext(basename)[0]
    frame = cv2.imread(img)
    output = model(frame,size=416)
    labels, cords = output.xyxyn[0][:,-1], output.xyxyn[0][:, :-1]
    cords = cords.cpu().detach().numpy()
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    writer = Writer(img,x_shape, y_shape)
    n = labels.shape[0]
    for i in range(n):
        confidence = cords[i][4]
        #print(labels)
        object_class = int(labels[i])
        if confidence > 0.5:
            xmin,ymin,xmax,ymax = int(cords[i][0]*x_shape), int(cords[i][1]*y_shape), int(cords[i][2]*x_shape), int(cords[i][3]*y_shape)
            writer.addObject(str(classes[object_class]),xmin,ymin,xmax,ymax)
    
    file_name = './labels/'+f'{filename}.xml'
    writer.save(file_name)
    print(file_name," annotations generated")
