import cv2
import torch
import glob
import os
from helper import get_color

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

    n = labels.shape[0]

    x_shape, y_shape = frame.shape[1], frame.shape[0]
    # print(x_shape,y_shape)

    for i in range(n):
        confidence = cords[i][4]
        #print(labels)
        object_class = int(labels[i])
        if confidence > 0.5:
            xmin,ymin,xmax,ymax = int(cords[i][0]*x_shape), int(cords[i][1]*y_shape), int(cords[i][2]*x_shape), int(cords[i][3]*y_shape)
            #### get majority color
            if object_class==0:
                helmet = frame[ymin:ymax, xmin:xmax]
                color = get_color(helmet)
                color = list(color)
                r, g, b = int(color[0]), int(color[1]), int(color[2])
                color = (r,g,b)
                cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),color=tuple(color),thickness=2)
                cv2.putText(frame, str(classes[object_class]), (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 3)
            else:
                cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,0,255),2)
                cv2.putText(frame, str(classes[object_class]), (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
    
    file_name = './output_frames/'+f'{filename}.png'
    cv2.imwrite(file_name,frame)
    cv2.imshow("Output Frame", frame)
