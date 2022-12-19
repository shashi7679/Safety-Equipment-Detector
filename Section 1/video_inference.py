from vidgear.gears import CamGear
import cv2
import torch

stream = CamGear(source='https://www.youtube.com/watch?v=6PoPwZ0WO9w', stream_mode = True, logging=True).start() # YouTube Video URL as input
model = torch.hub.load('../yolov5','custom', path="best.pt", source='local')
# infinite loop
classes = ['helmet','head','person']
# out = cv2.VideoWriter('output.avi', -1, 20.0, (1080,1920))
count = 0
while True:
    
    frame = stream.read()
    if frame is None:
        break

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
            cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,0,255),2)
            cv2.putText(frame, str(classes[object_class]), (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
    
    file_name = './output_frames/'+f'{count}.png'
    cv2.imwrite(file_name,frame)
    cv2.imshow("Output Frame", frame)
    # Show output window
    count+=1

    key = cv2.waitKey(1) & 0xFF
    # check for 'q' key-press
    if key == ord("q"):
        #if 'q' key-pressed break out
        break

cv2.destroyAllWindows()
# close output window

# safely close video stream.
stream.stop()