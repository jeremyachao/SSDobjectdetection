# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio
import numpy as np
import time
from PIL import ImageGrab

print(torch.cuda.is_available())
print(torch.cuda.current_device())
torch.cuda.set_device(0)

# Defining a function that will do the detections
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0)).cuda()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame

# Creating the SSD neural network
net = build_ssd('test').cuda()
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# Doing some Object Detection on a video
# reader = imageio.get_reader('funny_dog.mp4')
# fps = reader.get_meta_data()['fps']
# writer = imageio.get_writer('output.mp4', fps = fps)
# video_capture = cv2.VideoCapture(0)
# for i, frame in enumerate(video_capture.read()):
#     frame = detect(frame, net.eval(), transform)
#     writer.append_data(frame)
#     print(i)
# writer.close()
#webcam
# Doing some Face Recognition with the webcam
# video_capture = cv2.VideoCapture(0)
last_time = time.time()
while True:
    # _, frame = video_capture.read()
    printscreen_pil =  ImageGrab.grab(bbox=(0,40,800,640))
    canvas = detect(np.array(printscreen_pil), net.eval(), transform)
    cv2.imshow('Video', np.array(canvas).astype('uint8'))
    print('took {}'.format(time.time()-last_time))
    last_time = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# video_capture.release()
cv2.destroyAllWindows()
