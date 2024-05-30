import cv2
import pytesseract as ocr
import numpy as np

Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

alpha = 1.7
beta = 10

def crop_ocr(crop):
    adjusted = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
    adjusted = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)
    
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.dilate(crop, kernel, iterations=2)
    binary = cv2.erode(crop, kernel, iterations=1)
    
    ret, binary = cv2.threshold(binary, 50, 255, cv2.THRESH_BINARY)
    blur = cv2.medianBlur(binary, 3)
    cv2.imshow('img', blur)
    cv2.waitKey(1)
    text2 = ocr.image_to_string(blur)
    return text2

net = cv2.dnn.readNetFromDarknet('config/yolov4-container.cfg', 'weights/yolov4-container_last.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

cap = cv2.VideoCapture('movie/video_0001.avi')
img = cv2.imread('image_0010.jpg')

while True:
    ret, frame = cap.read()
    #frame_counter += 1
    if ret == False:
        break
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        #print(box)
        crop_img = frame[box[1]:box[1] + box[3], box[0]:box[0]+box[2]]
        ocr_text = crop_ocr(crop_img)
        #print('ocr',ocr_text)
        cv2.rectangle(frame, box, color, 1)
        #label = ocr_text
        
        cv2.putText(frame, ocr_text, (box[0], box[1]-10),
                   cv2.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
    #endingTime = time.time() - starting_time
    #fps = frame_counter/endingTime
    # print(fps)
    #cv2.putText(frame, f'FPS: {fps}', (20, 50),
    #           cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        
'''classes, scores, boxes = model.detect(img, Conf_threshold, NMS_threshold)
for (classid, score, box) in zip(classes, scores, boxes):
    color = COLORS[int(classid) % len(COLORS)]
    #label = "%f" % (score)
    crop_img = img[box[1]:box[1] + box[3], box[0]:box[0]+box[2]]
    #cv2.imshow('img', crop_img)
    ocr_text = crop_ocr(crop_img)
    cv2.rectangle(img, box, color, 1)
    cv2.waitKey(1000)'''
#cap.release()
cv2.destroyAllWindows()