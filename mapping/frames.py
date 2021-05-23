import cv2

vidcap = cv2.VideoCapture('exp3/pei_5g-mobix_10.0.19.201_554.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
