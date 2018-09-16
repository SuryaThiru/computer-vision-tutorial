import cv2
from darkflow.net.build import TFNet
import numpy as np
import urllib

options = {
    'model': '/home/apurvnit/Desktop/darkflow/cfg/yolo.cfg',
    'load': '/home/apurvnit/Projects/darkflow/bin/yolo.weights',
    'label': '/home/apurvnit/Projects/darkflow/cfg/coco.names',
    'threshold': 0.3,
    'gpu': 1
}

url = "http://10.10.191.132:8080//shot.jpg"
tfnet = TFNet(options)
cam = cv2.VideoCapture("/home/apurvnit/Projects/codespace-backend/ml/test.mp4")
# img = cv2.imread('/home/apurvnit/Projects/codespace-backend/current.jpg')
# print(type(img))
c = 0
while True:
	print(c)
	frame = urllib.request.urlopen(url)
	frame = np.array(bytearray(frame.read()), dtype=np.uint8)
	frame = cv2.imdecode(frame, -1)
	# print(frame)
	frame = cv2.flip(frame, 1)
	c += 1
	if c%20!=0:
		continue

	# ret, frame = cam.read()
	result = tfnet.return_predict(frame)
	# print(result)
	for i in range(len(result)):
		label=result[i]['label']
		tl = (result[i]['topleft']['x'], result[i]['topleft']['y'])
		br = (result[i]['bottomright']['x'], result[i]['bottomright']['y'])
		frame = cv2.rectangle(frame, tl, br, (255, 0, 0), 7)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

	cv2.imshow("objects",frame )
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

cam.release()
cv2.destroyAllWindows()




