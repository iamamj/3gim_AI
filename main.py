import base64
import socket
import threading
import cv2
import numpy
import torch as torch
from signalrcore.hub_connection_builder import HubConnectionBuilder

import myai

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best0815.pt')

## 원래카메라↓↓↓↓↓
hub_connection2 = HubConnectionBuilder()\
    .with_url('http://10.10.10.202:3333/CameraHub', options={"verify_ssl": False}) \
    .with_automatic_reconnect({
            "type": "interval",
            "keep_alive_interval": 10,
            "intervals": [1, 3, 5, 6, 7, 87, 3]
        })\
    .build()
hub_connection2.start()


## AI카메라 ↓↓↓↓↓
hub_connection3 = HubConnectionBuilder()\
    .with_url('http://10.10.10.202:3333/CameraHub', options={"verify_ssl": False}) \
    .with_automatic_reconnect({
            "type": "interval",
            "keep_alive_interval": 10,
            "intervals": [1, 3, 5, 6, 7, 87, 3]
        })\
    .build()
hub_connection3.start()






class ServerSocket:

    def __init__(self, ip, port):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.socketOpen()
        self.receiveThread = threading.Thread(target=self.receiveImages)
        self.receiveThread.start()

    def socketClose(self):
        self.sock.close()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is close')

    def socketOpen(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.TCP_IP, self.TCP_PORT))

    def receiveImages(self):

        try:
            while True:
                stringData, client = self.sock.recvfrom(99999)
                data = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8)
                data2 = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8)

                decimg = cv2.imdecode(data, 1)
                decimg2 = cv2.imdecode(data2, 1)

                result, label, x, y = myai.run(model=model, origin_img=decimg)

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                ai_result, imgencode = cv2.imencode('.jpg', result, encode_param)
                normal_result, imgencode2 = cv2.imencode('.jpg', decimg2, encode_param)

                aidata = numpy.array(imgencode)
                NMdata = numpy.array(imgencode2)

                ai64Data = base64.b64encode(aidata)
                NM64Data = base64.b64encode(NMdata)

                hub_connection3.send("AICamera", [ai64Data.decode('utf-8')])
                hub_connection2.send("BeltCamera", [NM64Data.decode('utf-8')])

                cv2.imshow("image", result)
                cv2.waitKey(1)
        except Exception as e:
            print(e)
            self.socketClose()
            cv2.destroyAllWindows()
            self.socketOpen()
            self.receiveThread = threading.Thread(target=self.receiveImages)
            self.receiveThread.start()

def main():

    server = ServerSocket('0.0.0.0', 8080)

if __name__ == "__main__":
    main()