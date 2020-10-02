import socket
from config import server

ip_port = ('127.0.0.1', server)
s = socket.socket()
s.connect(ip_port)
print('connected')
s.sendall('/data/shenzhonghai/FaceClustering/models/resnet50_acWebFace.tar'.encode())
s.close()
