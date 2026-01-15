"""Test video stream from Pi"""
import socket
import struct
import time

ROBOT_IP = "192.168.0.105"

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(10)
sock.connect((ROBOT_IP, 8000))
print('Connected to video port')

connection = sock.makefile('rb')
print('Waiting for video data...')

for i in range(3):
    header = connection.read(4)
    if len(header) == 4:
        length = struct.unpack('<L', header)[0]
        print(f'Frame {i+1}: length={length}')
        data = connection.read(length)
        print(f'  Received {len(data)} bytes')
    else:
        print(f'No header received (got {len(header)} bytes)')
    time.sleep(0.1)

sock.close()
print('Done')
