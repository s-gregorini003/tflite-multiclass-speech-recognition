import sys
import socket


# hostname	= "192.168.1.11"
# port		= 55443

def netcat(host, port, msg):
	
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.connect((host, port))
	
	sock.sendall(msg)
	
	res = ""
	
#	while True:
#		data = sock.recv(1024)
#		if (not data):
#			break
#		res += data.decode()
		
#	print(res)
	
	print("Connection closed.")
	sock.close()
	
# msg = "{\"id\":1,\"method\":\"set_power\",\"params\":[\"on\",\"smooth\",500]}\r\n"

# netcat(hostname, port, msg.encode())
