import threading
import time
import socket


# Creating TCP connection
def connect(port):
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.connect(('127.0.0.1',port))
        print('conexao TCP estabelecida')
        return s
    except:
        print('falhou em fazer o a conexao TCP como cliente')
        return 0
        #self.terminate()
    else:
        print('continuando')

# Gives the status and the reward the agent received
def get_state_reward(s , act):
    s.send(str(act).encode())
    data = "" 
    data_recv = False
    while(not data_recv):
        data = s.recv(1024).decode()
        try:
            data = eval(data)
            data_recv = True
        except:
            data_recv = False

    # Converting the data to decimal int
    estado = data['estado']
    recompensa = data['recompensa']

    return estado, recompensa
