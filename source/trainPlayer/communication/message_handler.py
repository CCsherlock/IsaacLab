import socket
import threading


class MessageHandler:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"已连接到服务器: {self.host}:{self.port}")

        # 启动接收消息的线程
        self.receiver_thread = threading.Thread(target=self._receive_message)
        self.receiver_thread.daemon = True  # 守护线程，主程序退出时会自动关闭
        self.receiver_thread.start()

    def _receive_message(self):
        try:
            while True:
                data = self.sock.recv(1024)  # 接收数据
                if data:
                    print(f"接收到服务器的消息: {data.decode()}")
                else:
                    print("服务器断开连接")
                    break
        except Exception as e:
            print(f"接收消息时发生错误: {e}")
        finally:
            self.sock.close()

    def send_message(self, message: str):
        try:
            # 发送消息到服务器
            self.sock.sendall(message.encode())
            print(f"已发送消息: {message}")
        except Exception as e:
            print(f"发送消息时发生错误: {e}")

    def close(self):
        self.sock.close()
        print("连接已关闭")
