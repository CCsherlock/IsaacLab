import socket
import threading
import struct


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

    def _calculate_checksum(self, data: bytes) -> int:
        """计算数据包的校验和"""
        return sum(data) % 256  # 校验和是所有字节之和的低8位

    def _create_packet(self, data: bytes) -> bytes:
        """构建数据包，包括帧头、数据内容和校验和"""
        frame_header = b"\xa5"  # 帧头
        checksum = self._calculate_checksum(data)  # 计算校验和
        frame_footer = struct.pack("B", checksum)  # 帧尾：加和校验（1字节）
        # 构建完整的数据包
        packet = frame_header + data + frame_footer
        return packet

    def _receive_message(self):
        """接收数据并进行校验"""
        try:
            while True:
                # 逐个字节接收数据
                data = self.sock.recv(1024)
                if data:
                    # 检查数据包的帧头是否是 0xA5
                    if data[0] == 0xA5:
                        # 提取数据内容和校验和
                        message_data = data[1:-1]  # 提取数据部分（去掉帧头和校验位）
                        received_checksum = data[-1]  # 最后的字节是校验和

                        # 计算接收到的数据的校验和
                        calculated_checksum = self._calculate_checksum(message_data)

                        # 校验和匹配
                        if received_checksum == calculated_checksum:
                            print(f"接收到服务器的消息: {message_data.decode()}")
                        else:
                            print("校验失败: 数据损坏")
                    else:
                        print("无效数据包: 帧头错误")
                else:
                    print("服务器断开连接")
                    break
        except Exception as e:
            print(f"接收消息时发生错误: {e}")
        finally:
            self.sock.close()

    def send_message(self, message: str):
        """发送消息到服务器"""
        try:
            message_data = message.encode()
            packet = self._create_packet(message_data)
            self.sock.sendall(packet)  # 发送数据包
            print(f"已发送消息: {message}")
        except Exception as e:
            print(f"发送消息时发生错误: {e}")

    def close(self):
        self.sock.close()
        print("连接已关闭")
