import socket
import threading
import struct


class MessageHandler:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        # 创建服务器套接字
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 绑定套接字到指定的地址和端口
        self.sock.bind((self.host, self.port))
        print(f"服务器启动，正在监听 {self.host}:{self.port}")

        # 监听来自客户端的连接请求，最大排队数为 5
        self.sock.listen(5)

        # 等待客户端连接
        self.client_sock, self.client_addr = self.sock.accept()
        print(f"客户端 {self.client_addr} 已连接")

        # 初始化 stop_event 用于停止接收线程
        self.stop_event = threading.Event()
        print(f"stop_event 已初始化: {self.stop_event}")

        # 启动接收消息的线程
        self.receiver_thread = threading.Thread(target=self._receive_message)
        self.receiver_thread.daemon = True  # 守护线程，主程序退出时会自动关闭
        self.receiver_thread.start()

    def _calculate_checksum(self, data: bytes) -> int:
        """计算数据包的校验和"""
        return sum(data) % 256  # 校验和是所有字节之和的低8位

    def _create_packet(self, data: bytes) -> bytes:
        """构建数据包，包括帧头、数据长度、数据内容和校验和"""
        frame_header = b"\xa5"  # 帧头，固定为0xA5
        data_length = len(data)  # 数据部分的长度

        # 将数据长度转换为2字节，适用于长度较大的数据包
        length_field = struct.pack(
            "H", data_length
        )  # 使用2个字节表示数据长度，'H'代表无符号短整型（2字节）

        checksum = self._calculate_checksum(data)  # 计算校验和
        frame_footer = struct.pack("B", checksum)  # 帧尾，校验和为1字节

        # 构建完整的数据包
        packet = frame_header + length_field + data + frame_footer
        return packet

    def _receive_message(self):
        """接收数据并进行校验"""
        try:
            while not self.stop_event.is_set():  # 确保 stop_event 设定为已停止时退出
                data = self.client_sock.recv(1024)
                if data:
                    # 检查数据包的帧头是否是 0xA5
                    if data[0] == 0xA5:
                        # 提取数据长度字段（2字节）
                        data_length = struct.unpack("H", data[1:3])[0]
                        message_data = data[3 : 3 + data_length]  # 获取数据部分

                        # 提取校验和并进行校验
                        received_checksum = data[3 + data_length]
                        calculated_checksum = self._calculate_checksum(message_data)

                        if received_checksum == calculated_checksum:
                            print(f"接收到客户端的消息: {message_data.decode()}")
                        else:
                            print("校验失败: 数据损坏")
                    else:
                        print("无效数据包: 帧头错误")
                else:
                    print("客户端断开连接")
                    break
        except Exception as e:
            print(f"接收消息时发生错误: {e}")
        finally:
            self.client_sock.close()

    def send_message(self, float_array):
        """发送浮动数据数组到客户端"""
        try:
            # 将浮动数组转为字节数据
            message_data = b"".join(
                struct.pack(">f", f) for f in float_array
            )  # 'f' 是浮动数的格式

            # 创建数据包
            packet = self._create_packet(message_data)

            if self.client_sock.fileno() != -1:  # 检查套接字是否有效
                self.client_sock.sendall(packet)  # 发送数据包
                print(f"数据包长度：{len(packet)}")
                print(f"已发送浮动数据: {float_array}")
                print(
                    "数据包（十六进制格式）: ",
                    " ".join(f"{byte:02x}" for byte in packet),
                )
            else:
                print("套接字已关闭，无法发送消息")
        except Exception as e:
            print(f"发送消息时发生错误: {e}")

    def close(self):
        """关闭服务器套接字和客户端套接字"""
        print("关闭服务器和客户端连接...")

        # 设置 stop_event，通知接收线程停止
        self.stop_event.set()
        self.receiver_thread.join()  # 等待接收线程结束

        # 关闭客户端和服务器套接字
        self.client_sock.close()
        self.sock.close()

        print("连接已关闭")
