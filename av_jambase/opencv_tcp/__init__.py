import logging
import socket
import struct

import cv2
import numpy as np

from .communicate import (
    receive_frame,
    receive_json,
    send_empty_frame,
    send_frame,
    send_json,
)

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class SocketConnection:

    def __init__(self, conn: socket.socket):
        self.conn = conn

    def send_frame(self, frame: np.ndarray | None, flags: int = 0) -> None:
        """
        Send a frame over the socket.
        """
        if frame is None:
            # If the frame is None, send an empty frame
            # logger.debug("Sending empty frame...")
            send_empty_frame(self.conn, flags=flags)
            # logger.debug("Sending empty frame... done.")
            return
        elif isinstance(frame, int):
            # If the frame is an integer, send it as a single byte
            # logger.debug("Sending integer frame...")
            frame = np.array([[[frame]*3]], dtype=np.uint8)
            # logger.debug("Sending integer frame... done.")

        # logger.debug(f"Sending frame (shape: {frame.shape})...")
        send_frame(self.conn, frame, flags=flags)
        # logger.debug("Sending frame... done.")

    def receive_frame(self, flags: int = 0) -> np.ndarray:
        """
        Receive a frame from the socket.
        """
        # logger.debug("Receiving frame...")
        frame = receive_frame(self.conn, flags=flags)
        if frame is None:
            # logger.debug("Received empty frame.")
            return None
        # logger.debug(f"Received frame (shape: {frame.shape}).")
        return frame

    def send_json(self, data: dict, flags: int = 0) -> None:
        """
        Send JSON data over the socket.
        """
        # logger.debug("Sending JSON data...")
        send_json(self.conn, data, flags=flags)
        # logger.debug("Sending JSON data... done.")

    def receive_json(self, flags: int = 0) -> dict:
        """
        Receive JSON data from the socket.
        """
        # logger.debug("Receiving JSON data...")
        data = receive_json(self.conn, flags=flags)
        # logger.debug("Receiving JSON data... done.")
        return data

    def close(self):
        """
        Close the socket connection.
        """
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is not None:
            logger.error(f"Exception occurred: {exc_val}")
        return False


class SocketClient:
    """
    A simple socket client for sending and receiving frames.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        return SocketConnection(self.sock)


class SocketServer:
    """
    A simple socket server for receiving and sending frames.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)

    def accept_connection(self) -> SocketConnection:
        """
        Accept a connection from a client.
        """
        # logger.debug("Waiting for a connection...")
        conn, addr = self.sock.accept()
        logger.info(f"Connection from {addr}")
        return SocketConnection(conn)

    def close(self):
        """
        Close the socket server.
        """
        self.sock.close()
        # logger.debug("Socket server closed.")


if __name__ == "don't run this":
    client = SocketClient("192.168.0.33", 12345)
    cap = cv2.VideoCapture(0)
    with client.connect() as conn:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            conn.send_frame(frame)
            received_frame = conn.receive_frame()

            if received_frame is not None:
                cv2.imshow("Received Frame", received_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG  # , format="%(asctime)s - %(levelname)s - %(message)s"
    )

    server = SocketServer("0.0.0.0", 12345)
    with server.accept_connection() as conn:
        while True:
            frame = server.receive_frame()
            if frame is not None:
                cv2.imshow("Received Frame", frame)
                conn.send_frame(None)
                # server.send_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
