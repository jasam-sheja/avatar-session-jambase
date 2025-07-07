""" """

import socket
import struct

import numpy as np

from .serialize import (deserialize_frame, deserialize_json, serialize_frame,
                        serialize_json)


def receive_all(sock: socket.socket, flags: int = 0) -> bytes:
    """
    Receive all bytes from the socket until the specified length is reached.
    """
    # First, receive the length of the frame
    raw_length = sock.recv(4, flags)
    if len(raw_length) < 4:
        raise RuntimeError("Socket is closed")
    length = struct.unpack("!I", raw_length)[0]
    if length == 0:
        return b""
    if length < 0:
        raise ValueError("Negative length received")
    if length > 2**31 - 1:
        raise ValueError(f"Length exceeds maximum value, got {length}")

    view = memoryview(bytearray(length))
    bytes_received = 0
    while bytes_received < length:
        bytes_received += sock.recv_into(view[bytes_received:], length - bytes_received, flags)
    return view.tobytes()


def send_all(sock: socket.socket, data: bytes, flags: int = 0) -> None:
    """
    Send all bytes over the socket.
    """
    length = struct.pack("!I", len(data))
    sock.sendall(length)
    if len(data) > 0:
        sock.sendall(data, flags)


def receive_frame(sock: socket.socket, flags: int = 0) -> np.ndarray | None:
    """
    Receive a frame from the socket.
    """
    data = receive_all(sock, flags=flags)
    if len(data) == 0:
        return None
    return deserialize_frame(data)


def send_frame(sock: socket.socket, frame: np.ndarray, flags: int = 0) -> None:
    """
    Send a frame over the socket.
    """
    data = serialize_frame(frame)
    send_all(sock, data, flags=flags)


def send_empty_frame(sock: socket.socket, flags: int = 0) -> None:
    """
    Send an empty frame over the socket.
    """
    # First, send the length of the frame
    send_all(sock, b"", flags=flags)


def send_json(sock: socket.socket, data: dict, flags: int = 0) -> None:
    """
    Send JSON data over the socket.
    """
    if len(data) == 0:
        send_all(sock, b"", flags=flags)
        return
    json_data = serialize_json(data)
    send_all(sock, json_data, flags=flags)


def receive_json(sock: socket.socket, flags: int = 0) -> dict:
    """
    Receive JSON data from the socket.
    """
    data = receive_all(sock, flags=flags)
    if len(data) == 0:
        return {}
    return deserialize_json(data)
