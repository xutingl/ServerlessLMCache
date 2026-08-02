# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from typing import Optional, Union

# Third Party
import msgspec
import torch
import zmq

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.rpc_utils import get_zmq_socket
from lmcache.v1.transfer_channel.py_socket_channel import (
    PySocketChannel,
    PySocketInitRequest,
    PySocketMsg,
)
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsgBase,
    InitSideRetMsgBase,
)


class TcpSocketWriteRequest(msgspec.Struct, tag=True):
    req_id: str
    remote_indexes: list[int]
    payloads: list[bytes]


class TcpSocketWriteHeader(msgspec.Struct, tag=True):
    req_id: str
    remote_indexes: list[int]


class TcpSocketWriteResponse(msgspec.Struct, tag=True):
    req_id: str
    written: int
    error: str = ""


class TcpSocketChannel(PySocketChannel):
    """Simple ZMQ/TCP data-plane channel for PD transfer."""

    def __init__(self, async_mode: bool = False, **kwargs):
        super().__init__(async_mode=async_mode, **kwargs)
        self._data_sockets: dict[tuple[int, str], zmq.Socket] = {}
        self._data_sockets_lock = threading.Lock()

    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        init_req = PySocketInitRequest(peer_init_url=self.peer_init_url)
        init_tmp_socket.send(msgspec.msgpack.encode(init_req))

        init_resp_bytes = init_tmp_socket.recv()
        _ = msgspec.msgpack.decode(init_resp_bytes, type=PySocketMsg)

        self.remote_connections[peer_id] = {
            "peer_init_url": peer_init_url,
            "local_id": local_id,
        }

        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = self.send_init_side_msg(init_tmp_socket, init_side_msg)

        init_tmp_socket.close()
        return init_ret_msg

    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        return receiver_or_sender_id in self.remote_connections

    def get_local_mem_indices(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> list[int]:
        if isinstance(objects[0], bytes):
            raise NotImplementedError(
                "Sending raw bytes is not supported in TcpSocketChannel"
            )
        local_indices: list[int] = []
        for mem_obj in objects:
            assert isinstance(mem_obj, MemoryObj)
            local_indices.append(mem_obj.meta.address)
        return local_indices

    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        receiver_data_url = transfer_spec.get("receiver_data_url")
        if not receiver_data_url:
            raise ValueError("TcpSocketChannel requires transfer_spec.receiver_data_url")

        remote_indexes = list(transfer_spec["remote_indexes"])
        payloads = [_memory_obj_to_buffer(obj) for obj in objects]
        if len(remote_indexes) != len(payloads):
            raise ValueError(
                "remote_indexes length must match objects length: "
                f"{len(remote_indexes)} != {len(payloads)}"
            )

        socket = self._get_data_socket(receiver_data_url)
        try:
            header = TcpSocketWriteHeader(
                req_id=str(transfer_spec.get("req_id") or ""),
                remote_indexes=remote_indexes,
            )
            _send_write_multipart(socket, header, payloads)
            resp_bytes = socket.recv()
            resp = msgspec.msgpack.decode(resp_bytes, type=TcpSocketWriteResponse)
        except Exception:
            self._drop_data_socket(receiver_data_url, socket)
            raise

        if resp.error:
            raise RuntimeError(f"TcpSocketChannel write failed: {resp.error}")
        return int(resp.written)

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        return await asyncio.to_thread(self.batched_write, objects, transfer_spec)

    def batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    def batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    def close(self):
        with self._data_sockets_lock:
            sockets = list(self._data_sockets.values())
            self._data_sockets.clear()
        for socket in sockets:
            socket.close(linger=0)
        super().close()

    def _get_data_socket(self, receiver_data_url: str) -> zmq.Socket:
        key = (threading.get_ident(), receiver_data_url)
        with self._data_sockets_lock:
            socket = self._data_sockets.get(key)
            if socket is None or socket.closed:
                socket = get_zmq_socket(
                    self.zmq_context,
                    receiver_data_url,
                    "tcp",
                    zmq.REQ,
                    "connect",
                )
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.SNDHWM, 1)
                socket.setsockopt(zmq.RCVHWM, 1)
                self._data_sockets[key] = socket
            return socket

    def _drop_data_socket(
        self,
        receiver_data_url: str,
        socket: zmq.Socket,
    ) -> None:
        key = (threading.get_ident(), receiver_data_url)
        with self._data_sockets_lock:
            if self._data_sockets.get(key) is socket:
                self._data_sockets.pop(key, None)
        socket.close(linger=0)


def _send_write_multipart(
    socket: zmq.Socket,
    header: TcpSocketWriteHeader,
    payloads: list[memoryview],
) -> None:
    header_bytes = msgspec.msgpack.encode(header)
    if not payloads:
        socket.send(header_bytes)
        return

    socket.send(header_bytes, flags=zmq.SNDMORE)
    for payload in payloads[:-1]:
        socket.send(payload, flags=zmq.SNDMORE, copy=False)
    socket.send(payloads[-1], copy=False)


def _memory_obj_to_buffer(obj: Union[bytes, MemoryObj]) -> memoryview:
    if isinstance(obj, bytes):
        return memoryview(obj)
    if not isinstance(obj, MemoryObj) or obj.tensor is None:
        raise ValueError("TcpSocketChannel can only write MemoryObj with tensor data")

    tensor = obj.tensor.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return memoryview(tensor.view(torch.uint8).reshape(-1).numpy())
