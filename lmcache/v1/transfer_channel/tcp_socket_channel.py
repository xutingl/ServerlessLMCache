# SPDX-License-Identifier: Apache-2.0

import asyncio
import ctypes
from dataclasses import dataclass
import os
import threading
import time
from typing import Optional, Union

# Third Party
import msgspec
import torch
import zmq

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.logging import init_logger
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

logger = init_logger(__name__)

_DEFAULT_DATA_CHUNK_BYTES = 512 * 1024
_DATA_SOCKET_HWM = 2048


class TcpSocketWriteHeader(msgspec.Struct, tag=True):
    req_id: str
    write_id: int
    remote_indexes: list[int]
    payload_sizes: list[int]


class TcpSocketChunkHeader(msgspec.Struct, tag=True):
    req_id: str
    write_id: int
    object_index: int
    offset: int
    size: int


class TcpSocketWriteResponse(msgspec.Struct, tag=True):
    req_id: str
    write_id: int
    written: int
    error: str = ""


TcpSocketDataMessage = Union[TcpSocketWriteHeader, TcpSocketChunkHeader]


@dataclass
class _PendingTcpSocketWrite:
    identity: bytes
    header: TcpSocketWriteHeader
    start_time: float
    decode_ms: float
    expected_bytes: int
    received_bytes: int = 0
    payload_frames: int = 0
    write_ms: float = 0.0


class TcpSocketChannel(PySocketChannel):
    """ZMQ/TCP data-plane channel that writes directly into its paged buffer."""

    def __init__(self, async_mode: bool = False, **kwargs):
        super().__init__(async_mode=async_mode, **kwargs)
        self._data_sockets: dict[tuple[int, str], zmq.Socket] = {}
        self._data_sockets_lock = threading.Lock()
        self._data_receiver_socket: Optional[zmq.Socket] = None
        self._data_receiver_thread: Optional[threading.Thread] = None
        self._data_receiver_buffer_owner = None
        self._data_receiver_buffer: Optional[memoryview] = None
        self._data_chunk_bytes = _tcp_socket_data_chunk_bytes()
        self._next_write_id = 0
        self._next_write_id_lock = threading.Lock()
        self._pending_writes: dict[
            tuple[bytes, str, int],
            _PendingTcpSocketWrite,
        ] = {}

        data_listen_url = kwargs.get("data_listen_url")
        if data_listen_url is not None:
            if self.role not in {"receiver", "both"}:
                raise ValueError(
                    "data_listen_url is only valid for a TCP receiver or both role"
                )
            if kwargs.get("device") != "cpu":
                raise ValueError(
                    "TcpSocketChannel direct receive requires a CPU paged buffer"
                )
            self._start_data_receiver(data_listen_url)

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
                write_id=self._allocate_write_id(),
                remote_indexes=remote_indexes,
                payload_sizes=[payload.nbytes for payload in payloads],
            )
            _send_write_stream(socket, header, payloads, self._data_chunk_bytes)
            resp_bytes = socket.recv()
            resp = msgspec.msgpack.decode(resp_bytes, type=TcpSocketWriteResponse)
        except Exception:
            self._drop_data_socket(receiver_data_url, socket)
            raise

        if resp.error:
            raise RuntimeError(f"TcpSocketChannel write failed: {resp.error}")
        if resp.req_id != header.req_id or resp.write_id != header.write_id:
            raise RuntimeError(
                "TcpSocketChannel write response mismatch: "
                f"expected=({header.req_id!r}, {header.write_id}) "
                f"got=({resp.req_id!r}, {resp.write_id})"
            )
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
        self.running = False
        if self._data_receiver_thread is not None:
            self._data_receiver_thread.join()
            self._data_receiver_thread = None
        if self._data_receiver_socket is not None:
            self._data_receiver_socket.close(linger=0)
            self._data_receiver_socket = None
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
                    zmq.DEALER,
                    "connect",
                )
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.SNDHWM, _DATA_SOCKET_HWM)
                socket.setsockopt(zmq.RCVHWM, _DATA_SOCKET_HWM)
                self._data_sockets[key] = socket
            return socket

    def _allocate_write_id(self) -> int:
        with self._next_write_id_lock:
            write_id = self._next_write_id
            self._next_write_id += 1
            return write_id

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

    def _start_data_receiver(self, data_listen_url: str) -> None:
        if self.buffer_ptr is None or self.buffer_size <= 0:
            raise ValueError("TcpSocketChannel receiver requires a valid paged buffer")

        buffer_type = ctypes.c_ubyte * self.buffer_size
        self._data_receiver_buffer_owner = buffer_type.from_address(self.buffer_ptr)
        self._data_receiver_buffer = memoryview(
            self._data_receiver_buffer_owner
        ).cast("B")

        socket = get_zmq_socket(
            self.zmq_context,
            data_listen_url,
            "tcp",
            zmq.ROUTER,
            "bind",
        )
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, _DATA_SOCKET_HWM)
        socket.setsockopt(zmq.RCVHWM, _DATA_SOCKET_HWM)
        socket.setsockopt(zmq.RCVTIMEO, 100)
        self._data_receiver_socket = socket
        self._data_receiver_thread = threading.Thread(
            target=self._data_receiver_loop,
            daemon=True,
            name="lmcache-tcp-data-recv",
        )
        self._data_receiver_thread.start()

    def _data_receiver_loop(self) -> None:
        assert self._data_receiver_socket is not None
        while self.running:
            try:
                identity, control_frame = self._recv_router_control_frame()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if self.running:
                    logger.exception("TCP data receiver stopped unexpectedly")
                return
            except Exception:
                if self.running:
                    logger.exception("Failed to read TCP data control frame")
                continue

            loop_start = time.perf_counter()
            req_id = ""
            write_id = -1
            try:
                stage_start = time.perf_counter()
                message = msgspec.msgpack.decode(
                    bytes(control_frame), type=TcpSocketDataMessage
                )
                decode_ms = _elapsed_ms(stage_start)

                if isinstance(message, TcpSocketWriteHeader):
                    req_id = message.req_id
                    write_id = message.write_id
                    self._handle_write_header(identity, message, loop_start, decode_ms)
                    continue

                req_id = message.req_id
                write_id = message.write_id
                completed = self._handle_chunk_header(identity, message)
                if completed is None:
                    continue

                write_resp = TcpSocketWriteResponse(
                    req_id=req_id,
                    write_id=write_id,
                    written=len(completed.header.remote_indexes),
                )
                response_send_ms = self._send_write_response(
                    identity,
                    write_resp,
                )
                self._log_completed_write(
                    completed,
                    response_send_ms,
                    error="",
                )
            except Exception as exc:
                self._discard_remaining_frames()
                completed = self._pending_writes.pop(
                    (identity, req_id, write_id),
                    None,
                )
                write_resp = TcpSocketWriteResponse(
                    req_id=req_id,
                    write_id=write_id,
                    written=0,
                    error=f"{type(exc).__name__}: {exc}",
                )
                try:
                    response_send_ms = self._send_write_response(identity, write_resp)
                except zmq.ZMQError:
                    if self.running:
                        logger.exception("Failed to acknowledge TCP data write")
                    continue
                if completed is None:
                    logger.info(
                        "[pd-demo-timing] receiver_data_loop "
                        "req_id=%s chunks=0 zmq_frames=0 decode_ms=%.4f "
                        "write_ms=0.0000 response_send_ms=%.4f total_ms=%.4f "
                        "error=%s",
                        req_id,
                        _elapsed_ms(loop_start),
                        response_send_ms,
                        _elapsed_ms(loop_start),
                        write_resp.error,
                    )
                else:
                    self._log_completed_write(
                        completed,
                        response_send_ms,
                        error=write_resp.error,
                    )

    def _recv_router_control_frame(self) -> tuple[bytes, zmq.Frame]:
        assert self._data_receiver_socket is not None
        identity_frame = self._data_receiver_socket.recv(copy=False)
        if not self._data_receiver_socket.getsockopt(zmq.RCVMORE):
            raise ValueError("missing TCP data control frame")
        control_frame = self._data_receiver_socket.recv(copy=False)
        return bytes(identity_frame), control_frame

    def _handle_write_header(
        self,
        identity: bytes,
        header: TcpSocketWriteHeader,
        start_time: float,
        decode_ms: float,
    ) -> None:
        if len(header.remote_indexes) != len(header.payload_sizes):
            raise ValueError(
                "remote_indexes length must match payload_sizes length: "
                f"{len(header.remote_indexes)} != {len(header.payload_sizes)}"
            )
        for remote_index, payload_size in zip(
            header.remote_indexes,
            header.payload_sizes,
            strict=True,
        ):
            offset = remote_index * self.align_bytes
            if (
                remote_index < 0
                or payload_size <= 0
                or payload_size > self.align_bytes
                or offset + payload_size > self.buffer_size
            ):
                raise ValueError(
                    "invalid paged-buffer write: "
                    f"index={remote_index} size={payload_size} "
                    f"page_size={self.align_bytes} buffer_size={self.buffer_size}"
                )

        key = (identity, header.req_id, header.write_id)
        if key in self._pending_writes:
            raise ValueError(
                "duplicate TCP data write "
                f"req_id={header.req_id!r} write_id={header.write_id}"
            )
        self._pending_writes[key] = _PendingTcpSocketWrite(
            identity=identity,
            header=header,
            start_time=start_time,
            decode_ms=decode_ms,
            expected_bytes=sum(header.payload_sizes),
        )

    def _handle_chunk_header(
        self,
        identity: bytes,
        chunk_header: TcpSocketChunkHeader,
    ) -> Optional[_PendingTcpSocketWrite]:
        key = (identity, chunk_header.req_id, chunk_header.write_id)
        pending = self._pending_writes.get(key)
        if pending is None:
            # A header validation error may already have been reported while the
            # sender had queued this request's payload chunks. Drop those orphan
            # chunks silently so a later socket does not receive stale errors.
            self._discard_remaining_frames()
            return None
        if self._data_receiver_buffer is None:
            raise RuntimeError("TCP data receiver buffer was not initialized")
        if not self._data_receiver_socket.getsockopt(zmq.RCVMORE):
            raise ValueError("missing TCP data payload frame")

        if (
            chunk_header.object_index < 0
            or chunk_header.object_index >= len(pending.header.remote_indexes)
        ):
            raise ValueError(
                "invalid TCP data object index: "
                f"{chunk_header.object_index} for {len(pending.header.remote_indexes)}"
            )
        payload_size = pending.header.payload_sizes[chunk_header.object_index]
        if (
            chunk_header.offset < 0
            or chunk_header.size <= 0
            or chunk_header.offset + chunk_header.size > payload_size
        ):
            raise ValueError(
                "invalid TCP data chunk: "
                f"object_index={chunk_header.object_index} "
                f"offset={chunk_header.offset} size={chunk_header.size} "
                f"payload_size={payload_size}"
            )

        remote_index = pending.header.remote_indexes[chunk_header.object_index]
        buffer_offset = remote_index * self.align_bytes + chunk_header.offset
        target = self._data_receiver_buffer[
            buffer_offset : buffer_offset + chunk_header.size
        ]
        stage_start = time.perf_counter()
        received = self._data_receiver_socket.recv_into(target)
        pending.write_ms += _elapsed_ms(stage_start)
        if received != chunk_header.size:
            raise ValueError(
                "payload size mismatch for TCP data chunk: "
                f"{received} != {chunk_header.size}"
            )
        pending.received_bytes += received
        pending.payload_frames += 1

        if pending.received_bytes > pending.expected_bytes:
            raise ValueError(
                "received too many TCP data bytes: "
                f"{pending.received_bytes} > {pending.expected_bytes}"
            )
        if pending.received_bytes == pending.expected_bytes:
            return self._pending_writes.pop(key)
        return None

    def _send_write_response(
        self,
        identity: bytes,
        response: TcpSocketWriteResponse,
    ) -> float:
        assert self._data_receiver_socket is not None
        stage_start = time.perf_counter()
        self._data_receiver_socket.send(identity, flags=zmq.SNDMORE)
        self._data_receiver_socket.send(msgspec.msgpack.encode(response))
        return _elapsed_ms(stage_start)

    def _log_completed_write(
        self,
        completed: _PendingTcpSocketWrite,
        response_send_ms: float,
        error: str,
    ) -> None:
        written = len(completed.header.remote_indexes)
        logger.info(
            "[pd-demo-timing] receiver_tcp_socket_write "
            "req_id=%s chunks=%d total_ms=%.4f",
            completed.header.req_id,
            written,
            completed.write_ms,
        )
        logger.info(
            "[pd-demo-timing] receiver_data_loop "
            "req_id=%s chunks=%d zmq_frames=%d decode_ms=%.4f write_ms=%.4f "
            "response_send_ms=%.4f total_ms=%.4f error=%s",
            completed.header.req_id,
            written,
            completed.payload_frames,
            completed.decode_ms,
            completed.write_ms,
            response_send_ms,
            _elapsed_ms(completed.start_time),
            error,
        )

    def _discard_remaining_frames(self) -> None:
        assert self._data_receiver_socket is not None
        while self._data_receiver_socket.getsockopt(zmq.RCVMORE):
            self._data_receiver_socket.recv(copy=False)


def _send_write_stream(
    socket: zmq.Socket,
    header: TcpSocketWriteHeader,
    payloads: list[memoryview],
    chunk_bytes: int,
) -> None:
    socket.send(msgspec.msgpack.encode(header))
    for object_index, payload in enumerate(payloads):
        offset = 0
        while offset < payload.nbytes:
            size = min(chunk_bytes, payload.nbytes - offset)
            chunk_header = TcpSocketChunkHeader(
                req_id=header.req_id,
                write_id=header.write_id,
                object_index=object_index,
                offset=offset,
                size=size,
            )
            socket.send(msgspec.msgpack.encode(chunk_header), flags=zmq.SNDMORE)
            socket.send(payload[offset : offset + size], copy=False)
            offset += size


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


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _tcp_socket_data_chunk_bytes() -> int:
    raw_value = os.environ.get("LMCACHE_TCP_SOCKET_DATA_CHUNK_BYTES")
    if raw_value is None:
        return _DEFAULT_DATA_CHUNK_BYTES
    value = int(raw_value)
    if value <= 0:
        raise ValueError("LMCACHE_TCP_SOCKET_DATA_CHUNK_BYTES must be positive")
    return value
