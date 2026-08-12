# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import unittest

import zmq

from lmcache.v1.transfer_channel.tcp_socket_channel import (
    TcpSocketChannel,
    TcpSocketChunkHeader,
    TcpSocketWriteHeader,
    _coalesce_contiguous_payloads,
)


class _ReceiveSocket:
    def __init__(self, payload: bytes):
        self.payload = payload

    def getsockopt(self, option: int) -> bool:
        assert option == zmq.RCVMORE
        return True

    def recv_into(self, target: memoryview) -> int:
        target[:] = self.payload
        return len(self.payload)


def _make_receiver(payload: bytes = b"data") -> tuple[TcpSocketChannel, bytearray]:
    channel = object.__new__(TcpSocketChannel)
    channel.align_bytes = 4096
    channel.buffer_size = 16 * 4096
    channel._pending_writes = {}
    channel._data_receiver_socket = _ReceiveSocket(payload)
    buffer = bytearray(channel.buffer_size)
    channel._data_receiver_buffer = memoryview(buffer)
    return channel, buffer


class TcpSocketChannelTest(unittest.TestCase):
    def test_direct_receive_uses_byte_offset(self) -> None:
        channel, buffer = _make_receiver()
        header = TcpSocketWriteHeader(
            req_id="request",
            write_id=1,
            remote_indexes=[2 * 4096],
            remote_capacities=[4096],
            payload_sizes=[4],
        )
        channel._handle_write_header(b"sender", header, time.perf_counter(), 0.0)

        completed = channel._handle_chunk_header(
            b"sender",
            TcpSocketChunkHeader(
                req_id="request",
                write_id=1,
                object_index=0,
                offset=0,
                size=4,
            ),
        )

        self.assertIsNotNone(completed)
        self.assertEqual(buffer[2 * 4096 : 2 * 4096 + 4], b"data")
        self.assertEqual(buffer[8 * 4096 : 8 * 4096 + 4], b"\x00" * 4)

    def test_receive_rejects_payload_larger_than_allocation(self) -> None:
        channel, _ = _make_receiver()
        header = TcpSocketWriteHeader(
            req_id="request",
            write_id=1,
            remote_indexes=[4096],
            remote_capacities=[4096],
            payload_sizes=[4097],
        )

        with self.assertRaisesRegex(ValueError, "invalid receive-buffer write"):
            channel._handle_write_header(
                b"sender", header, time.perf_counter(), 0.0
            )

    def test_receive_rejects_unaligned_offset(self) -> None:
        channel, _ = _make_receiver()
        header = TcpSocketWriteHeader(
            req_id="request",
            write_id=1,
            remote_indexes=[1],
            remote_capacities=[4096],
            payload_sizes=[4],
        )

        with self.assertRaisesRegex(ValueError, "invalid receive-buffer write"):
            channel._handle_write_header(
                b"sender", header, time.perf_counter(), 0.0
            )

    def test_contiguous_objects_use_one_payload_span(self) -> None:
        data = bytearray(b"abcdefgh")
        payloads = [memoryview(data)[:4], memoryview(data)[4:]]
        header = TcpSocketWriteHeader(
            req_id="request",
            write_id=1,
            remote_indexes=[4096, 8192],
            remote_capacities=[4096, 4096],
            payload_sizes=[4, 4],
        )

        count, payload = _coalesce_contiguous_payloads(payloads, header, 0)

        self.assertEqual(count, 1)
        self.assertEqual(bytes(payload), b"abcd")

        header.remote_indexes = [4096, 4100]
        header.remote_capacities = [4, 4]
        count, payload = _coalesce_contiguous_payloads(payloads, header, 0)
        self.assertEqual(count, 2)
        self.assertEqual(bytes(payload), b"abcdefgh")

    def test_receive_writes_contiguous_object_span(self) -> None:
        payload = b"a" * (2 * 4096)
        channel, buffer = _make_receiver(payload=payload)
        header = TcpSocketWriteHeader(
            req_id="request",
            write_id=1,
            remote_indexes=[4096, 8192],
            remote_capacities=[4096, 4096],
            payload_sizes=[4096, 4096],
        )
        channel._handle_write_header(b"sender", header, time.perf_counter(), 0.0)

        completed = channel._handle_chunk_header(
            b"sender",
            TcpSocketChunkHeader(
                req_id="request",
                write_id=1,
                object_index=0,
                object_count=2,
                offset=0,
                size=len(payload),
            ),
        )

        self.assertIsNotNone(completed)
        self.assertEqual(buffer[4096 : 3 * 4096], payload)

if __name__ == "__main__":
    unittest.main()
