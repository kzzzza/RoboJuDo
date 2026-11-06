"""Latest-value TCP transport for the existing UFO latent packet.

The SSH tunnel supplies authentication and encryption. Both endpoints bind to
loopback; this module only handles framing, clock calibration, and reconnects.
"""

import logging
import queue
import socket
import struct
import threading
import time

try:
    from .latent_packet import PACKET_BYTES, decode_packet
except ImportError:  # standalone vision worker imports this directory directly
    from latent_packet import PACKET_BYTES, decode_packet

logger = logging.getLogger(__name__)

PING = struct.Struct("<4sd")
PONG = struct.Struct("<4sdd")
ACK = b"RZOK"
SESSION_SECONDS = 60.0
MAX_CLOCK_ERROR = 0.1


def _read_exact(sock: socket.socket, size: int, stop: threading.Event, timeout: float = 5.0) -> bytes:
    data = bytearray()
    deadline = time.monotonic() + timeout
    while len(data) < size and not stop.is_set():
        if time.monotonic() >= deadline:
            raise TimeoutError("latent TCP read timed out")
        try:
            chunk = sock.recv(size - len(data))
        except socket.timeout:
            continue
        if not chunk:
            raise ConnectionError("latent TCP connection closed")
        data.extend(chunk)
    if stop.is_set():
        raise ConnectionError("latent TCP transport stopped")
    return bytes(data)


class LatestTcpSender:
    """Never queue old camera frames while the robot is offline."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.stop_event = threading.Event()
        self.changed = threading.Condition()
        self.latest = None
        self.pending_person_loss = None
        self.generation = 0
        self.sent = 0
        self.thread = threading.Thread(target=self._run, name="LatentTcpSender", daemon=True)
        self.thread.start()

    def send(self, packet: bytes):
        if len(packet) != PACKET_BYTES:
            raise ValueError("Invalid latent packet length")
        with self.changed:
            self.latest = packet
            self.generation += 1
            if decode_packet(packet).person_lost:
                self.pending_person_loss = (self.generation, packet)
            self.changed.notify()

    def _run(self):
        was_connected = False
        while not self.stop_event.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=1.0) as sock:
                    sock.settimeout(0.5)
                    magic, _ = PING.unpack(_read_exact(sock, PING.size, self.stop_event, 2.0))
                    if magic != b"RZTP":
                        raise ValueError("Invalid latent TCP greeting")
                    received = time.monotonic()
                    sock.sendall(PONG.pack(b"RZTA", received, time.monotonic()))
                    if _read_exact(sock, len(ACK), self.stop_event, 2.0) != ACK:
                        raise ValueError("Latent TCP clock calibration rejected")
                    if not was_connected:
                        logger.info("Latent TCP connected to %s:%d", self.host, self.port)
                    was_connected = True
                    connected_at = time.monotonic()
                    last_sent = -1
                    while not self.stop_event.is_set() and time.monotonic() - connected_at < SESSION_SECONDS:
                        with self.changed:
                            if self.generation == last_sent:
                                self.changed.wait(timeout=0.1)
                            packet = self.latest
                            generation = self.generation
                            person_loss = self.pending_person_loss
                        if person_loss is not None:
                            loss_generation, loss_packet = person_loss
                            sock.sendall(loss_packet)
                            self.sent += 1
                            with self.changed:
                                if self.pending_person_loss == person_loss:
                                    self.pending_person_loss = None
                            if loss_generation == generation:
                                last_sent = generation
                        if packet is None or generation == last_sent:
                            continue
                        sock.sendall(packet)
                        last_sent = generation
                        self.sent += 1
            except (OSError, ConnectionError, ValueError) as exc:
                if was_connected and not self.stop_event.is_set():
                    logger.warning("Latent TCP disconnected: %s", exc)
                    was_connected = False
                else:
                    logger.debug("Latent TCP sender waiting for receiver: %s", exc)
            self.stop_event.wait(0.2)

    def close(self):
        self.stop_event.set()
        with self.changed:
            self.changed.notify_all()
        self.thread.join(timeout=2.0)


class LatestTcpReceiver:
    """Read TCP outside the 50 Hz loop and translate host capture time."""

    def __init__(self, port: int):
        self.stop_event = threading.Event()
        self.items = queue.Queue(maxsize=1)
        self.person_loss_items = queue.Queue(maxsize=1)
        self._link_lock = threading.Lock()
        self._connected = False
        self._disconnected_at = None
        self._active_session = -1
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.listener.bind(("127.0.0.1", port))
            self.listener.listen(1)
            self.listener.settimeout(0.5)
        except BaseException:
            self.listener.close()
            raise
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self._run, name="LatentTcpReceiver", daemon=True)
        self.thread.start()

    def _publish(self, item):
        target = self.person_loss_items if item[1].person_lost else self.items
        try:
            target.put_nowait(item)
        except queue.Full:
            target.get_nowait()
            target.put_nowait(item)

    def _run(self):
        session = 0
        while not self.stop_event.is_set():
            try:
                conn, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            session += 1
            try:
                with conn:
                    conn.settimeout(0.5)
                    sent = time.monotonic()
                    conn.sendall(PING.pack(b"RZTP", sent))
                    magic, host_received, host_sent = PONG.unpack(
                        _read_exact(conn, PONG.size, self.stop_event, 2.0)
                    )
                    arrived = time.monotonic()
                    rtt = (arrived - sent) - (host_sent - host_received)
                    if magic != b"RZTA" or rtt < 0 or rtt / 2 > MAX_CLOCK_ERROR:
                        raise ValueError("Latent TCP clock uncertainty exceeds 100 ms")
                    offset = ((host_received - sent) + (host_sent - arrived)) / 2
                    error = max(0.0, rtt / 2)
                    conn.sendall(ACK)
                    with self._link_lock:
                        self._connected = True
                        self._disconnected_at = None
                        self._active_session = session
                    logger.info("Latent TCP session ready; clock uncertainty %.1f ms", error * 1000)
                    last_sequence = -1
                    last_source_session = None
                    began = time.monotonic()
                    while not self.stop_event.is_set() and time.monotonic() - began < SESSION_SECONDS:
                        packet = _read_exact(conn, PACKET_BYTES, self.stop_event)
                        try:
                            item = decode_packet(packet)
                        except ValueError:
                            continue
                        if (item.sequence <= last_sequence and item.session_id == last_source_session
                                and not (item.eof and item.sequence == last_sequence)):
                            continue
                        if item.session_id != last_source_session:
                            last_source_session = item.session_id
                            last_sequence = -1
                        last_sequence = item.sequence
                        # Conservative age: include clock-calibration uncertainty.
                        local_capture = item.capture_time - offset - error
                        local_produced = item.produced_time - offset - error
                        received_at = time.monotonic()
                        if local_capture > received_at + MAX_CLOCK_ERROR:
                            continue
                        self._publish((session, item._replace(capture_time=local_capture,
                                                               produced_time=local_produced), received_at))
            except (OSError, ConnectionError, ValueError) as exc:
                logger.debug("Latent TCP receiver disconnected: %s", exc)
            finally:
                with self._link_lock:
                    if self._connected:
                        self._connected = False
                        self._disconnected_at = time.monotonic()

    def get_latest(self):
        try:
            return self.person_loss_items.get_nowait()
        except queue.Empty:
            pass
        try:
            return self.items.get_nowait()
        except queue.Empty:
            return None

    def link_ready(self) -> bool:
        with self._link_lock:
            return self._connected or (
                self._disconnected_at is not None
                and time.monotonic() - self._disconnected_at <= 0.5
            )

    def active_session(self) -> int:
        with self._link_lock:
            return self._active_session

    def close(self):
        self.stop_event.set()
        self.listener.close()
        self.thread.join(timeout=2.0)
