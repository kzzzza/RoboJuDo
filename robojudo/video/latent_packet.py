"""Fixed-size local datagram carrying one UFO latent and source timing."""

import struct
import time
from typing import NamedTuple

import numpy as np

MAGIC = b"RZ02"
VARIANTS = ("original", "tcn", "mlp")
HEADER = struct.Struct("<4sBBIQIddd")
# magic, variant, flags, sample index, session, source frame, sample time,
# capture monotonic time, produced monotonic time
Z_BYTES = 256 * 4
PACKET_BYTES = HEADER.size + Z_BYTES
FLAG_EOF = 1
FLAG_PERSON_LOST = 2


class LatentPacket(NamedTuple):
    variant: str
    sequence: int
    capture_time: float
    z: np.ndarray
    eof: bool
    session_id: int
    source_frame_id: int
    sample_time: float
    produced_time: float
    person_lost: bool


def encode_packet(variant: str, sequence: int, capture_time: float, z: np.ndarray, *,
                  eof=False, session_id=0, source_frame_id=None,
                  sample_time=0.0, produced_time=None, person_lost=False) -> bytes:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown latent variant: {variant}")
    value = np.asarray(z, dtype="<f4")
    if value.shape != (256,) or not np.all(np.isfinite(value)):
        raise ValueError("Expected finite z[256]")
    if person_lost and (eof or np.any(value)):
        raise ValueError("Person-lost packet must contain zero z and cannot be EOF")
    if source_frame_id is None:
        source_frame_id = sequence
    if produced_time is None:
        produced_time = time.monotonic()
    if (not 0 <= sequence <= 0xffffffff or not 0 <= source_frame_id <= 0xffffffff
            or not 0 <= session_id <= 0xffffffffffffffff
            or not all(np.isfinite(v) for v in (capture_time, produced_time, sample_time))
            or capture_time <= 0 or produced_time < capture_time or sample_time < 0):
        raise ValueError("Invalid latent timing or identity")
    flags = (FLAG_EOF if eof else 0) | (FLAG_PERSON_LOST if person_lost else 0)
    return HEADER.pack(MAGIC, VARIANTS.index(variant), flags,
                       sequence, session_id, source_frame_id, sample_time,
                       capture_time, produced_time) + value.tobytes()


def decode_packet(packet: bytes) -> LatentPacket:
    if len(packet) != PACKET_BYTES:
        raise ValueError("Invalid latent packet length")
    magic, variant_id, flags, sequence, session_id, source_id, sample_time, capture_time, produced_time = HEADER.unpack_from(packet)
    if magic != MAGIC or variant_id >= len(VARIANTS) or flags & ~(FLAG_EOF | FLAG_PERSON_LOST):
        raise ValueError("Invalid latent packet header")
    z = np.frombuffer(packet, dtype="<f4", count=256, offset=HEADER.size).copy()
    if (not all(np.isfinite(v) for v in (capture_time, produced_time, sample_time))
            or capture_time <= 0 or produced_time < capture_time or sample_time < 0
            or not np.all(np.isfinite(z)) or
            (flags & FLAG_PERSON_LOST and (flags & FLAG_EOF or np.any(z)))):
        raise ValueError("Invalid latent packet data")
    return LatentPacket(VARIANTS[variant_id], sequence, capture_time, z, bool(flags & FLAG_EOF),
                        session_id, source_id, sample_time, produced_time,
                        bool(flags & FLAG_PERSON_LOST))
