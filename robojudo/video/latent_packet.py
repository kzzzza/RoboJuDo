"""Fixed-size local datagram carrying one UFO latent and source timing."""

import struct

import numpy as np

MAGIC = b"RZ01"
VARIANTS = ("original", "tcn", "mlp")
HEADER = struct.Struct("<4sBBId")  # magic, variant, flags, sequence, capture monotonic time
Z_BYTES = 256 * 4
PACKET_BYTES = HEADER.size + Z_BYTES
FLAG_EOF = 1


def encode_packet(variant: str, sequence: int, capture_time: float, z: np.ndarray, *, eof=False) -> bytes:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown latent variant: {variant}")
    value = np.asarray(z, dtype="<f4")
    if value.shape != (256,) or not np.all(np.isfinite(value)):
        raise ValueError("Expected finite z[256]")
    if not np.isfinite(capture_time) or capture_time <= 0:
        raise ValueError("Invalid capture time")
    return HEADER.pack(MAGIC, VARIANTS.index(variant), FLAG_EOF if eof else 0, sequence, capture_time) + value.tobytes()


def decode_packet(packet: bytes) -> tuple[str, int, float, np.ndarray, bool]:
    if len(packet) != PACKET_BYTES:
        raise ValueError("Invalid latent packet length")
    magic, variant_id, flags, sequence, capture_time = HEADER.unpack_from(packet)
    if magic != MAGIC or variant_id >= len(VARIANTS) or flags & ~FLAG_EOF:
        raise ValueError("Invalid latent packet header")
    z = np.frombuffer(packet, dtype="<f4", count=256, offset=HEADER.size).copy()
    if not np.isfinite(capture_time) or capture_time <= 0 or not np.all(np.isfinite(z)):
        raise ValueError("Invalid latent packet data")
    return VARIANTS[variant_id], sequence, capture_time, z, bool(flags & FLAG_EOF)
