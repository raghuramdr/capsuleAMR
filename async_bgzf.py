"""
Copied and modified from Bio-Python bgzf module.
https://github.com/biopython/biopython/blob/master/Bio/bgzf.py
"""

import struct
import zlib

_bgzf_header = b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00\x42\x43\x02\x00"
_bgzf_eof = b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00\x1b\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00"


def make_virtual_offset(block_start_offset, within_block_offset):
    """Compute a BGZF virtual offset from block start and within block offsets.
    The BAM indexing scheme records read positions using a 64 bit
    'virtual offset', comprising in C terms:
    block_start_offset << 16 | within_block_offset
    Here block_start_offset is the file offset of the BGZF block
    start (unsigned integer using up to 64-16 = 48 bits), and
    within_block_offset within the (decompressed) block (unsigned
    16 bit integer).
    >>> make_virtual_offset(0, 0)
    0
    >>> make_virtual_offset(0, 1)
    1
    >>> make_virtual_offset(0, 2**16 - 1)
    65535
    >>> make_virtual_offset(0, 2**16)
    Traceback (most recent call last):
    ...
    ValueError: Require 0 <= within_block_offset < 2**16, got 65536
    >>> 65536 == make_virtual_offset(1, 0)
    True
    >>> 65537 == make_virtual_offset(1, 1)
    True
    >>> 131071 == make_virtual_offset(1, 2**16 - 1)
    True
    >>> 6553600000 == make_virtual_offset(100000, 0)
    True
    >>> 6553600001 == make_virtual_offset(100000, 1)
    True
    >>> 6553600010 == make_virtual_offset(100000, 10)
    True
    >>> make_virtual_offset(2**48, 0)
    Traceback (most recent call last):
    ...
    ValueError: Require 0 <= block_start_offset < 2**48, got 281474976710656
    """
    if within_block_offset < 0 or within_block_offset >= 65536:
        raise ValueError(
            "Require 0 <= within_block_offset < 2**16, got %i" % within_block_offset
        )
    if block_start_offset < 0 or block_start_offset >= 281474976710656:
        raise ValueError(
            "Require 0 <= block_start_offset < 2**48, got %i" % block_start_offset
        )
    return (block_start_offset << 16) | within_block_offset


class AsyncBgzfWriter:
    """Define an async BGZFWriter object."""

    def __init__(self, async_fileobj, compresslevel=6):
        """Initialize the class."""

        self._handle = async_fileobj
        self._buffer = b""
        self.compresslevel = compresslevel

    def _compress_block(self, block):
        start_offset = self._handle.tell()
        if len(block) > 65536:
            raise ValueError(f"{len(block)} Block length > 65536")
        # Giving a negative window bits means no gzip/zlib headers,
        # -15 used in samtools
        c = zlib.compressobj(
            self.compresslevel, zlib.DEFLATED, -15, zlib.DEF_MEM_LEVEL, 0
        )
        compressed = c.compress(block) + c.flush()
        del c
        if len(compressed) > 65536:
            raise RuntimeError(
                "TODO - Didn't compress enough, try less data in this block"
            )
        crc = zlib.crc32(block)
        # Should cope with a mix of Python platforms...
        if crc < 0:
            crc = struct.pack("<i", crc)
        else:
            crc = struct.pack("<I", crc)
        bsize = struct.pack("<H", len(compressed) + 25)  # includes -1
        crc = struct.pack("<I", zlib.crc32(block) & 0xFFFFFFFF)
        uncompressed_length = struct.pack("<I", len(block))
        # Fixed 16 bytes,
        # gzip magic bytes (4) mod time (4),
        # gzip flag (1), os (1), extra length which is six (2),
        # sub field which is BC (2), sub field length of two (2),
        # Variable data,
        # 2 bytes: block length as BC sub field (2)
        # X bytes: the data
        # 8 bytes: crc (4), uncompressed data length (4)
        data = _bgzf_header + bsize + compressed + crc + uncompressed_length
        return data

    async def _write_block(self, block):
        """Write provided data to file as a single BGZF compressed block (PRIVATE)."""
        data = self._compress_block(block)
        await self._handle.write(data)

    async def write(self, data):
        """Write method for the class."""
        if isinstance(data, str):
            # When reading we can't cope with multi-byte characters
            # being split between BGZF blocks, so we restrict to a
            # single byte encoding - like ASCII or latin-1.
            # On output we could probably allow any encoding, as we
            # don't care about splitting unicode characters between blocks
            data = data.encode("latin-1")
        # block_size = 2**16 = 65536
        data_len = len(data)
        if len(self._buffer) + data_len < 65536:
            # print("Cached %r" % data)
            self._buffer += data
        else:
            # print("Got %r, writing out some data..." % data)
            self._buffer += data
            while len(self._buffer) >= 65536:
                await self._write_block(self._buffer[:65536])
                self._buffer = self._buffer[65536:]

    async def flush(self):
        """Flush data explicitally."""
        while len(self._buffer) >= 65536:
            await self._write_block(self._buffer[:65535])
            self._buffer = self._buffer[65535:]
        await self._write_block(self._buffer)
        self._buffer = b""
        await self._handle.flush()

    async def close(self):
        """Flush data, write 28 bytes BGZF EOF marker, and close BGZF file.
        samtools will look for a magic EOF marker, just a 28 byte empty BGZF
        block, and if it is missing warns the BAM file may be truncated. In
        addition to samtools writing this block, so too does bgzip - so this
        implementation does too.
        """
        if self._buffer:
            await self.flush()
        await self._handle.write(_bgzf_eof)
        await self._handle.flush()
        await self._handle.close()

    def tell(self):
        """Return a BGZF 64-bit virtual offset."""
        return make_virtual_offset(self._handle.tell(), len(self._buffer))

    def seekable(self):
        """Return True indicating the BGZF supports random access."""
        # Not seekable, but we do support tell...
        return False

    def isatty(self):
        """Return True if connected to a TTY device."""
        return False

    def fileno(self):
        """Return integer file descriptor."""
        return self._handle.fileno()