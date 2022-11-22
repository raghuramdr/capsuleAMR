"""
Copied and modified from Bio-Python bgzf module.
https://github.com/biopython/biopython/blob/master/Bio/bgzf.py
"""

import struct
import zlib

_bgzf_magic = b"\x1f\x8b\x08\x04"
_bgzf_header = b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00\x42\x43\x02\x00"
_bgzf_eof = b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00\x1b\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00"
_bytes_BC = b"BC"


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


async def _async_load_bgzf_block(handle, text_mode=False):
    """Load the next BGZF block of compressed data (PRIVATE).
    Returns a tuple (block size and data), or at end of file
    will raise StopIteration.
    """
    magic = await handle.read(4)
    if not magic:
        # End of file - should we signal this differently now?
        # See https://www.python.org/dev/peps/pep-0479/
        raise StopAsyncIteration
    if magic != _bgzf_magic:
        raise ValueError(
            r"A BGZF (e.g. a BAM file) block should start with "
            r"%r, not %r; handle.tell() now says %r"
            % (_bgzf_magic, magic, await handle.tell())
        )
    gzip_mod_time, gzip_extra_flags, gzip_os, extra_len = struct.unpack(
        "<LBBH", await handle.read(8)
    )

    block_size = None
    x_len = 0
    while x_len < extra_len:
        subfield_id = await handle.read(2)
        subfield_len = struct.unpack("<H", await handle.read(2))[0]  # uint16_t
        subfield_data = await handle.read(subfield_len)
        x_len += subfield_len + 4
        if subfield_id == _bytes_BC:
            if subfield_len != 2:
                raise ValueError("Wrong BC payload length")
            if block_size is not None:
                raise ValueError("Two BC subfields?")
            block_size = struct.unpack("<H", subfield_data)[0] + 1  # uint16_t
    if x_len != extra_len:
        raise ValueError(f"x_len and extra_len differ {x_len}, {extra_len}")
    if block_size is None:
        raise ValueError("Missing BC, this isn't a BGZF file!")
    # Now comes the compressed data, CRC, and length of uncompressed data.
    deflate_size = block_size - 1 - extra_len - 19
    d = zlib.decompressobj(-15)  # Negative window size means no headers
    data = d.decompress(await handle.read(deflate_size)) + d.flush()
    expected_crc = await handle.read(4)
    expected_size = struct.unpack("<I", await handle.read(4))[0]
    if expected_size != len(data):
        raise RuntimeError("Decompressed to %i, not %i" % (len(data), expected_size))
    # Should cope with a mix of Python platforms...
    crc = zlib.crc32(data)
    if crc < 0:
        crc = struct.pack("<i", crc)
    else:
        crc = struct.pack("<I", crc)
    if expected_crc != crc:
        raise RuntimeError(f"CRC is {crc}, not {expected_crc}")
    if text_mode:
        # Note ISO-8859-1 aka Latin-1 preserves first 256 chars
        # (i.e. ASCII), but critically is a single byte encoding
        return block_size, data.decode("latin-1")
    else:
        return block_size, data


class AsyncBgzfWriter:
    """Define an async BGZFWriter object."""

    def __init__(self, async_fileobj, compresslevel=6):
        """Initialize the class."""

        self._handle = async_fileobj
        self._buffer = b""
        self.compresslevel = compresslevel

    def _compress_block(self, block):
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

    async def tell(self):
        """Return a BGZF 64-bit virtual offset."""
        return make_virtual_offset(await self._handle.tell(), len(self._buffer))

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


class AsyncBgzfReader:
    r"""BGZF reader, acts like a read only handle but seek/tell differ.
    """

    def __init__(self, async_fileobj, max_cache=100):
        r"""Initialize the class for reading a BGZF file.
        Argument ``max_cache`` controls the maximum number of BGZF blocks to
        cache in memory. Each can be up to 64kb thus the default of 100 blocks
        could take up to 6MB of RAM. This is important for efficient random
        access, a small value is fine for reading the file in one pass.
        """
        if max_cache < 1:
            raise ValueError("Use max_cache with a minimum of 1")
        # Must open the BGZF file in binary mode, but we may want to
        # treat the contents as either text or binary (unicode or
        # bytes under Python 3)
        self._handle = async_fileobj
        self._text = "b" not in self._handle.mode.lower()
        if self._text:
            self._newline = "\n"
        else:
            self._newline = b"\n"
        self.max_cache = max_cache
        self._buffers = {}
        self._block_start_offset = None
        self._block_raw_length = None

    async def start(self):
        await self._load_block(await self._handle.tell())

    async def _load_block(self, start_offset=None):
        if start_offset is None:
            # If the file is being read sequentially, then _handle.tell()
            # should be pointing at the start of the next block.
            # However, if seek has been used, we can't assume that.
            start_offset = self._block_start_offset + self._block_raw_length
        if start_offset == self._block_start_offset:
            self._within_block_offset = 0
            return
        elif start_offset in self._buffers:
            # Already in cache
            self._buffer, self._block_raw_length = self._buffers[start_offset]
            self._within_block_offset = 0
            self._block_start_offset = start_offset
            return
        # Must hit the disk... first check cache limits,
        while len(self._buffers) >= self.max_cache:
            self._buffers.popitem()
        # Now load the block
        handle = self._handle
        if start_offset is not None:
            handle.seek(start_offset)
        self._block_start_offset = await handle.tell()
        try:
            block_size, self._buffer = await _async_load_bgzf_block(handle, self._text)
        except StopAsyncIteration:
            # EOF
            block_size = 0
            if self._text:
                self._buffer = ""
            else:
                self._buffer = b""
        self._within_block_offset = 0
        self._block_raw_length = block_size
        # Finally save the block in our cache,
        self._buffers[self._block_start_offset] = self._buffer, block_size

    def tell(self):
        """Return a 64-bit unsigned BGZF virtual offset."""
        if 0 < self._within_block_offset and self._within_block_offset == len(
                self._buffer
        ):
            # Special case where we're right at the end of a (non empty) block.
            # For non-maximal blocks could give two possible virtual offsets,
            # but for a maximal block can't use 65536 as the within block
            # offset. Therefore for consistency, use the next block and a
            # within block offset of zero.
            return (self._block_start_offset + self._block_raw_length) << 16
        else:
            # return make_virtual_offset(self._block_start_offset,
            #                           self._within_block_offset)
            return (self._block_start_offset << 16) | self._within_block_offset

    def seek(self, virtual_offset):
        """Seek to a 64-bit unsigned BGZF virtual offset."""
        # Do this inline to avoid a function call,
        # start_offset, within_block = split_virtual_offset(virtual_offset)
        start_offset = virtual_offset >> 16
        within_block = virtual_offset ^ (start_offset << 16)
        if start_offset != self._block_start_offset:
            # Don't need to load the block if already there
            # (this avoids a function call since _load_block would do nothing)
            self._load_block(start_offset)
            if start_offset != self._block_start_offset:
                raise ValueError("start_offset not loaded correctly")
        if within_block > len(self._buffer):
            if not (within_block == 0 and len(self._buffer) == 0):
                raise ValueError(
                    "Within offset %i but block size only %i"
                    % (within_block, len(self._buffer))
                )
        self._within_block_offset = within_block
        # assert virtual_offset == self.tell(), \
        #    "Did seek to %i (%i, %i), but tell says %i (%i, %i)" \
        #    % (virtual_offset, start_offset, within_block,
        #       self.tell(), self._block_start_offset,
        #       self._within_block_offset)
        return virtual_offset

    async def read(self, size=-1):
        """Read method for the BGZF module."""
        if size < 0:
            raise NotImplementedError("Don't be greedy, that could be massive!")

        result = "" if self._text else b""
        while size and self._block_raw_length:
            if self._within_block_offset + size <= len(self._buffer):
                # This may leave us right at the end of a block
                # (lazy loading, don't load the next block unless we have too)
                data = self._buffer[
                       self._within_block_offset: self._within_block_offset + size
                       ]
                self._within_block_offset += size
                if not data:
                    raise ValueError("Must be at least 1 byte")
                result += data
                break
            else:
                data = self._buffer[self._within_block_offset:]
                size -= len(data)
                await self._load_block()  # will reset offsets
                result += data

        return result

    async def readline(self):
        """Read a single line for the BGZF file."""
        result = "" if self._text else b""
        while self._block_raw_length:
            i = self._buffer.find(self._newline, self._within_block_offset)
            # Three cases to consider,
            if i == -1:
                # No newline, need to read in more data
                data = self._buffer[self._within_block_offset:]
                await self._load_block()  # will reset offsets
                result += data
            elif i + 1 == len(self._buffer):
                # Found new line, but right at end of block (SPECIAL)
                data = self._buffer[self._within_block_offset:]
                # Must now load the next block to ensure tell() works
                await self._load_block()  # will reset offsets
                if not data:
                    raise ValueError("Must be at least 1 byte")
                result += data
                break
            else:
                # Found new line, not at end of block (easy case, no IO)
                data = self._buffer[self._within_block_offset: i + 1]
                self._within_block_offset = i + 1
                # assert data.endswith(self._newline)
                result += data
                break

        return result

    async def __anext__(self):
        """Return the next line."""
        line = await self.readline()
        if not line:
            raise StopAsyncIteration
        return line

    def __aiter__(self):
        """Iterate over the lines in the BGZF file."""
        return self

    async def close(self):
        """Close BGZF file."""
        await self._handle.close()
        self._buffer = None
        self._block_start_offset = None
        self._buffers = None

    def seekable(self):
        """Return True indicating the BGZF supports random access."""
        return True

    def isatty(self):
        """Return True if connected to a TTY device."""
        return False

    def fileno(self):
        """Return integer file descriptor."""
        return self._handle.fileno()
