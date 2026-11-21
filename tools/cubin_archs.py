#!/usr/bin/env python3
"""List the CUDA cubins embedded in an executable or object by architecture, without cuobjdump.

    tools/cubin_archs.py bin/basalt        ->  basalt {'sm_121': 45}

Reads each embedded ELF with e_machine EM_CUDA (190) and decodes the SM version from e_flags bits 8-15
(0x78 = sm_120, 0x79 = sm_121). Use it to confirm a GB10 build holds no sm_120 cubins (they would not load).
"""
import collections
import struct
import sys


def scan(path):
    b = open(path, "rb").read()
    c = collections.Counter()
    i = 0
    while True:
        i = b.find(b"\x7fELF\x02\x01", i)
        if i < 0:
            break
        if struct.unpack_from("<H", b, i + 18)[0] == 190:  # EM_CUDA
            flags = struct.unpack_from("<I", b, i + 48)[0]
            c[f"sm_{(flags >> 8) & 0xff}"] += 1
        i += 4
    return dict(c)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        print(p.split("/")[-1], scan(p))
