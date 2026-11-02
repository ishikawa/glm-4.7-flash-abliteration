#!/usr/bin/env python3
"""Parse GGUF header (KV + tensor infos) from a possibly-truncated file."""

import json
import struct
import sys

GGUF_MAGIC = b"GGUF"

# gguf value types
(
    T_UINT8,
    T_INT8,
    T_UINT16,
    T_INT16,
    T_UINT32,
    T_INT32,
    T_FLOAT32,
    T_BOOL,
    T_STRING,
    T_ARRAY,
    T_UINT64,
    T_INT64,
    T_FLOAT64,
) = range(13)

SCALAR_FMT = {
    T_UINT8: "<B",
    T_INT8: "<b",
    T_UINT16: "<H",
    T_INT16: "<h",
    T_UINT32: "<I",
    T_INT32: "<i",
    T_FLOAT32: "<f",
    T_BOOL: "<?",
    T_UINT64: "<Q",
    T_INT64: "<q",
    T_FLOAT64: "<d",
}

# ggml tensor types: name -> (block_size_elems, type_size_bytes)
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    31: ("Q4_0_4_4", 32, 18),
    32: ("Q4_0_4_8", 32, 18),
    33: ("Q4_0_8_8", 32, 18),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    36: ("MXFP4", 32, 17),
}


class Reader:
    def __init__(self, buf=None, fileobj=None):
        self.buf = buf
        self.fileobj = fileobj
        self.pos = 0

    def read(self, n):
        if self.fileobj is None:
            if self.pos + n > len(self.buf):
                raise EOFError(
                    f"need {n} bytes at pos {self.pos}, have {len(self.buf)}"
                )
            b = self.buf[self.pos : self.pos + n]
        else:
            b = self.fileobj.read(n)
            if len(b) != n:
                raise EOFError(f"need {n} bytes at pos {self.pos}, got {len(b)}")
        self.pos += n
        return b

    def skip(self, n):
        if self.fileobj is None:
            self.read(n)
        else:
            self.fileobj.seek(n, 1)
            self.pos += n

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        n = self.u64()
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, vtype):
        if vtype in SCALAR_FMT:
            fmt = SCALAR_FMT[vtype]
            return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]
        if vtype == T_STRING:
            return self.string()
        if vtype == T_ARRAY:
            et = self.u32()
            n = self.u64()
            # avoid huge arrays (tokenizer): summarize
            if et == T_STRING:
                vals = [self.string() for _ in range(min(n, 5))]
                # skip rest
                for _ in range(n - min(n, 5)):
                    self.string()
                return {"_array": "string", "n": n, "head": vals}
            fmt = SCALAR_FMT[et]
            sz = struct.calcsize(fmt)
            head = [struct.unpack(fmt, self.read(sz))[0] for _ in range(min(n, 8))]
            rest = n - min(n, 8)
            self.skip(rest * sz)
            return {"_array": et, "n": n, "head": head}
        raise ValueError(f"unknown value type {vtype}")


def parse(path):
    with open(path, "rb") as f:
        r = Reader(fileobj=f)
        assert r.read(4) == GGUF_MAGIC, "bad magic"
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        kv = {}
        for _ in range(n_kv):
            k = r.string()
            vt = r.u32()
            kv[k] = r.value(vt)
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ttype = r.u32()
            offset = r.u64()
            nelem = 1
            for d in dims:
                nelem *= d
            tname, bsz, tsz = GGML_TYPES.get(ttype, (f"UNK{ttype}", 1, 0))
            nbytes = (nelem // bsz) * tsz if bsz else 0
            tensors.append(
                {
                    "name": name,
                    "dims": dims,
                    "type": tname,
                    "offset": offset,
                    "nelem": nelem,
                    "nbytes": nbytes,
                }
            )
        align = kv.get("general.alignment", 32)
        data_start = (r.pos + align - 1) // align * align
    return {
        "version": version,
        "n_tensors": n_tensors,
        "n_kv": n_kv,
        "kv": kv,
        "tensors": tensors,
        "header_end": r.pos,
        "data_start": data_start,
    }


if __name__ == "__main__":
    out = parse(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "summary"
    if mode == "kv":
        print(json.dumps({k: v for k, v in out["kv"].items()}, indent=1, default=str))
    elif mode == "tensors":
        print(json.dumps(out["tensors"], indent=0))
    else:
        print(
            f"version={out['version']} n_tensors={out['n_tensors']} n_kv={out['n_kv']} "
            f"header_end={out['header_end']} data_start={out['data_start']}"
        )
