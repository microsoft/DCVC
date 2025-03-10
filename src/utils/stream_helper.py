# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import struct
from pathlib import Path


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values)


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return 0
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values)


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def write_ushorts(fd, values, fmt=">{:d}H"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 2


def read_ushorts(fd, n, fmt=">{:d}H"):
    sz = struct.calcsize("H")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_uint_adaptive(f, a):
    if a < (1 << 7):
        a0 = (a >> 0) & 0xff
        a0 = a0 | (0x00 << 7)
        write_uchars(f, (a0,))
        return 1

    if a < (1 << 14):
        a0 = (a >> 0) & 0xff
        a1 = (a >> 8) & 0xff
        a1 = a1 | (0x02 << 6)
        write_uchars(f, (a1, a0))
        return 2

    assert a < (1 << 30)
    a0 = (a >> 0) & 0xff
    a1 = (a >> 8) & 0xff
    a2 = (a >> 16) & 0xff
    a3 = (a >> 24) & 0xff
    a3 = a3 | (0x03 << 6)
    write_uchars(f, (a3, a2, a1, a0))
    return 4


def read_uint_adaptive(f):
    a3 = read_uchars(f, 1)[0]
    if (a3 >> 7) == 0:
        return a3

    a2 = read_uchars(f, 1)[0]

    if (a3 >> 6) == 0x02:
        a3 = a3 & 0x3f
        return (a3 << 8) + a2
    a3 = a3 & 0x3f
    a1 = read_uchars(f, 1)[0]
    a0 = read_uchars(f, 1)[0]
    return (a3 << 24) + (a2 << 16) + (a1 << 8) + a0


class NalType(enum.IntEnum):
    NAL_SPS = 0
    NAL_I = 1
    NAL_P = 2


class SPSHelper():
    def __init__(self):
        super().__init__()
        self.spss = []

    def get_sps_id(self, target_sps):
        min_id = -1
        for sps in self.spss:
            if sps['height'] == target_sps['height'] and sps['width'] == target_sps['width'] and \
                    sps['use_ada_i'] == target_sps['use_ada_i'] and \
                    sps['ec_part'] == target_sps['ec_part']:
                return sps['sps_id'], False
            if sps['sps_id'] > min_id:
                min_id = sps['sps_id']
        assert min_id < 15
        sps = target_sps.copy()
        sps['sps_id'] = min_id + 1
        self.spss.append(sps)
        return sps['sps_id'], True

    def add_sps_by_id(self, sps):
        for i in range(len(self.spss)):
            if self.spss[i]['sps_id'] == sps['sps_id']:
                self.spss[i] = sps.copy()
                return
        self.spss.append(sps.copy())

    def get_sps_by_id(self, sps_id):
        for sps in self.spss:
            if sps['sps_id'] == sps_id:
                return sps
        return None


def write_sps(f, sps):
    # nal_type(4), sps_id(4)
    # height (variable)
    # width (vairable)
    # 0(6), ec_part(1) use_ada_i(1)
    assert sps['sps_id'] < 16
    assert sps['use_ada_i'] < 2
    written = 0
    flag = int((NalType.NAL_SPS << 4) + sps['sps_id'])
    written += write_uchars(f, (flag,))
    written += write_uint_adaptive(f, sps['height'])
    written += write_uint_adaptive(f, sps['width'])
    flag = (sps['ec_part'] << 2) + sps['use_ada_i']
    written += write_uchars(f, (flag,))
    return written


def read_header(f):
    header = {}
    flag = read_uchars(f, 1)[0]
    nal_type = flag >> 4
    header['nal_type'] = NalType(nal_type)
    if nal_type < 3:
        header['sps_id'] = flag & 0x0f
        return header

    frame_num_minus1 = flag & 0x0f
    frame_num = frame_num_minus1 + 1
    header['frame_num'] = frame_num
    sps_ids = []
    for _ in range(0, frame_num, 2):
        flag = read_uchars(f, 1)[0]
        sps_ids.append(flag >> 4)
        sps_ids.append(flag & 0x0f)
    sps_ids = sps_ids[:frame_num]
    header['sps_ids'] = sps_ids
    return header


def read_sps_remaining(f, sps_id):
    sps = {}
    sps['sps_id'] = sps_id
    sps['height'] = read_uint_adaptive(f)
    sps['width'] = read_uint_adaptive(f)
    flag = read_uchars(f, 1)[0]
    sps['ec_part'] = (flag >> 2) & 0x01
    sps['use_ada_i'] = flag & 0x01
    return sps


def write_ip(f, is_i_frame, sps_id, qp, bit_stream):
    written = 0
    flag = (int(NalType.NAL_I if is_i_frame else NalType.NAL_P) << 4) + sps_id
    written += write_uchars(f, (flag,))
    assert qp < 256 and qp >= 0
    flag = qp
    written += write_uchars(f, (flag,))
    # we write all the streams in the same file, thus, we need to write the per-frame length
    # if packed independently, we do not need to write it
    written += write_uint_adaptive(f, len(bit_stream))
    written += write_bytes(f, bit_stream)
    return written


def read_ip_remaining(f):
    flag = read_uchars(f, 1)[0]
    qp = flag
    stream_length = read_uint_adaptive(f)
    bit_stream = read_bytes(f, stream_length)
    return qp, bit_stream
