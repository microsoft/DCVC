// The code is from https://github.com/rygorous/ryg_rans
// The original lisence is below.

// To the extent possible under law, Fabian Giesen has waived all
// copyright and related or neighboring rights to ryg_rans, as
// per the terms of the CC0 license:

//   https://creativecommons.org/publicdomain/zero/1.0

// This work is published from the United States.

// Simple byte-aligned rANS encoder/decoder - public domain - Fabian 'ryg'
// Giesen 2014
//
// Not intended to be "industrial strength"; just meant to illustrate the
// general idea.

#pragma once

#include <stdint.h>

#ifdef assert
    #define RansAssert assert
#else
    #define RansAssert(x)
#endif

// READ ME FIRST:
//
// This is designed like a typical arithmetic coder API, but there's three
// twists you absolutely should be aware of before you start hacking:
//
// 1. You need to encode data in *reverse* - last symbol first. rANS works
//    like a stack: last in, first out.
// 2. Likewise, the encoder outputs bytes *in reverse* - that is, you give
//    it a pointer to the *end* of your buffer (exclusive), and it will
//    slowly move towards the beginning as more bytes are emitted.
// 3. Unlike basically any other entropy coder implementation you might
//    have used, you can interleave data from multiple independent rANS
//    encoders into the same bytestream without any extra signaling;
//    you can also just write some bytes by yourself in the middle if
//    you want to. This is in addition to the usual arithmetic encoder
//    property of being able to switch models on the fly. Writing raw
//    bytes can be useful when you have some data that you know is
//    incompressible, and is cheaper than going through the rANS encode
//    function. Using multiple rANS coders on the same byte stream wastes
//    a few bytes compared to using just one, but execution of two
//    independent encoders can happen in parallel on superscalar and
//    Out-of-Order CPUs, so this can be *much* faster in tight decoding
//    loops.
//
//    This is why all the rANS functions take the write pointer as an
//    argument instead of just storing it in some context struct.

// --------------------------------------------------------------------------

// L ('l' in the paper) is the lower bound of our normalization interval.
// Between this and our byte-aligned emission, we use 31 (not 32!) bits.
// This is done intentionally because exact reciprocals for 31-bit uints
// fit in 32-bit uints: this permits some optimizations during encoding.
constexpr int SCALE_BITS = 16;
constexpr int RANS_SHIFT_BITS = 23;
constexpr uint32_t RANS_BYTE_L = (1u << RANS_SHIFT_BITS);
constexpr int ENC_RENORM_SHIFT_BITS = RANS_SHIFT_BITS - SCALE_BITS + 8;
constexpr int DEC_MASK = ((1u << SCALE_BITS) - 1);

// State for a rANS encoder. Yep, that's all there is to it.
typedef uint32_t RansState;

// Initialize a rANS encoder.
static inline void RansEncInit(RansState& r)
{
    r = RANS_BYTE_L;
}

// Renormalize the encoder. Internal function.
static inline void RansEncRenorm(RansState& x, uint8_t*& ptr, uint32_t freq)
{
    const uint32_t x_max = freq << ENC_RENORM_SHIFT_BITS;
    while (x >= x_max) {
        *(--ptr) = static_cast<uint8_t>(x & 0xff);
        x >>= 8;
    }
}

// Encodes a single symbol with range start "start" and frequency "freq".
// All frequencies are assumed to sum to "1 << scale_bits", and the
// resulting bytes get written to ptr (which is updated).
//
// NOTE: With rANS, you need to encode symbols in *reverse order*, i.e. from
// beginning to end! Likewise, the output bytestream is written *backwards*:
// ptr starts pointing at the end of the output buffer and keeps decrementing.
static inline void RansEncPut(RansState& r, uint8_t*& ptr, uint32_t start, uint32_t freq)
{
    // renormalize
    RansEncRenorm(r, ptr, freq);

    // x = C(s,x)
    r = ((r / freq) << SCALE_BITS) + (r % freq) + start;
}

// Flushes the rANS encoder.
static inline void RansEncFlush(const RansState& r, uint8_t*& ptr)
{
    ptr -= 4;
    ptr[0] = (uint8_t)(r >> 0);
    ptr[1] = (uint8_t)(r >> 8);
    ptr[2] = (uint8_t)(r >> 16);
    ptr[3] = (uint8_t)(r >> 24);
}

// Initializes a rANS decoder.
// Unlike the encoder, the decoder works forwards as you'd expect.
static inline void RansDecInit(RansState& r, uint8_t*& ptr)
{
    r = (*ptr++) << 0;
    r |= (*ptr++) << 8;
    r |= (*ptr++) << 16;
    r |= (*ptr++) << 24;
}

// Returns the current cumulative frequency (map it to a symbol yourself!)
static inline uint32_t RansDecGet(RansState& r)
{
    return r & DEC_MASK;
}

// Advances in the bit stream by "popping" a single symbol with range start
// "start" and frequency "freq". All frequencies are assumed to sum to "1 <<
// scale_bits", and the resulting bytes get written to ptr (which is updated).
static inline void RansDecAdvance(RansState& r, uint8_t*& ptr, uint32_t start, uint32_t freq)
{

    // s, x = D(x)
    r = freq * (r >> SCALE_BITS) + (r & DEC_MASK) - start;

    // renormalize
    while (r < RANS_BYTE_L) {
        r = (r << 8) | *ptr++;
    }
}
