/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef OPTEST_JIT_SHA256_H
#define OPTEST_JIT_SHA256_H

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace CatlassKernel {

namespace detail {

inline uint32_t rotr(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32 - n));
}
inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (~x & z);
}
inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (x & z) ^ (y & z);
}
inline uint32_t sigma0(uint32_t x)
{
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}
inline uint32_t sigma1(uint32_t x)
{
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}
inline uint32_t gamma0(uint32_t x)
{
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}
inline uint32_t gamma1(uint32_t x)
{
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

constexpr uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

} // namespace detail

struct Sha256 {
    std::array<uint32_t, 8> h = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    void update(const uint8_t* data, size_t len)
    {
        for (size_t i = 0; i < len; ++i) {
            buf_[bufLen_++] = data[i];
            if (bufLen_ == 64) {
                processBlock();
                totalBits_ += 512;
                bufLen_ = 0;
            }
        }
    }

    void update(const std::string& s)
    {
        update(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }

    std::array<uint8_t, 32> finalize()
    {
        totalBits_ += bufLen_ * 8;
        buf_[bufLen_++] = 0x80;
        if (bufLen_ > 56) {
            while (bufLen_ < 64)
                buf_[bufLen_++] = 0;
            processBlock();
            bufLen_ = 0;
        }
        while (bufLen_ < 56)
            buf_[bufLen_++] = 0;
        for (int i = 7; i >= 0; --i) {
            buf_[56 + i] = static_cast<uint8_t>(totalBits_ >> ((7 - i) * 8));
        }
        processBlock();

        std::array<uint8_t, 32> digest;
        for (int i = 0; i < 8; ++i) {
            for (int j = 3; j >= 0; --j) {
                digest[i * 4 + (3 - j)] = static_cast<uint8_t>(h[i] >> (j * 8));
            }
        }
        return digest;
    }

    static std::string hex(const std::array<uint8_t, 32>& digest)
    {
        const char* hex = "0123456789abcdef";
        std::string out(64, '\0');
        for (size_t i = 0; i < 32; ++i) {
            out[i * 2] = hex[digest[i] >> 4];
            out[i * 2 + 1] = hex[digest[i] & 0xf];
        }
        return out;
    }

    static std::string hash(const std::string& input)
    {
        Sha256 ctx;
        ctx.update(input);
        return hex(ctx.finalize());
    }

private:
    void processBlock()
    {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(buf_[i * 4]) << 24) | (static_cast<uint32_t>(buf_[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(buf_[i * 4 + 2]) << 8) | (static_cast<uint32_t>(buf_[i * 4 + 3]));
        }
        for (int i = 16; i < 64; ++i) {
            w[i] = detail::gamma1(w[i - 2]) + w[i - 7] + detail::gamma0(w[i - 15]) + w[i - 16];
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

        for (int i = 0; i < 64; ++i) {
            uint32_t t1 = hh + detail::sigma1(e) + detail::ch(e, f, g) + detail::k[i] + w[i];
            uint32_t t2 = detail::sigma0(a) + detail::maj(a, b, c);
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    uint8_t buf_[64]{};
    size_t bufLen_ = 0;
    uint64_t totalBits_ = 0;
};

} // namespace CatlassKernel
#endif // OPTEST_JIT_SHA256_H