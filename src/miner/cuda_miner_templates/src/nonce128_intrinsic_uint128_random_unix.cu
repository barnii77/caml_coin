// NVRTC-OPTIONS: --device-int128
/*
 * Copied from:
 * https://github.com/mochimodev/cuda-hashing-algos/blob/master/sha256.cu
 *
 * sha256.cu Implementation of SHA256 Hashing
 *
 * Date: 12 June 2019
 * Revision: 1
 * *
 * Based on the public domain Reference Implementation in C, by
 * Brad Conte, original code here:
 *
 * https://github.com/B-Con/crypto-algorithms
 *
 * This file is released into the Public Domain.
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

typedef unsigned char BYTE;
typedef unsigned int WORD;
typedef unsigned long long LONG;
typedef unsigned __int128 uint128_t;

#define SHA256_BLOCK_SIZE 32  // SHA256 outputs a 32 byte digest

#ifndef MAX_VALID_HASH_LOCATION
#define MAX_VALID_HASH_LOCATION \
  2  // 0 = global mem, 1 = shared mem, >=2 = registers
#endif

typedef struct {
    BYTE data[64];
    WORD datalen;
    LONG bitlen;
    WORD state[8];
} CUDA_SHA256_CTX;

typedef struct __align__(8 * sizeof(WORD)) uint256_t {
    WORD _0;
    WORD _1;
    WORD _2;
    WORD _3;
    WORD _4;
    WORD _5;
    WORD _6;
    WORD _7;
} uint256_t;

#ifndef ROTLEFT
#define ROTLEFT(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
#endif

#define ROTRIGHT(a, b) (((a) >> (b)) | ((a) << (32 - (b))))

#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x, 2) ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22))
#define EP1(x) (ROTRIGHT(x, 6) ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25))
#define SIG0(x) (ROTRIGHT(x, 7) ^ ROTRIGHT(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ ((x) >> 10))

__constant__ WORD k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

__device__ __forceinline__ void cuda_sha256_transform(CUDA_SHA256_CTX *ctx,
                                                      const BYTE data[]) {
    WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

#pragma unroll 16
    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) |
               (data[j + 3]);
#pragma unroll 48
    for (; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

#pragma unroll 64
    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + k[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

__device__ void cuda_sha256_init(CUDA_SHA256_CTX *ctx) {
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
}

__device__ void cuda_sha256_update(CUDA_SHA256_CTX *ctx, const BYTE data[],
                                   size_t len) {
    WORD i;

    for (i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64) {
            cuda_sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

__device__ void cuda_sha256_final(CUDA_SHA256_CTX *ctx, BYTE hash[]) {
    WORD i;

    i = ctx->datalen;

    // Pad whatever data is left in the buffer.
    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56) ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64) ctx->data[i++] = 0x00;
        cuda_sha256_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56);
    }

    // Append to the padding the total message's length in bits and transform.
    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = ctx->bitlen;
    ctx->data[62] = ctx->bitlen >> 8;
    ctx->data[61] = ctx->bitlen >> 16;
    ctx->data[60] = ctx->bitlen >> 24;
    ctx->data[59] = ctx->bitlen >> 32;
    ctx->data[58] = ctx->bitlen >> 40;
    ctx->data[57] = ctx->bitlen >> 48;
    ctx->data[56] = ctx->bitlen >> 56;
    cuda_sha256_transform(ctx, ctx->data);

    // normally sha256 outputs big endian, but here we output little endian
#pragma unroll 4
    for (i = 0; i < 4; ++i) {
        hash[i] = ctx->state[7] >> i * 8;
        hash[i + 4] = ctx->state[6] >> i * 8;
        hash[i + 8] = ctx->state[5] >> i * 8;
        hash[i + 12] = ctx->state[4] >> i * 8;
        hash[i + 16] = ctx->state[3] >> i * 8;
        hash[i + 20] = ctx->state[2] >> i * 8;
        hash[i + 24] = ctx->state[1] >> i * 8;
        hash[i + 28] = ctx->state[0] >> i * 8;
    }
}

/// checks if a 256 bit little endian integer is less than another
__forceinline__ __device__ bool cuda_u256_lte(uint256_t const a, uint256_t const b) {
#pragma unroll sizeof(uint256_t) / sizeof(WORD)
    for (int i = sizeof(uint256_t) / sizeof(WORD) - 1; i >= 0; i--) {
        WORD a_ = ((WORD *) &a)[i], b_ = ((WORD *) &b)[i];
        // NOTE: first checking a_ > b_ is more efficient because most nonces are
        // invalid and therefore if a_ > b_ is the most common exit path (very often
        // in the first loop iteration), thus avoiding almost all computations
        if (a_ > b_) return false;
        if (a_<b_) return true;
    }
    return true;
}

/// mines until any(nonce_out[i] != 0 for i in range(0, sizeof(uint128_t)))
extern "C" __global__ void mine_sha256(BYTE const const_in[SHA256_BLOCK_SIZE],
                                       BYTE nonce_out[sizeof(uint128_t)],
                                       WORD const nonce_step_size,
                                       WORD const n_batch_device,
                                       BYTE const max_valid_hash[SHA256_BLOCK_SIZE],
                                       BYTE const init_nonce[sizeof(uint128_t)]) {
    curandStateXORWOW_t state;
    WORD thread = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(clock64(), thread, 0, &state);

#if MAX_VALID_HASH_LOCATION == 0  // leave it in global mem
    uint256_t _max_valid_hash = *((uint256_t *)max_valid_hash);
#elif MAX_VALID_HASH_LOCATION == 1  // put it into shared mem
    __shared__ uint256_t _max_valid_hash;
    if (threadIdx.x < SHA256_BLOCK_SIZE / sizeof(WORD)) {
        // populate _max_valid_hash
        ((WORD *)&_max_valid_hash)[threadIdx.x] =
            ((WORD *)max_valid_hash)[threadIdx.x];
    }
    __syncthreads();
#else                               // put it into thread-local registers
    uint256_t _max_valid_hash;
    // populate _max_valid_hash
#pragma unroll 32 / sizeof(WORD)
    for (int i = 0; i < SHA256_BLOCK_SIZE / sizeof(WORD); i++) {
        ((WORD *) &_max_valid_hash)[i] = ((WORD *) max_valid_hash)[i];
    }
#endif

    if (thread >= n_batch_device) {
        return;
    }

    uint128_t nonce = *((uint128_t *) init_nonce) + thread;
    BYTE in[sizeof(uint128_t) +
            SHA256_BLOCK_SIZE];  // nonce + const_in = 384 bytes
#pragma unroll 32 / sizeof(WORD)
    for (int i = 0; i < SHA256_BLOCK_SIZE / sizeof(WORD); i++) {
        ((WORD *) in)[i + sizeof(uint128_t) / sizeof(WORD)] = ((WORD *) const_in)[i];
    }

    uint256_t out;
    CUDA_SHA256_CTX ctx;
    uint128_t *_nonce_out = (uint128_t *) nonce_out;
    bool success = false;
    while (1) {
        // write nonce to first bytes of in
        *((uint128_t *) in) = nonce;
        // sha256 with little endian output
        cuda_sha256_init(&ctx);
        cuda_sha256_update(&ctx, in, sizeof(in));
        cuda_sha256_final(&ctx, (BYTE *) &out);
        // check if nonce is valid
        if (cuda_u256_lte(out, _max_valid_hash)) {
            success = atomicCAS((WORD *) nonce_out, 0, 1) == 0;
            break;
        }
        if (*_nonce_out != 0) break;
        for (int i = 0; i < sizeof(uint128_t) / sizeof(WORD); i++) {
            ((WORD *) &nonce)[i] = curand(&state);
        }
    }

    // without syncthreads, if another thread was just doing the atomicCAS and I
    // am currently writing into nonce_out, there is a tiny chance there could be
    // interference. let's not risk this.
    __syncthreads();
    if (success) *((uint128_t *)nonce_out) = nonce;
}
