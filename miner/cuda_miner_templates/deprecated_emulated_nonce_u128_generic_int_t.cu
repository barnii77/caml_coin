/*
 * SHA256 copied from: https://github.com/mochimodev/cuda-hashing-algos/blob/master/sha256.cu
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

// currently, arrays are being used for 128-bit and 256-bit integers
// TODO use __int128 and __int256 when available

/**************************** CONSTANTS *****************************/
typedef unsigned char BYTE;
typedef unsigned int WORD;
typedef unsigned long long LONG;

/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32  // SHA256 outputs a 32 byte digest
#define NONCE_SIZE 16  // 16 bytes for nonce

#ifndef MAX_VALID_HASH_LOCATION
#define MAX_VALID_HASH_LOCATION 1  // 0 = global mem, 1 = shared mem, >=2 = registers
#endif

#ifndef INT_T
#define INT_T WORD
#endif

#ifdef __int128
typedef unsigned __int128 uint128_t;
#else
#define unsigned long long[2] uint128_t
#define EMULATE_INT128
#endif

/**************************** DATA TYPES ****************************/

typedef struct {
    BYTE data[64];
    WORD datalen;
    unsigned long long bitlen;
    WORD state[8];
} CUDA_SHA256_CTX;

/****************************** MACROS ******************************/
#ifndef ROTLEFT
#define ROTLEFT(a, b) (((a) << (b)) | ((a) >> (32-(b))))
#endif

#define ROTRIGHT(a, b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

/**************************** VARIABLES *****************************/
__constant__ WORD k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/*********************** FUNCTION DEFINITIONS ***********************/
__device__  __forceinline__ void cuda_sha256_transform(CUDA_SHA256_CTX *ctx, const BYTE data[]) {
    WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

#pragma unroll 16
    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
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

__device__ void cuda_sha256_update(CUDA_SHA256_CTX *ctx, const BYTE data[], size_t len) {
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
        while (i < 56)
            ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64)
            ctx->data[i++] = 0x00;
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

/// checks if a 256 bit little endian integer is less than or equal to another
__forceinline__ __device__ bool cuda_u256_lte(LONG *a, LONG *b) {
#pragma unroll (32 / sizeof(LONG))
    for (int i = 32 / sizeof(LONG) - 1; i >= 0; i--) {
        LONG a_ = a[i], b_ = b[i];
        if (a_ < b_) return true;
        else if (a_ > b_) return false;
    }
    return true;
}

/// adds two 128 bit little endian integers in place. for inplace add, a and c must be same pointer
__forceinline__ __device__ void cuda_u128_add(LONG a[2], LONG b[2], LONG c[2]) {
    BYTE carry = 0;
#pragma unroll (16 / sizeof(LONG))
    for (int i = 0; i < 16 / sizeof(LONG); i++) {
        LONG a_ = a[i], b_ = b[i];
        LONG sum = a_ + b_ + carry;
        carry = (sum < a_) | (sum < b_);
        c[i] = sum;
    }
}

__forceinline__ __device__ bool cuda_u128_eq0(LONG *a) {
    for (int i = 0; i < 16 / sizeof(LONG); i++) {
        if (a[i] != 0) return false;
    }
    return true;
}

// all parameters are little endian
// *nonce_out should be 0 when called, because *nonce_out == 0 will be interpreted as no nonce having been found yet
extern "C" __global__
void mine_sha256(BYTE const_in[SHA256_BLOCK_SIZE], BYTE nonce_out[NONCE_SIZE], WORD nonce_step_size,
                 WORD n_batch_device, BYTE max_valid_hash[SHA256_BLOCK_SIZE], BYTE init_nonce[NONCE_SIZE]) {
    WORD thread = blockIdx.x * blockDim.x + threadIdx.x;

#if MAX_VALID_HASH_LOCATION == 0  // leave it in global mem
    INT_T *_max_valid_hash = (INT_T *) max_valid_hash;
#elif MAX_VALID_HASH_LOCATION == 1  // put it into shared mem
    __shared__ INT_T _max_valid_hash[SHA256_BLOCK_SIZE / sizeof(INT_T)];
    if (threadIdx.x < SHA256_BLOCK_SIZE / sizeof(INT_T)) {
        // populate _max_valid_hash
        _max_valid_hash[threadIdx.x] = ((INT_T *) max_valid_hash)[threadIdx.x];
    }
    __syncthreads();
#else  // put it into thread-local registers
    INT_T _max_valid_hash[SHA256_BLOCK_SIZE / sizeof(INT_T)];
    // populate _max_valid_hash
#pragma unroll (SHA256_BLOCK_SIZE / sizeof(INT_T))
    for (int i = 0; i < SHA256_BLOCK_SIZE / sizeof(INT_T); i++) {
        _max_valid_hash[i] = ((INT_T *) max_valid_hash)[i];
    }
#endif

    if (thread >= n_batch_device) {
        return;
    }

    INT_T nonce[NONCE_SIZE / sizeof(INT_T)];
    // populate nonce
#pragma unroll (NONCE_SIZE / sizeof(INT_T))
    for (int i = 0; i < NONCE_SIZE / sizeof(INT_T); i++) {
        nonce[i] = ((INT_T *) init_nonce)[i];
    }
    BYTE carry = 0;
    for (int i = 0; i < sizeof(WORD) / sizeof(INT_T); i++) {
        nonce[i] += thread + carry;
        carry = nonce[i] < thread;
        thread >>= sizeof(INT_T) * 8;
    }

    thread = -1;  // invalidate thread

    INT_T _nonce_step_size[NONCE_SIZE / sizeof(INT_T)];
    // populate nonce_step_size
#pragma unroll (NONCE_SIZE / sizeof(INT_T))
    for (int i = 0; i < NONCE_SIZE / sizeof(INT_T); i++) {
        _nonce_step_size[i] = 0;
    }
    _nonce_step_size[0] = nonce_step_size;

    BYTE in[NONCE_SIZE + SHA256_BLOCK_SIZE];  // nonce + const_in = 384 bytes
    BYTE out[SHA256_BLOCK_SIZE];
    CUDA_SHA256_CTX ctx;
    for (; cuda_u128_eq0(nonce_out); cuda_u128_add(nonce, _nonce_step_size, nonce)) {
        // write nonce to first bytes of in
        for (int i = 0; i < sizeof(nonce) / sizeof(INT_T); i++) {
            ((INT_T *) in)[i] = nonce[i];
        }
        // sha256 with little endian output
        cuda_sha256_init(&ctx);
        cuda_sha256_update(&ctx, in, sizeof(in));
        cuda_sha256_final(&ctx, out);
        // check if nonce is valid
        if (cuda_u256_lte((INT_T *) out, _max_valid_hash)) {
            // write nonce to nonce_out
            for (int i = 0; i < NONCE_SIZE / sizeof(INT_T); i++) {
                ((INT_T *) nonce_out)[i] = nonce[i];
            }
        }
    }
}