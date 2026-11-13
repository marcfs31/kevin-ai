/* lm_bf16.h - bf16 and f16 widening. bf16 widening is ported from kimi-k3-in-c
 * k3_bf16_to_f32 in src/io/k3_st.h (Apache-2.0); the f16 conversion is original.
 *
 * bf16 IS the top 16 bits of an f32, so widening is a pure left shift: no table, no
 * rounding, no exponent rebias. f16 has a different exponent width and needs the full
 * conversion, including subnormals, infinities and NaN. */
#ifndef LM_BF16_H
#define LM_BF16_H

#include <stdint.h>
#include <string.h>

static inline float lm_bf16_to_f32(uint16_t h)
{
    const uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* Round to nearest even; NaN stays NaN (quiet bit forced so truncation cannot yield Inf). */
static inline uint16_t lm_f32_to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static inline float lm_f16_to_f32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t u;
    if (exp == 0) {
        if (man == 0) {
            u = sign;                                   /* signed zero */
        } else {                                        /* subnormal: normalise */
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; exp--; }
            man &= 0x3ffu;
            u = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7f800000u | (man << 13);           /* Inf or NaN */
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}

#endif /* LM_BF16_H */
