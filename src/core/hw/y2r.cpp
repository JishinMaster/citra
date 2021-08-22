// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include "common/assert.h"
#include "common/color.h"
#include "common/common_types.h"
#include "common/vector_math.h"
#include "core/core.h"
#include "core/hle/service/y2r_u.h"
#include "core/hw/y2r.h"
#include "core/memory.h"

#ifdef __SSE4_1__
#include <immintrin.h>
#endif

namespace HW::Y2R {

using namespace Service::Y2R;

static const std::size_t MAX_TILES = 1024 / 8;
static const std::size_t TILE_SIZE = 8 * 8;
using ImageTile = std::array<u32, TILE_SIZE>;

// The following section uses SSE 4.1 as it is available on most X86 CPUs
// and is easy to port to ARM NEON
#ifdef __SSE4_1__

// This function uses SIMD intrinsics to speed up the conversion
// The floating point method is simpler in this case
static void ConvertYUVToRGB_YUV422_420(InputFormat input_format, const u8* __restrict__ input_Y,
                                       const u8* __restrict__ input_U,
                                       const u8* __restrict__ input_V, ImageTile output[],
                                       unsigned int width, unsigned int height,
                                       const CoefficientSet& coefficients) {

    // Floating point coefficients are used. The function argument is kept to match the
    // non vectorised version of this function
    // R = Y + 1.402 * (V-128)
    // G = Y -0.344*(U-128) - 0.714*(V-128)
    // B = Y +  1.772*(U-128)
    __m128 c0, c1, c2, c3;
    c0 = _mm_set1_ps(1.402f);
    c1 = _mm_set1_ps(-0.344f);
    c2 = _mm_set1_ps(-0.714f);
    c3 = _mm_set1_ps(1.772f);

    __m128 zero_float = _mm_setzero_ps();
    __m128i zero_int = _mm_setzero_si128();
    __m128i half_byte_int = _mm_set1_epi16(128);
    __m128 max_byte_float = _mm_set1_ps(255.0f);
    __m128i uv_mask_lo = _mm_set_epi8(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);

    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; x += 16) {
            int y_offset, uv_offset;
            switch (input_format) {
            case InputFormat::YUV422_Indiv8:
            case InputFormat::YUV422_Indiv16:
                y_offset = y * width + x;
                uv_offset = (y * width + x) / 2;
                break;
            case InputFormat::YUV420_Indiv8:
            case InputFormat::YUV420_Indiv16:
                y_offset = y * width + x;
                uv_offset = ((y / 2) * width + x) / 2;
                break;
            default:
                y_offset = 0;
                uv_offset = 0;
                break;
            }

            // No assumption here on the alignement of the pointers,
            // sticking with the unaligned load/stores
            // We load unsigned 8bit integers and need to convert them to 32bit floats
            // They are first converted to signed 16bit integers, then to signed 32bit integers
            __m128i Y_vec = _mm_loadu_si128(reinterpret_cast<__m128i const*>(input_Y + y_offset));
            __m128i Y_vec_lo = _mm_cvtepu8_epi16(Y_vec);
            __m128i Y_vec_hi = _mm_unpackhi_epi8(Y_vec, zero_int);

            __m128i U_vec = _mm_loadu_si128(reinterpret_cast<__m128i const*>(input_U + uv_offset));
            // The uv_mask_lo is used to double the values in the vector
            U_vec = _mm_shuffle_epi8(U_vec, uv_mask_lo);
            __m128i U_vec_lo = _mm_cvtepu8_epi16(U_vec);
            __m128i U_vec_hi = _mm_unpackhi_epi8(U_vec, zero_int);

            __m128i V_vec = _mm_loadu_si128(reinterpret_cast<__m128i const*>(input_V + uv_offset));
            V_vec = _mm_shuffle_epi8(V_vec, uv_mask_lo);
            __m128i V_vec_lo = _mm_cvtepu8_epi16(V_vec);
            __m128i V_vec_hi = _mm_unpackhi_epi8(V_vec, zero_int);

            // subtracting 128 to U and V when they are still in 16bit form,
            // which should be faster than doing it on 32bit floats
            U_vec_lo = _mm_sub_epi16(U_vec_lo, half_byte_int);
            U_vec_hi = _mm_sub_epi16(U_vec_hi, half_byte_int);

            V_vec_lo = _mm_sub_epi16(V_vec_lo, half_byte_int);
            V_vec_hi = _mm_sub_epi16(V_vec_hi, half_byte_int);

            __m128 Y0to3 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(Y_vec_lo));
            __m128 Y4to7 =
                _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(Y_vec_lo, zero_int)));
            __m128 Y8to11 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(Y_vec_hi));
            __m128 Y12to15 =
                _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(Y_vec_hi, zero_int)));

            __m128 U0to3 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(U_vec_lo));
            __m128 U4to7 =
                _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(U_vec_lo, zero_int)));
            __m128 U8to11 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(U_vec_hi));
            __m128 U12to15 =
                _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(U_vec_hi, zero_int)));

            __m128 V0to3 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(V_vec_lo));
            __m128 V4to7 =
                _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(V_vec_lo, zero_int)));
            __m128 V8to11 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(V_vec_hi));
            __m128 V12to15 =
                _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(V_vec_hi, zero_int)));

            __m128 r0to3 = _mm_add_ps(_mm_mul_ps(c0, V0to3), Y0to3);
            __m128 r4to7 = _mm_add_ps(_mm_mul_ps(c0, V4to7), Y4to7);
            __m128 r8to11 = _mm_add_ps(_mm_mul_ps(c0, V8to11), Y8to11);
            __m128 r12to15 = _mm_add_ps(_mm_mul_ps(c0, V12to15), Y12to15);

            __m128 g0to3 = _mm_add_ps(Y0to3, _mm_mul_ps(c1, U0to3));
            __m128 g4to7 = _mm_add_ps(Y4to7, _mm_mul_ps(c1, U4to7));
            __m128 g8to11 = _mm_add_ps(Y8to11, _mm_mul_ps(c1, U8to11));
            __m128 g12to15 = _mm_add_ps(Y12to15, _mm_mul_ps(c1, U12to15));
            g0to3 = _mm_add_ps(g0to3, _mm_mul_ps(c2, V0to3));
            g4to7 = _mm_add_ps(g4to7, _mm_mul_ps(c2, V4to7));
            g8to11 = _mm_add_ps(g8to11, _mm_mul_ps(c2, V8to11));
            g12to15 = _mm_add_ps(g12to15, _mm_mul_ps(c2, V12to15));

            __m128 b0to3 = _mm_add_ps(_mm_mul_ps(c3, U0to3), Y0to3);
            __m128 b4to7 = _mm_add_ps(_mm_mul_ps(c3, U4to7), Y4to7);
            __m128 b8to11 = _mm_add_ps(_mm_mul_ps(c3, U8to11), Y8to11);
            __m128 b12to15 = _mm_add_ps(_mm_mul_ps(c3, U12to15), Y12to15);

            // clamp the values between 0.0f and 255.0f
            r0to3 = _mm_min_ps(r0to3, max_byte_float);
            r4to7 = _mm_min_ps(r4to7, max_byte_float);
            r8to11 = _mm_min_ps(r8to11, max_byte_float);
            r12to15 = _mm_min_ps(r12to15, max_byte_float);

            g0to3 = _mm_min_ps(g0to3, max_byte_float);
            g4to7 = _mm_min_ps(g4to7, max_byte_float);
            g8to11 = _mm_min_ps(g8to11, max_byte_float);
            g12to15 = _mm_min_ps(g12to15, max_byte_float);

            b0to3 = _mm_min_ps(b0to3, max_byte_float);
            b4to7 = _mm_min_ps(b4to7, max_byte_float);
            b8to11 = _mm_min_ps(b8to11, max_byte_float);
            b12to15 = _mm_min_ps(b12to15, max_byte_float);

            r0to3 = _mm_max_ps(r0to3, zero_float);
            r4to7 = _mm_max_ps(r4to7, zero_float);
            r8to11 = _mm_max_ps(r8to11, zero_float);
            r12to15 = _mm_max_ps(r12to15, zero_float);

            g0to3 = _mm_max_ps(g0to3, zero_float);
            g4to7 = _mm_max_ps(g4to7, zero_float);
            g8to11 = _mm_max_ps(g8to11, zero_float);
            g12to15 = _mm_max_ps(g12to15, zero_float);

            b0to3 = _mm_max_ps(b0to3, zero_float);
            b4to7 = _mm_max_ps(b4to7, zero_float);
            b8to11 = _mm_max_ps(b8to11, zero_float);
            b12to15 = _mm_max_ps(b12to15, zero_float);

            // convert back to integer and apply bit shifting to get the final 32bit output
            __m128i out0to3 = _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(r0to3), 24),
                                           _mm_slli_epi32(_mm_cvtps_epi32(g0to3), 16));
            out0to3 = _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(b0to3), 8), out0to3);

            __m128i out4to7 = _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(r4to7), 24),
                                           _mm_slli_epi32(_mm_cvtps_epi32(g4to7), 16));
            out4to7 = _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(b4to7), 8), out4to7);

            __m128i out8to11 = _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(r8to11), 24),
                                            _mm_slli_epi32(_mm_cvtps_epi32(g8to11), 16));
            out8to11 = _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(b8to11), 8), out8to11);

            __m128i out12to15 = _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(r12to15), 24),
                                             _mm_slli_epi32(_mm_cvtps_epi32(g12to15), 16));
            out12to15 = _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(b12to15), 8), out12to15);

            unsigned int tile = x / 8;
            unsigned int tile_x = x % 8;

            // since SSE is 128bit wide we compute 16 pixels in parallel (input), which
            // becomes 4 128bit wide int32 vector, hence 4 stores
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&output[tile][y * 8 + tile_x]), out0to3);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&output[tile][y * 8 + tile_x + 4]),
                             out4to7);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&output[tile + 1][y * 8 + tile_x]),
                             out8to11);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&output[tile + 1][y * 8 + tile_x + 4]),
                             out12to15);
        }
    }
}

#else

static void ConvertYUVToRGB_YUV422_420(InputFormat input_format, const u8* __restrict__ input_Y,
                                       const u8* __restrict__ input_U,
                                       const u8* __restrict__ input_V, ImageTile output[],
                                       unsigned int width, unsigned int height,
                                       const CoefficientSet& coefficients) {
    auto& c = coefficients;
    const s32 rounding_offset = 0x18;

    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; x += 8) {
            s32 Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7;
            s32 U0, U2, U4, U6;
            s32 V0, V2, V4, V6;
            int y_offset, uv_offset;

            switch (input_format) {
            case InputFormat::YUV422_Indiv8:
            case InputFormat::YUV422_Indiv16:
                y_offset = y * width + x;
                uv_offset = (y * width + x) / 2;
                break;
            case InputFormat::YUV420_Indiv8:
            case InputFormat::YUV420_Indiv16:
                y_offset = y * width + x;
                uv_offset = ((y / 2) * width + x) / 2;
                break;
            default:
                y_offset = 0;
                uv_offset = 0;
                break;
            }

            Y0 = input_Y[y_offset];
            Y1 = input_Y[y_offset + 1];
            Y2 = input_Y[y_offset + 2];
            Y3 = input_Y[y_offset + 3];
            Y4 = input_Y[y_offset + 4];
            Y5 = input_Y[y_offset + 5];
            Y6 = input_Y[y_offset + 6];
            Y7 = input_Y[y_offset + 7];

            U0 = input_U[uv_offset];
            U2 = input_U[uv_offset + 1];
            U4 = input_U[uv_offset + 2];
            U6 = input_U[uv_offset + 3];

            V0 = input_V[uv_offset];
            V2 = input_V[uv_offset + 1];
            V4 = input_V[uv_offset + 2];
            V6 = input_V[uv_offset + 3];

            s32 cY0 = c[0] * Y0;
            s32 cY1 = c[0] * Y1;
            s32 cY2 = c[0] * Y2;
            s32 cY3 = c[0] * Y3;
            s32 cY4 = c[0] * Y4;
            s32 cY5 = c[0] * Y5;
            s32 cY6 = c[0] * Y6;
            s32 cY7 = c[0] * Y7;

            s32 r0 = ((cY0 + c[1] * V0) >> 3) + c[5] + rounding_offset;
            s32 r1 = ((cY1 + c[1] * V0) >> 3) + c[5] + rounding_offset;
            s32 r2 = ((cY2 + c[1] * V2) >> 3) + c[5] + rounding_offset;
            s32 r3 = ((cY3 + c[1] * V2) >> 3) + c[5] + rounding_offset;
            s32 r4 = ((cY4 + c[1] * V4) >> 3) + c[5] + rounding_offset;
            s32 r5 = ((cY5 + c[1] * V4) >> 3) + c[5] + rounding_offset;
            s32 r6 = ((cY6 + c[1] * V6) >> 3) + c[5] + rounding_offset;
            s32 r7 = ((cY7 + c[1] * V6) >> 3) + c[5] + rounding_offset;

            s32 g0 = ((cY0 - c[2] * V0 - c[3] * U0) >> 3) + c[6] + rounding_offset;
            s32 g1 = ((cY1 - c[2] * V0 - c[3] * U0) >> 3) + c[6] + rounding_offset;
            s32 g2 = ((cY2 - c[2] * V2 - c[3] * U2) >> 3) + c[6] + rounding_offset;
            s32 g3 = ((cY3 - c[2] * V2 - c[3] * U2) >> 3) + c[6] + rounding_offset;
            s32 g4 = ((cY4 - c[2] * V4 - c[3] * U4) >> 3) + c[6] + rounding_offset;
            s32 g5 = ((cY5 - c[2] * V4 - c[3] * U4) >> 3) + c[6] + rounding_offset;
            s32 g6 = ((cY6 - c[2] * V6 - c[3] * U6) >> 3) + c[6] + rounding_offset;
            s32 g7 = ((cY7 - c[2] * V6 - c[3] * U6) >> 3) + c[6] + rounding_offset;

            s32 b0 = ((cY0 + c[4] * U0) >> 3) + c[7] + rounding_offset;
            s32 b1 = ((cY1 + c[4] * U0) >> 3) + c[7] + rounding_offset;
            s32 b2 = ((cY2 + c[4] * U2) >> 3) + c[7] + rounding_offset;
            s32 b3 = ((cY3 + c[4] * U2) >> 3) + c[7] + rounding_offset;
            s32 b4 = ((cY4 + c[4] * U4) >> 3) + c[7] + rounding_offset;
            s32 b5 = ((cY5 + c[4] * U4) >> 3) + c[7] + rounding_offset;
            s32 b6 = ((cY6 + c[4] * U6) >> 3) + c[7] + rounding_offset;
            s32 b7 = ((cY7 + c[4] * U6) >> 3) + c[7] + rounding_offset;

            unsigned int tile = x >> 3;
            unsigned int tile_x = x % 8;

            int out_offset = y * 8 + tile_x;
            output[tile][out_offset] = ((u32)std::clamp(r0 >> 5, 0, 0xFF) << 24) |
                                       ((u32)std::clamp(g0 >> 5, 0, 0xFF) << 16) |
                                       ((u32)std::clamp(b0 >> 5, 0, 0xFF) << 8);
            output[tile][out_offset + 1] = ((u32)std::clamp(r1 >> 5, 0, 0xFF) << 24) |
                                           ((u32)std::clamp(g1 >> 5, 0, 0xFF) << 16) |
                                           ((u32)std::clamp(b1 >> 5, 0, 0xFF) << 8);
            output[tile][out_offset + 2] = ((u32)std::clamp(r2 >> 5, 0, 0xFF) << 24) |
                                           ((u32)std::clamp(g2 >> 5, 0, 0xFF) << 16) |
                                           ((u32)std::clamp(b2 >> 5, 0, 0xFF) << 8);
            output[tile][out_offset + 3] = ((u32)std::clamp(r3 >> 5, 0, 0xFF) << 24) |
                                           ((u32)std::clamp(g3 >> 5, 0, 0xFF) << 16) |
                                           ((u32)std::clamp(b3 >> 5, 0, 0xFF) << 8);
            output[tile][out_offset + 4] = ((u32)std::clamp(r4 >> 5, 0, 0xFF) << 24) |
                                           ((u32)std::clamp(g4 >> 5, 0, 0xFF) << 16) |
                                           ((u32)std::clamp(b4 >> 5, 0, 0xFF) << 8);
            output[tile][out_offset + 5] = ((u32)std::clamp(r5 >> 5, 0, 0xFF) << 24) |
                                           ((u32)std::clamp(g5 >> 5, 0, 0xFF) << 16) |
                                           ((u32)std::clamp(b5 >> 5, 0, 0xFF) << 8);
            output[tile][out_offset + 6] = ((u32)std::clamp(r6 >> 5, 0, 0xFF) << 24) |
                                           ((u32)std::clamp(g6 >> 5, 0, 0xFF) << 16) |
                                           ((u32)std::clamp(b6 >> 5, 0, 0xFF) << 8);
            output[tile][out_offset + 7] = ((u32)std::clamp(r7 >> 5, 0, 0xFF) << 24) |
                                           ((u32)std::clamp(g7 >> 5, 0, 0xFF) << 16) |
                                           ((u32)std::clamp(b7 >> 5, 0, 0xFF) << 8);
        }
    }
}

#endif

static void ConvertYUVToRGB_YUV422_Interleaved(const u8* input_Y, const u8* input_U,
                                               const u8* input_V, ImageTile output[],
                                               unsigned int width, unsigned int height,
                                               const CoefficientSet& coefficients) {
    auto& c = coefficients;
    const s32 rounding_offset = 0x18;

    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; x += 8) {
            s32 Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7;
            s32 U0, U2, U4, U6;
            s32 V0, V2, V4, V6;
            Y0 = input_Y[(y * width + x) * 2];
            Y1 = input_Y[(y * width + x) * 2 + 2];
            Y2 = input_Y[(y * width + x) * 2 + 4];
            Y3 = input_Y[(y * width + x) * 2 + 6];
            Y4 = input_Y[(y * width + x) * 2 + 8];
            Y5 = input_Y[(y * width + x) * 2 + 10];
            Y6 = input_Y[(y * width + x) * 2 + 12];
            Y7 = input_Y[(y * width + x) * 2 + 14];

            U0 = input_Y[(y * width + (x / 2) * 2) * 2 + 1];
            U2 = input_Y[(y * width + (x / 2) * 2) * 2 + 5];
            U4 = input_Y[(y * width + (x / 2) * 2) * 2 + 9];
            U6 = input_Y[(y * width + (x / 2) * 2) * 2 + 13];

            V0 = input_Y[(y * width + (x / 2) * 2) * 2 + 3];
            V2 = input_Y[(y * width + (x / 2) * 2) * 2 + 7];
            V4 = input_Y[(y * width + (x / 2) * 2) * 2 + 11];
            V6 = input_Y[(y * width + (x / 2) * 2) * 2 + 15];

            // This conversion process is bit-exact with hardware, as far as could be tested.

            s32 cY0 = c[0] * Y0;
            s32 cY1 = c[0] * Y1;
            s32 cY2 = c[0] * Y2;
            s32 cY3 = c[0] * Y3;
            s32 cY4 = c[0] * Y4;
            s32 cY5 = c[0] * Y5;
            s32 cY6 = c[0] * Y6;
            s32 cY7 = c[0] * Y7;

            s32 r0 = ((cY0 + c[1] * V0) >> 3) + c[5] + rounding_offset;
            s32 r1 = ((cY1 + c[1] * V0) >> 3) + c[5] + rounding_offset;
            s32 r2 = ((cY2 + c[1] * V2) >> 3) + c[5] + rounding_offset;
            s32 r3 = ((cY3 + c[1] * V2) >> 3) + c[5] + rounding_offset;
            s32 r4 = ((cY4 + c[1] * V4) >> 3) + c[5] + rounding_offset;
            s32 r5 = ((cY5 + c[1] * V4) >> 3) + c[5] + rounding_offset;
            s32 r6 = ((cY6 + c[1] * V6) >> 3) + c[5] + rounding_offset;
            s32 r7 = ((cY7 + c[1] * V6) >> 3) + c[5] + rounding_offset;

            s32 g0 = ((cY0 - c[2] * V0 - c[3] * U0) >> 3) + c[6] + rounding_offset;
            s32 g1 = ((cY1 - c[2] * V0 - c[3] * U0) >> 3) + c[6] + rounding_offset;
            s32 g2 = ((cY2 - c[2] * V2 - c[3] * U2) >> 3) + c[6] + rounding_offset;
            s32 g3 = ((cY3 - c[2] * V2 - c[3] * U2) >> 3) + c[6] + rounding_offset;
            s32 g4 = ((cY4 - c[2] * V4 - c[3] * U4) >> 3) + c[6] + rounding_offset;
            s32 g5 = ((cY5 - c[2] * V4 - c[3] * U4) >> 3) + c[6] + rounding_offset;
            s32 g6 = ((cY6 - c[2] * V6 - c[3] * U6) >> 3) + c[6] + rounding_offset;
            s32 g7 = ((cY7 - c[2] * V6 - c[3] * U6) >> 3) + c[6] + rounding_offset;

            s32 b0 = ((cY0 + c[4] * U0) >> 3) + c[7] + rounding_offset;
            s32 b1 = ((cY1 + c[4] * U0) >> 3) + c[7] + rounding_offset;
            s32 b2 = ((cY2 + c[4] * U2) >> 3) + c[7] + rounding_offset;
            s32 b3 = ((cY3 + c[4] * U2) >> 3) + c[7] + rounding_offset;
            s32 b4 = ((cY4 + c[4] * U4) >> 3) + c[7] + rounding_offset;
            s32 b5 = ((cY5 + c[4] * U4) >> 3) + c[7] + rounding_offset;
            s32 b6 = ((cY6 + c[4] * U6) >> 3) + c[7] + rounding_offset;
            s32 b7 = ((cY7 + c[4] * U6) >> 3) + c[7] + rounding_offset;

            unsigned int tile = x >> 3;
            unsigned int tile_x = x % 8;

            output[tile][y * 8 + tile_x] = ((u32)std::clamp(r0 >> 5, 0, 0xFF) << 24) |
                                           ((u32)std::clamp(g0 >> 5, 0, 0xFF) << 16) |
                                           ((u32)std::clamp(b0 >> 5, 0, 0xFF) << 8);
            output[tile][y * 8 + tile_x + 1] = ((u32)std::clamp(r1 >> 5, 0, 0xFF) << 24) |
                                               ((u32)std::clamp(g1 >> 5, 0, 0xFF) << 16) |
                                               ((u32)std::clamp(b1 >> 5, 0, 0xFF) << 8);
            output[tile][y * 8 + tile_x + 2] = ((u32)std::clamp(r2 >> 5, 0, 0xFF) << 24) |
                                               ((u32)std::clamp(g2 >> 5, 0, 0xFF) << 16) |
                                               ((u32)std::clamp(b2 >> 5, 0, 0xFF) << 8);
            output[tile][y * 8 + tile_x + 3] = ((u32)std::clamp(r3 >> 5, 0, 0xFF) << 24) |
                                               ((u32)std::clamp(g3 >> 5, 0, 0xFF) << 16) |
                                               ((u32)std::clamp(b3 >> 5, 0, 0xFF) << 8);
            output[tile][y * 8 + tile_x + 4] = ((u32)std::clamp(r4 >> 5, 0, 0xFF) << 24) |
                                               ((u32)std::clamp(g4 >> 5, 0, 0xFF) << 16) |
                                               ((u32)std::clamp(b4 >> 5, 0, 0xFF) << 8);
            output[tile][y * 8 + tile_x + 5] = ((u32)std::clamp(r5 >> 5, 0, 0xFF) << 24) |
                                               ((u32)std::clamp(g5 >> 5, 0, 0xFF) << 16) |
                                               ((u32)std::clamp(b5 >> 5, 0, 0xFF) << 8);
            output[tile][y * 8 + tile_x + 6] = ((u32)std::clamp(r6 >> 5, 0, 0xFF) << 24) |
                                               ((u32)std::clamp(g6 >> 5, 0, 0xFF) << 16) |
                                               ((u32)std::clamp(b6 >> 5, 0, 0xFF) << 8);
            output[tile][y * 8 + tile_x + 7] = ((u32)std::clamp(r7 >> 5, 0, 0xFF) << 24) |
                                               ((u32)std::clamp(g7 >> 5, 0, 0xFF) << 16) |
                                               ((u32)std::clamp(b7 >> 5, 0, 0xFF) << 8);
        }
    }
}

/// Converts a image strip from the source YUV format into individual 8x8 RGB32 tiles.
static void ConvertYUVToRGB(InputFormat input_format, const u8* __restrict__ input_Y,
                            const u8* __restrict__ input_U, const u8* __restrict__ input_V,
                            ImageTile output[], unsigned int width, unsigned int height,
                            const CoefficientSet& coefficients) {
    switch (input_format) {
    case InputFormat::YUV422_Indiv8:
    case InputFormat::YUV422_Indiv16:
    case InputFormat::YUV420_Indiv8:
    case InputFormat::YUV420_Indiv16:
        ConvertYUVToRGB_YUV422_420(input_format, input_Y, input_U, input_V, output, width, height,
                                   coefficients);
        break;
    case InputFormat::YUYV422_Interleaved:
        ConvertYUVToRGB_YUV422_Interleaved(input_Y, input_U, input_V, output, width, height,
                                           coefficients);
    default:
        break;
    }
}

/// Simulates an incoming CDMA transfer. The N parameter is used to automatically convert 16-bit
/// formats to 8-bit.
template <std::size_t N>
static void ReceiveData(Memory::MemorySystem& memory, u8* output, ConversionBuffer& buf,
                        std::size_t amount_of_data) {
    const u8* input = memory.GetPointer(buf.address);

    std::size_t output_unit = buf.transfer_unit / N;
    ASSERT(amount_of_data % output_unit == 0);

    while (amount_of_data > 0) {
        for (std::size_t i = 0; i < output_unit; ++i) {
            output[i] = input[i * N];
        }

        output += output_unit;
        input += buf.transfer_unit + buf.gap;

        buf.address += buf.transfer_unit + buf.gap;
        buf.image_size -= buf.transfer_unit;
        amount_of_data -= output_unit;
    }
}

/// Convert intermediate RGB32 format to the final output format while simulating an outgoing CDMA
/// transfer.
static void SendData(Memory::MemorySystem& memory, const u32* input, ConversionBuffer& buf,
                     int amount_of_data, OutputFormat output_format, u8 alpha) {

    u8* output = memory.GetPointer(buf.address);

    while (amount_of_data > 0) {
        u8* unit_end = output + buf.transfer_unit;
        while (output < unit_end) {
            u32 color = *input++;
            Common::Vec4<u8> col_vec{(u8)(color >> 24), (u8)(color >> 16), (u8)(color >> 8), alpha};

            switch (output_format) {
            case OutputFormat::RGBA8:
                Color::EncodeRGBA8(col_vec, output);
                output += 4;
                break;
            case OutputFormat::RGB8:
                Color::EncodeRGB8(col_vec, output);
                output += 3;
                break;
            case OutputFormat::RGB5A1:
                Color::EncodeRGB5A1(col_vec, output);
                output += 2;
                break;
            case OutputFormat::RGB565:
                Color::EncodeRGB565(col_vec, output);
                output += 2;
                break;
            }

            amount_of_data -= 1;
        }

        output += buf.gap;
        buf.address += buf.transfer_unit + buf.gap;
        buf.image_size -= buf.transfer_unit;
    }
}

static const u8 linear_lut[TILE_SIZE] = {
    // clang-format off
     0,  1,  2,  3,  4,  5,  6,  7,
     8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63,
    // clang-format on
};

static const u8 morton_lut[TILE_SIZE] = {
    // clang-format off
     0,  1,  4,  5, 16, 17, 20, 21,
     2,  3,  6,  7, 18, 19, 22, 23,
     8,  9, 12, 13, 24, 25, 28, 29,
    10, 11, 14, 15, 26, 27, 30, 31,
    32, 33, 36, 37, 48, 49, 52, 53,
    34, 35, 38, 39, 50, 51, 54, 55,
    40, 41, 44, 45, 56, 57, 60, 61,
    42, 43, 46, 47, 58, 59, 62, 63,
    // clang-format on
};

static void RotateTile0(const ImageTile& input, ImageTile& output, int height,
                        const u8 out_map[64]) {
    for (int i = 0; i < height * 8; ++i) {
        output[out_map[i]] = input[i];
    }
}

static void RotateTile90(const ImageTile& input, ImageTile& output, int height,
                         const u8 out_map[64]) {
    int out_i = 0;
    for (int x = 0; x < 8; ++x) {
        for (int y = height - 1; y >= 0; --y) {
            output[out_map[out_i++]] = input[y * 8 + x];
        }
    }
}

static void RotateTile180(const ImageTile& input, ImageTile& output, int height,
                          const u8 out_map[64]) {
    int out_i = 0;
    for (int i = height * 8 - 1; i >= 0; --i) {
        output[out_map[out_i++]] = input[i];
    }
}

static void RotateTile270(const ImageTile& input, ImageTile& output, int height,
                          const u8 out_map[64]) {
    int out_i = 0;
    for (int x = 8 - 1; x >= 0; --x) {
        for (int y = 0; y < height; ++y) {
            output[out_map[out_i++]] = input[y * 8 + x];
        }
    }
}

static void WriteTileToOutput(u32* output, const ImageTile& tile, int height, int line_stride) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < 8; ++x) {
            output[y * line_stride + x] = tile[y * 8 + x];
        }
    }
}

/**
 * Performs a Y2R colorspace conversion.
 *
 * The Y2R hardware implements hardware-accelerated YUV to RGB colorspace conversions. It is most
 * commonly used for video playback or to display camera input to the screen.
 *
 * The conversion process is quite configurable, and can be divided in distinct steps. From
 * observation, it appears that the hardware buffers a single 8-pixel tall strip of image data
 * internally and converts it in one go before writing to the output and loading the next strip.
 *
 * The steps taken to convert one strip of image data are:
 *
 * - The hardware receives data via CDMA (http://3dbrew.org/wiki/Corelink_DMA_Engines), which is
 *   presumably stored in one or more internal buffers. This process can be done in several separate
 *   transfers, as long as they don't exceed the size of the internal image buffer. This allows
 *   flexibility in input strides.
 * - The input data is decoded into a YUV tuple. Several formats are suported, see the `InputFormat`
 *   enum.
 * - The YUV tuple is converted, using fixed point calculations, to RGB. This step can be configured
 *   using a set of coefficients to support different colorspace standards. See `CoefficientSet`.
 * - The strip can be optionally rotated 90, 180 or 270 degrees. Since each strip is processed
 *   independently, this notably rotates each *strip*, not the entire image. This means that for 90
 *   or 270 degree rotations, the output will be in terms of several 8 x height images, and for any
 *   non-zero rotation the strips will have to be re-arranged so that the parts of the image will
 *   not be shuffled together. This limitation makes this a feature of somewhat dubious utility. 90
 *   or 270 degree rotations in images with non-even height don't seem to work properly.
 * - The data is converted to the output RGB format. See the `OutputFormat` enum.
 * - The data can be output either linearly line-by-line or in the swizzled 8x8 tile format used by
 *   the PICA. This is decided by the `BlockAlignment` enum. If 8x8 alignment is used, then the
 *   image must have a height divisible by 8. The image width must always be divisible by 8.
 * - The final data is then CDMAed out to main memory and the next image strip is processed. This
 *   offers the same flexibility as the input stage.
 *
 * In this implementation, to avoid the combinatorial explosion of parameter combinations, common
 * intermediate formats are used and where possible tables or parameters are used instead of
 * diverging code paths to keep the amount of branches in check. Some steps are also merged to
 * increase efficiency.
 *
 * Output for all valid settings combinations matches hardware, however output in some edge-cases
 * differs:
 *
 * - `Block8x8` alignment with non-mod8 height produces different garbage patterns on the last
 *   strip, especially when combined with rotation.
 * - Hardware, when using `Linear` alignment with a non-even height and 90 or 270 degree rotation
 *   produces misaligned output on the last strip. This implmentation produces output with the
 *   correct "expected" alignment.
 *
 * Hardware behaves strangely (doesn't fire the completion interrupt, for example) in these cases,
 * so they are believed to be invalid configurations anyway.
 */
void PerformConversion(Memory::MemorySystem& memory, ConversionConfiguration& cvt) {
    ASSERT(cvt.input_line_width % 8 == 0);
    ASSERT(cvt.block_alignment != BlockAlignment::Block8x8 || cvt.input_lines % 8 == 0);
    // Tiles per row
    std::size_t num_tiles = cvt.input_line_width / 8;
    ASSERT(num_tiles <= MAX_TILES);

    // Buffer used as a CDMA source/target.
    std::unique_ptr<u8[]> data_buffer(new u8[cvt.input_line_width * 8 * 4]);
    // Intermediate storage for decoded 8x8 image tiles. Always stored as RGB32.
    std::unique_ptr<ImageTile[]> tiles(new ImageTile[num_tiles]);
    ImageTile tmp_tile;

    // LUT used to remap writes to a tile. Used to allow linear or swizzled output without
    // requiring two different code paths.
    const u8* tile_remap = nullptr;
    switch (cvt.block_alignment) {
    case BlockAlignment::Linear:
        tile_remap = linear_lut;
        break;
    case BlockAlignment::Block8x8:
        tile_remap = morton_lut;
        break;
    }

    for (unsigned int y = 0; y < cvt.input_lines; y += 8) {
        unsigned int row_height = std::min(cvt.input_lines - y, 8u);

        // Total size in pixels of incoming data required for this strip.
        const std::size_t row_data_size = row_height * cvt.input_line_width;

        u8* input_Y = data_buffer.get();
        u8* input_U = input_Y + 8 * cvt.input_line_width;
        u8* input_V = input_U + 8 * cvt.input_line_width / 2;

        switch (cvt.input_format) {
        case InputFormat::YUV422_Indiv8:
            ReceiveData<1>(memory, input_Y, cvt.src_Y, row_data_size);
            ReceiveData<1>(memory, input_U, cvt.src_U, row_data_size / 2);
            ReceiveData<1>(memory, input_V, cvt.src_V, row_data_size / 2);
            break;
        case InputFormat::YUV420_Indiv8:
            ReceiveData<1>(memory, input_Y, cvt.src_Y, row_data_size);
            ReceiveData<1>(memory, input_U, cvt.src_U, row_data_size / 4);
            ReceiveData<1>(memory, input_V, cvt.src_V, row_data_size / 4);
            break;
        case InputFormat::YUV422_Indiv16:
            ReceiveData<2>(memory, input_Y, cvt.src_Y, row_data_size);
            ReceiveData<2>(memory, input_U, cvt.src_U, row_data_size / 2);
            ReceiveData<2>(memory, input_V, cvt.src_V, row_data_size / 2);
            break;
        case InputFormat::YUV420_Indiv16:
            ReceiveData<2>(memory, input_Y, cvt.src_Y, row_data_size);
            ReceiveData<2>(memory, input_U, cvt.src_U, row_data_size / 4);
            ReceiveData<2>(memory, input_V, cvt.src_V, row_data_size / 4);
            break;
        case InputFormat::YUYV422_Interleaved:
            input_U = nullptr;
            input_V = nullptr;
            ReceiveData<1>(memory, input_Y, cvt.src_YUYV, row_data_size * 2);
            break;
        }

        // Note(yuriks): If additional optimization is required, input_format can be moved to a
        // template parameter, so that its dispatch can be moved to outside the inner loop.
        ConvertYUVToRGB(cvt.input_format, input_Y, input_U, input_V, tiles.get(),
                        cvt.input_line_width, row_height, cvt.coefficients);

        u32* output_buffer = reinterpret_cast<u32*>(data_buffer.get());

        for (std::size_t i = 0; i < num_tiles; ++i) {
            int image_strip_width = 0;
            int output_stride = 0;

            switch (cvt.rotation) {
            case Rotation::None:
                RotateTile0(tiles[i], tmp_tile, row_height, tile_remap);
                image_strip_width = cvt.input_line_width;
                output_stride = 8;
                break;
            case Rotation::Clockwise_90:
                RotateTile90(tiles[i], tmp_tile, row_height, tile_remap);
                image_strip_width = 8;
                output_stride = 8 * row_height;
                break;
            case Rotation::Clockwise_180:
                // For 180 and 270 degree rotations we also invert the order of tiles in the strip,
                // since the rotates are done individually on each tile.
                RotateTile180(tiles[num_tiles - i - 1], tmp_tile, row_height, tile_remap);
                image_strip_width = cvt.input_line_width;
                output_stride = 8;
                break;
            case Rotation::Clockwise_270:
                RotateTile270(tiles[num_tiles - i - 1], tmp_tile, row_height, tile_remap);
                image_strip_width = 8;
                output_stride = 8 * row_height;
                break;
            }

            switch (cvt.block_alignment) {
            case BlockAlignment::Linear:
                WriteTileToOutput(output_buffer, tmp_tile, row_height, image_strip_width);
                output_buffer += output_stride;
                break;
            case BlockAlignment::Block8x8:
                WriteTileToOutput(output_buffer, tmp_tile, 8, 8);
                output_buffer += TILE_SIZE;
                break;
            }
        }

        // Note(yuriks): If additional optimization is required, output_format can be moved to a
        // template parameter, so that its dispatch can be moved to outside the inner loop.
        SendData(memory, reinterpret_cast<u32*>(data_buffer.get()), cvt.dst, (int)row_data_size,
                 cvt.output_format, (u8)cvt.alpha);
    }
}
} // namespace HW::Y2R
