#include "hlf.h"

// Source-order translation of kernels::skin_adjust::hlf_face_apply and the
// retouch branch of body_hlf_apply. Build with -fmad=false.

__device__ __forceinline__ float hlf_add(float a, float b) {
  return __fadd_rn(a, b);
}

__device__ __forceinline__ float hlf_sub(float a, float b) {
  return __fsub_rn(a, b);
}

__device__ __forceinline__ float hlf_mul(float a, float b) {
  return __fmul_rn(a, b);
}

__device__ __forceinline__ float hlf_div(float a, float b) {
  return __fdiv_rn(a, b);
}

__device__ __forceinline__ float hlf_clamp(float value, float lo, float hi) {
  return fminf(fmaxf(value, lo), hi);
}

// Matches Rust mix(from, to, amount) = from*(1-amount) + to*amount.
__device__ __forceinline__ float hlf_mix(float from, float to, float amount) {
  return hlf_add(hlf_mul(from, hlf_sub(1.0f, amount)),
                 hlf_mul(to, amount));
}

__device__ __forceinline__ int hlf_clamp_index(int64_t value, int extent) {
  return static_cast<int>(min(max(value, int64_t{0}),
                              int64_t{extent - 1}));
}

__device__ __forceinline__ void hlf_sort9(float* values) {
  for (int i = 1; i < 9; ++i) {
    const float key = values[i];
    int j = i - 1;
    while (j >= 0 && values[j] > key) {
      values[j + 1] = values[j];
      --j;
    }
    values[j + 1] = key;
  }
}

extern "C" __global__ void hlf_median_rgb_kernel(
    const float* source_rgb,
    float* out_rgb,
    HlfPreprocessUniforms uniforms) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= static_cast<int>(uniforms.width) ||
      y >= static_cast<int>(uniforms.height)) {
    return;
  }

  const size_t pixel =
      static_cast<size_t>(y) * static_cast<size_t>(uniforms.width) +
      static_cast<size_t>(x);
  const size_t rgb = pixel * 3u;
  const int64_t step = static_cast<int64_t>(uniforms.interval);
  for (unsigned int channel = 0; channel < 3u; ++channel) {
    float samples[9];
    int index = 0;
    for (int dy = -1; dy <= 1; ++dy) {
      const int sy = hlf_clamp_index(
          static_cast<int64_t>(y) + static_cast<int64_t>(dy) * step,
          static_cast<int>(uniforms.height));
      for (int dx = -1; dx <= 1; ++dx) {
        const int sx = hlf_clamp_index(
            static_cast<int64_t>(x) + static_cast<int64_t>(dx) * step,
            static_cast<int>(uniforms.width));
        samples[index++] =
            source_rgb[(static_cast<size_t>(sy) * uniforms.width +
                        static_cast<size_t>(sx)) *
                           3u +
                       channel];
      }
    }
    hlf_sort9(samples);
    out_rgb[rgb + channel] = samples[4];
  }
}

extern "C" __global__ void hlf_gauss_rgb_kernel(
    const float* source_rgb,
    float* out_rgb,
    HlfPreprocessUniforms uniforms) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= static_cast<int>(uniforms.width) ||
      y >= static_cast<int>(uniforms.height)) {
    return;
  }

  const float weights[9] = {
      0.0162f, 0.0540f, 0.1216f, 0.1945f, 0.2270f,
      0.1945f, 0.1216f, 0.0540f, 0.0162f};
  const size_t pixel =
      static_cast<size_t>(y) * static_cast<size_t>(uniforms.width) +
      static_cast<size_t>(x);
  const size_t rgb = pixel * 3u;
  const int64_t step = static_cast<int64_t>(uniforms.interval);
  for (unsigned int channel = 0; channel < 3u; ++channel) {
    float horizontal[9];
    for (int dy = -4; dy <= 4; ++dy) {
      const int sy = hlf_clamp_index(
          static_cast<int64_t>(y) + static_cast<int64_t>(dy) * step,
          static_cast<int>(uniforms.height));
      float value = 0.0f;
      for (int dx = -4; dx <= 4; ++dx) {
        const int sx = hlf_clamp_index(
            static_cast<int64_t>(x) + static_cast<int64_t>(dx) * step,
            static_cast<int>(uniforms.width));
        const float sample =
            source_rgb[(static_cast<size_t>(sy) * uniforms.width +
                        static_cast<size_t>(sx)) *
                           3u +
                       channel];
        value = hlf_add(value, hlf_mul(sample, weights[dx + 4]));
      }
      horizontal[dy + 4] = value;
    }
    float value = 0.0f;
    for (int dy = 0; dy < 9; ++dy) {
      value = hlf_add(value, hlf_mul(horizontal[dy], weights[dy]));
    }
    out_rgb[rgb + channel] = value;
  }
}

__device__ __forceinline__ float hlf_face_weight(
    const float* src,
    const float* median,
    const float* gauss,
    float degree) {
  float sum_variance = 0.0f;
  for (int channel = 0; channel < 3; ++channel) {
    const float value = hlf_mul(hlf_sub(src[channel], median[channel]), 7.07f);
    sum_variance = hlf_add(
        sum_variance, fminf(hlf_mul(value, value), 1.0f));
  }
  const float mean_variance = hlf_div(sum_variance, 3.0f);
  const float weight_factor = hlf_mix(
      1.0f,
      hlf_sub(1.0f, hlf_div(mean_variance,
                            hlf_add(mean_variance, 0.1f))),
      0.9f);
  const float probe = hlf_clamp(
      hlf_mul(hlf_sub(fminf(src[0], hlf_sub(gauss[0], 0.1f)), 0.2f),
              4.0f),
      0.0f, 1.0f);
  return hlf_mul(hlf_mul(weight_factor, probe), degree);
}

__device__ __forceinline__ unsigned char hlf_trunc_u8(float value) {
  if (value <= 0.0f) return 0;
  if (value >= 1.0f) return 255;
  return (unsigned char)(value * 255.0f);
}

// Vendor AdjustLut: 256x1 L8, byte/255, centre-coordinate linear tap,
// clamp-to-edge, 8-bit interpolation weights. The row is fixed; degree is
// applied only after this sample.
__device__ __forceinline__ float hlf_lut_sample(
    const unsigned char* lut,
    uint32_t width,
    float u) {
  u = hlf_clamp(u, 0.0f, 1.0f);
  const float fx =
      hlf_sub(hlf_mul(u, static_cast<float>(width)), 0.5f);
  const float x0f = floorf(fx);
  const int x0_raw = static_cast<int>(x0f);
  const int kx = static_cast<int>(floorf(hlf_add(
      hlf_mul(hlf_sub(fx, x0f), 256.0f), 0.5f)));
  const int max_x = static_cast<int>(width) - 1;
  const int x0 = min(max(x0_raw, 0), max_x);
  const int x1 = min(max(x0_raw + 1, 0), max_x);
  const int q = static_cast<int>(lut[x0]) * (256 - kx) +
                static_cast<int>(lut[x1]) * kx;
  return hlf_div(static_cast<float>(q), 65280.0f);
}

extern "C" __global__ void pf30021_hlf_face_kernel(
    const float* source_rgb,
    const float* median_rgb,
    const float* gauss_rgb,
    const float* face_mask,
    const float* skin_mask,
    float* out_rgb,
    HlfFaceUniforms uniforms) {
  const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= static_cast<int32_t>(uniforms.width) ||
      y >= static_cast<int32_t>(uniforms.height)) {
    return;
  }

  const size_t pixel =
      static_cast<size_t>(y) * static_cast<size_t>(uniforms.width) +
      static_cast<size_t>(x);
  const size_t rgb = pixel * 3u;
  const float src[3] = {
      source_rgb[rgb], source_rgb[rgb + 1u], source_rgb[rgb + 2u]};
  const float med[3] = {
      median_rgb[rgb], median_rgb[rgb + 1u], median_rgb[rgb + 2u]};
  const float gau[3] = {
      gauss_rgb[rgb], gauss_rgb[rgb + 1u], gauss_rgb[rgb + 2u]};
  const float weight =
      hlf_face_weight(src, med, gau, uniforms.degree);
  const float coverage = hlf_mul(skin_mask[pixel], face_mask[pixel]);

  for (unsigned int channel = 0; channel < 3u; ++channel) {
    const float high = hlf_sub(src[channel], med[channel]);
    const float reference = hlf_add(
        hlf_mix(src[channel], gau[channel], weight), hlf_mul(high, 0.01f));
    const float diff = hlf_clamp(
        hlf_sub(reference, src[channel]),
        -0.047058823529411764f,
        0.054901960784313725f);
    out_rgb[rgb + channel] =
        hlf_add(src[channel], hlf_mul(diff, coverage));
  }
}

extern "C" __global__ void pf30022_hlf_body_kernel(
    const unsigned char* source_rgba,
    const float* median_rgb,
    const float* gauss_rgb,
    const float* face_mask_rgba,
    const float* skin_mask,
    const unsigned char* adjust_lut,
    unsigned char* out_rgba,
    HlfBodyUniforms uniforms) {
  const uint32_t x =
      uniforms.roi_x + blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y =
      uniforms.roi_y + blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= uniforms.roi_x + uniforms.roi_w ||
      y >= uniforms.roi_y + uniforms.roi_h) {
    return;
  }

  const size_t pixel =
      static_cast<size_t>(y) * static_cast<size_t>(uniforms.width) +
      static_cast<size_t>(x);
  const size_t rgba = pixel * 4u;
  const size_t rgb = pixel * 3u;
  const float face = face_mask_rgba[rgba];
  if (face <= 0.001f) return;

  const float src[3] = {
      hlf_mul(static_cast<float>(source_rgba[rgba]), 0.00392156862745098f),
      hlf_mul(static_cast<float>(source_rgba[rgba + 1u]),
              0.00392156862745098f),
      hlf_mul(static_cast<float>(source_rgba[rgba + 2u]),
              0.00392156862745098f)};
  const float med[3] = {
      median_rgb[rgb], median_rgb[rgb + 1u], median_rgb[rgb + 2u]};
  const float gau[3] = {
      gauss_rgb[rgb], gauss_rgb[rgb + 1u], gauss_rgb[rgb + 2u]};
  const float high[3] = {
      hlf_sub(src[0], med[0]),
      hlf_sub(src[1], med[1]),
      hlf_sub(src[2], med[2])};
  float result_high[3] = {high[0], high[1], high[2]};
  float reference[3] = {src[0], src[1], src[2]};

  if (uniforms.spotless_degree > 0.0f) {
    const float first = hlf_lut_sample(
        adjust_lut, uniforms.lut_width,
        hlf_add(hlf_mul(high[0], 0.5f), 0.5f));
    const float second = hlf_lut_sample(adjust_lut, uniforms.lut_width, first);
    const float adjusted[3] = {
        hlf_mul(hlf_sub(first, 0.5f), 2.0f),
        hlf_mul(hlf_sub(second, 0.5f), 2.0f),
        hlf_mul(hlf_sub(second, 0.5f), 2.0f)};
    const float magnitude = hlf_mul(
        sqrtf(hlf_add(hlf_add(hlf_mul(adjusted[0], adjusted[0]),
                              hlf_mul(adjusted[1], adjusted[1])),
                      hlf_mul(adjusted[2], adjusted[2]))),
        0.57735026f);
    const float darkest = fminf(fminf(src[0], src[1]), src[2]);
    if (magnitude > uniforms.spotless_acne_threshold || darkest < 0.01f) {
      const float spotless[3] = {
          hlf_mul(high[0], uniforms.spotless_high_freq_factor),
          hlf_mul(high[1], uniforms.spotless_high_freq_factor),
          hlf_mul(high[2], uniforms.spotless_high_freq_factor)};
      const float amount =
          hlf_mul(uniforms.spotless_degree, face_mask_rgba[rgba + 1u]);
      for (int channel = 0; channel < 3; ++channel) {
        result_high[channel] =
            hlf_mix(high[channel], spotless[channel], amount);
      }
      const float low_alpha =
          fminf(hlf_add(uniforms.spotless_low_freq_alpha, 0.2f), 1.0f);
      const float weight =
          hlf_mul(hlf_mul(low_alpha, uniforms.spotless_degree),
                  face_mask_rgba[rgba + 1u]);
      for (int channel = 0; channel < 3; ++channel) {
        reference[channel] = hlf_add(
            hlf_mix(med[channel], gau[channel], weight),
            result_high[channel]);
      }
    }
  }

  if (uniforms.retouch_degree > 0.0f) {
    float variance[3];
    for (int channel = 0; channel < 3; ++channel) {
      const float value =
          fminf(hlf_mul(result_high[channel], 7.07f), 1.0f);
      variance[channel] = fminf(hlf_mul(value, value), 1.0f);
    }
    const float probe = hlf_clamp(
        hlf_mul(hlf_sub(fminf(src[0], hlf_sub(gau[0], 0.1f)), 0.2f),
                4.0f),
        0.0f, 1.0f);
    const float mean_variance =
        hlf_div(hlf_add(hlf_add(variance[0], variance[1]), variance[2]),
                3.0f);
    const float weight_factor = hlf_mix(
        1.0f,
        hlf_sub(1.0f, hlf_div(mean_variance,
                              hlf_add(mean_variance, 0.1f))),
        0.9f);
    const float weight =
        hlf_mul(hlf_mul(weight_factor, probe), uniforms.retouch_degree);
    for (int channel = 0; channel < 3; ++channel) {
      reference[channel] = hlf_add(
          hlf_mix(reference[channel], gau[channel], weight),
          hlf_mul(result_high[channel],
                  uniforms.retouch_high_freq_factor));
    }
  }

  const float coverage = hlf_mul(skin_mask[pixel], face);
  for (unsigned int channel = 0; channel < 3u; ++channel) {
    const float diff = hlf_clamp(
        hlf_sub(reference[channel], src[channel]),
        uniforms.diff_min_range,
        uniforms.diff_max_range);
    const float corrected = hlf_add(src[channel], diff);
    out_rgba[rgba + channel] = hlf_trunc_u8(hlf_add(
        src[channel], hlf_mul(hlf_sub(corrected, src[channel]), coverage)));
  }
  out_rgba[rgba + 3u] = source_rgba[rgba + 3u];
}
