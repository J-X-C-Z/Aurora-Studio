#include "pf90004_radiance.h"

// Source-order CUDA translation of kernels::radiance. The brightness pass
// stores its 8-bit intermediate before the HSL pass reads it, matching the CPU
// oracle's two native-order passes. Compile with --fmad=false.

__device__ __forceinline__ float p4_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float p4_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float p4_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float p4_div(float a, float b) { return __fdiv_rn(a, b); }

__device__ __forceinline__ int p4_mini(int a, int b) { return b < a ? b : a; }
__device__ __forceinline__ int p4_maxi(int a, int b) { return b > a ? b : a; }

__device__ __forceinline__ float p4_clamp01(float value) {
  return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

__device__ __forceinline__ unsigned char p4_round_byte(float value) {
  int quantized = __float2int_rz(p4_add(p4_mul(value, 255.0f), 0.5f));
  return (unsigned char)p4_maxi(0, p4_mini(quantized, 255));
}

__device__ __forceinline__ float p4_gamma(
    const float* gamma,
    float value,
    unsigned int channel) {
  int index = __float2int_rz(p4_mul(p4_clamp01(value), 4096.0f));
  index = p4_mini(index, 4095);
  return gamma[index * 4u + channel];
}

__device__ __forceinline__ float p4_tone_component(
    float component,
    float maximum,
    float minimum,
    float high,
    float low) {
  if (component == maximum) return high;
  if (component == minimum) return low;
  float numerator = p4_mul(p4_sub(component, minimum), p4_sub(high, low));
  return p4_add(low, p4_div(numerator, p4_sub(maximum, minimum)));
}

__device__ __forceinline__ float3 p4_tone(
    float3 value,
    unsigned int channel,
    const float* gamma) {
  float v0 = p4_clamp01(value.x);
  float v1 = p4_clamp01(value.y);
  float v2 = p4_clamp01(value.z);
  float maximum = v0 > v1 ? v0 : v1;
  maximum = maximum > v2 ? maximum : v2;
  float minimum = v0 < v1 ? v0 : v1;
  minimum = minimum < v2 ? minimum : v2;
  float high = p4_gamma(gamma, maximum, channel);
  float low = p4_gamma(gamma, minimum, channel);
  if (maximum == minimum) return make_float3(high, high, high);
  return make_float3(
      p4_tone_component(v0, maximum, minimum, high, low),
      p4_tone_component(v1, maximum, minimum, high, low),
      p4_tone_component(v2, maximum, minimum, high, low));
}

__device__ __forceinline__ float3 p4_matrix(
    float3 value,
    const float* matrices,
    unsigned int start) {
  float x0 = p4_add(
      p4_add(p4_mul(value.x, matrices[start]),
             p4_mul(value.y, matrices[start + 1])),
      p4_mul(value.z, matrices[start + 2]));
  float x1 = p4_add(
      p4_add(p4_mul(value.x, matrices[start + 3]),
             p4_mul(value.y, matrices[start + 4])),
      p4_mul(value.z, matrices[start + 5]));
  float x2 = p4_add(
      p4_add(p4_mul(value.x, matrices[start + 6]),
             p4_mul(value.y, matrices[start + 7])),
      p4_mul(value.z, matrices[start + 8]));
  return make_float3(x0, x1, x2);
}

__device__ __forceinline__ float3 p4_to_rgb(float hue, float maximum, float minimum) {
  while (hue > 6.0f) hue = p4_sub(hue, 6.0f);
  float difference = p4_sub(maximum, minimum);
  if (hue >= 3.0f && hue <= 5.0f) {
    if (hue <= 4.0f) {
      return make_float3(
          minimum,
          p4_sub(minimum, p4_mul(p4_sub(hue, 4.0f), difference)),
          maximum);
    }
    return make_float3(
        p4_add(minimum, p4_mul(p4_sub(hue, 4.0f), difference)),
        minimum,
        maximum);
  }
  if (hue >= 1.0f && hue <= 3.0f) {
    if (hue <= 2.0f) {
      return make_float3(
          p4_sub(minimum, p4_mul(p4_sub(hue, 2.0f), difference)),
          maximum,
          minimum);
    }
    return make_float3(
        minimum,
        maximum,
        p4_add(minimum, p4_mul(p4_sub(hue, 2.0f), difference)));
  }
  if (hue > 5.0f) {
    return make_float3(
        maximum,
        minimum,
        p4_sub(minimum, p4_mul(p4_sub(hue, 6.0f), difference)));
  }
  return make_float3(
      maximum,
      p4_add(minimum, p4_mul(hue, difference)),
      minimum);
}

__device__ __forceinline__ float p4_palette(
    const float* hue_maps,
    unsigned int start,
    unsigned int index,
    float fraction) {
  float low = hue_maps[start + index];
  float high = hue_maps[start + index + 1];
  return p4_add(low, p4_mul(fraction, p4_sub(high, low)));
}

__device__ __forceinline__ float p4_mask_sum(
    float parsing_a,
    float parsing_g,
    float parsing_b,
    float face_g) {
  float value = p4_add(p4_mul(parsing_a, 0.39f), p4_mul(parsing_g, 0.39f));
  value = p4_add(value, p4_mul(parsing_b, 0.39f));
  return p4_add(value, p4_mul(face_g, 0.39f));
}

extern "C" __global__ void pf90004_radiance_kernel(
    const unsigned char* source_rgba,
    const unsigned char* skin_r8,
    const unsigned char* parsing_rgba,
    const unsigned char* face_rgba,
    const unsigned char* brightness_lut,
    const unsigned char* contrast_lut,
    const float* hue_maps,
    const float* gamma,
    const float* matrices,
    unsigned char* output_rgba,
    Pf90004Uniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  unsigned int source_index =
      (unsigned int)y * uniforms.stride + (unsigned int)x * 4u;
  unsigned int mask_index =
      (unsigned int)y * uniforms.mask_row_bytes + (unsigned int)x;
  unsigned int parsing_index =
      (unsigned int)y * uniforms.parsing_row_bytes + (unsigned int)x * 4u;
  unsigned int face_index =
      (unsigned int)y * uniforms.face_row_bytes + (unsigned int)x * 4u;

  unsigned char source_r = source_rgba[source_index + 0u];
  unsigned char source_g = source_rgba[source_index + 1u];
  unsigned char source_b = source_rgba[source_index + 2u];
  unsigned char source_a = source_rgba[source_index + 3u];
  output_rgba[source_index + 0u] = source_r;
  output_rgba[source_index + 1u] = source_g;
  output_rgba[source_index + 2u] = source_b;
  output_rgba[source_index + 3u] = source_a;

  float skin = p4_div((float)skin_r8[mask_index], 255.0f);
  float parsing_a = p4_div((float)parsing_rgba[parsing_index + 3u], 255.0f);
  float parsing_g = p4_div((float)parsing_rgba[parsing_index + 1u], 255.0f);
  float parsing_b = p4_div((float)parsing_rgba[parsing_index + 2u], 255.0f);
  float face_g = p4_div((float)face_rgba[face_index + 1u], 255.0f);
  float mask_sum = p4_mask_sum(parsing_a, parsing_g, parsing_b, face_g);
  float brightness_alpha = p4_add(mask_sum, skin);
  if (brightness_alpha > 1.0f) brightness_alpha = 1.0f;

  float r = p4_div((float)source_r, 255.0f);
  float g = p4_div((float)source_g, 255.0f);
  float b = p4_div((float)source_b, 255.0f);

  if (brightness_alpha > 0.001f) {
    float maximum = r > g ? r : g;
    maximum = maximum > b ? maximum : b;
    float minimum = r < g ? r : g;
    minimum = minimum < b ? minimum : b;
    float weight = p4_sub(
        1.0f,
        p4_mul(p4_add(maximum, minimum), 0.5f));

    unsigned char values[3] = {source_r, source_g, source_b};
    unsigned char adjusted_bytes[3] = {source_r, source_g, source_b};
    for (int channel = 0; channel < 3; ++channel) {
      if (uniforms.use_brightness) {
        adjusted_bytes[channel] = brightness_lut[adjusted_bytes[channel]];
      }
      if (uniforms.use_contrast) {
        adjusted_bytes[channel] = contrast_lut[adjusted_bytes[channel]];
      }
    }
    float source_float[3] = {r, g, b};
    for (int channel = 0; channel < 3; ++channel) {
      float adjusted = p4_add(
          source_float[channel],
          p4_mul(
              p4_sub(p4_div((float)adjusted_bytes[channel], 255.0f),
                     source_float[channel]),
              weight));
      float result = p4_add(
          source_float[channel],
          p4_mul(p4_sub(adjusted, source_float[channel]), brightness_alpha));
      output_rgba[source_index + (unsigned int)channel] =
          p4_round_byte(result);
    }
    values[0] = output_rgba[source_index + 0u];
    values[1] = output_rgba[source_index + 1u];
    values[2] = output_rgba[source_index + 2u];
    r = p4_div((float)values[0], 255.0f);
    g = p4_div((float)values[1], 255.0f);
    b = p4_div((float)values[2], 255.0f);
  }

  float hsl_alpha = skin > mask_sum ? skin : mask_sum;
  if (!(hsl_alpha > 0.001f)) return;

  float3 decoded = make_float3(
      p4_gamma(gamma, r, 1u),
      p4_gamma(gamma, g, 1u),
      p4_gamma(gamma, b, 1u));
  float3 toned = p4_tone(p4_matrix(decoded, matrices, 9u), 3u, gamma);
  float maximum = toned.x > toned.y ? toned.x : toned.y;
  maximum = maximum > toned.z ? maximum : toned.z;
  float minimum = toned.x < toned.y ? toned.x : toned.y;
  minimum = minimum < toned.z ? minimum : toned.z;
  float difference = p4_sub(maximum, minimum);
  float hue = 0.0f;
  if (difference != 0.0f) {
    if (maximum == toned.x) {
      hue = p4_div(p4_sub(toned.y, toned.z), difference);
    } else if (maximum == toned.y) {
      hue = p4_add(p4_div(p4_sub(toned.z, toned.x), difference), 2.0f);
    } else {
      hue = p4_add(p4_div(p4_sub(toned.x, toned.y), difference), 4.0f);
    }
  }
  if (hue < 0.0f) hue = p4_add(hue, 6.0f);

  float low = minimum;
  float high = maximum;
  if (difference != 0.0f) {
    float h = p4_mul(hue, 12.0f);
    unsigned int index = (unsigned int)__float2int_rz(h);
    if (index > 71u) index = 71u;
    float fraction = p4_sub(h, (float)index);
    float light = p4_palette(hue_maps, 148u, index, fraction);
    if (light != 0.0f) {
      float two_minus_difference = p4_sub(2.0f, difference);
      float biquad = p4_mul(
          p4_mul(two_minus_difference, difference),
          p4_sub(2.0f, p4_mul(two_minus_difference, difference)));
      light = p4_mul(p4_mul(biquad, p4_sub(2.0f, biquad)), light);
      float lo_mix = p4_add(
          p4_mul(p4_sub(1.0f, low), light), 1.0f);
      low = p4_mul(low, lo_mix);
      lo_mix = p4_add(p4_mul(p4_sub(1.0f, low), light), 1.0f);
      low = p4_mul(low, lo_mix);
      float hi_mix = p4_add(
          p4_mul(p4_sub(1.0f, high), light), 1.0f);
      high = p4_mul(high, hi_mix);
      hi_mix = p4_add(p4_mul(p4_sub(1.0f, high), light), 1.0f);
      high = p4_mul(high, hi_mix);
    }
    float saturation = p4_palette(hue_maps, 74u, index, fraction);
    if (saturation > 0.0f) {
      float s = p4_div(p4_sub(high, low), high);
      float m = p4_mul(high, 16.0f);
      if (m > 1.0f) m = 1.0f;
      float k = p4_mul(p4_sub(1.0f, low), p4_mul(saturation, 0.83333331f));
      k = p4_mul(k, p4_mul(p4_sub(2.0f, m), m));
      k = p4_mul(k, p4_sub(1.0f, s));
      k = p4_sub(1.0f, k);
      low = p4_sub(high, p4_mul(p4_div(s, k), high));
    } else {
      float average = p4_mul(p4_add(low, high), 0.5f);
      low = p4_add(low, p4_mul(p4_sub(low, average), saturation));
      high = p4_add(high, p4_mul(p4_sub(high, average), saturation));
    }
    hue = p4_palette(hue_maps, 0u, index, fraction);
  }

  float3 linear = p4_matrix(
      p4_tone(p4_to_rgb(hue, high, low), 2u, gamma),
      matrices,
      0u);
  float result_r = p4_gamma(gamma, linear.x, 0u);
  float result_g = p4_gamma(gamma, linear.y, 0u);
  float result_b = p4_gamma(gamma, linear.z, 0u);
  float source_channels[3] = {r, g, b};
  float result_channels[3] = {result_r, result_g, result_b};
  for (int channel = 0; channel < 3; ++channel) {
    float value = p4_add(
        source_channels[channel],
        p4_mul(p4_sub(result_channels[channel], source_channels[channel]),
               hsl_alpha));
    output_rgba[source_index + (unsigned int)channel] =
        p4_round_byte(value);
  }
  output_rgba[source_index + 3u] = source_a;
}
