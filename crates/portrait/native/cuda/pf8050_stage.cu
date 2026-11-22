// PF8050/PF8051 six-stage candidate geometry consumer.
//
// One thread preserves the recovered sequential update and candidate-commit
// order. Explicit round-to-nearest float operations and -fmad=false keep every
// mesh byte comparable with the Rust oracle.

namespace {

struct Vec2 {
  float x;
  float y;
};

struct Uniforms {
  float degrees[12];
  Vec2 center_left;
  Vec2 center_right;
  Vec2 mid_left;
  Vec2 mid_right;
  Vec2 axis1_left;
  Vec2 axis1_right;
  Vec2 axis2_left;
  Vec2 axis2_right;
  Vec2 face_center;
  Vec2 face_direction_point;
  float cos_left;
  float sin_left;
  float cos_right;
  float sin_right;
};

constexpr unsigned int kMeshPoints = 144u;
constexpr unsigned int kStageCount = 6u;

__device__ __forceinline__ float fadd(float left, float right) {
  return __fadd_rn(left, right);
}

__device__ __forceinline__ float fsub(float left, float right) {
  return __fsub_rn(left, right);
}

__device__ __forceinline__ float fmul(float left, float right) {
  return __fmul_rn(left, right);
}

__device__ __forceinline__ float fdiv(float left, float right) {
  return (float)__ddiv_rn((double)left, (double)right);
}

__device__ __forceinline__ float fsqrt(float value) {
  return (float)__dsqrt_rn((double)value);
}

__device__ __forceinline__ Vec2 add2(Vec2 left, Vec2 right) {
  return Vec2{fadd(left.x, right.x), fadd(left.y, right.y)};
}

__device__ __forceinline__ Vec2 sub2(Vec2 left, Vec2 right) {
  return Vec2{fsub(left.x, right.x), fsub(left.y, right.y)};
}

__device__ __forceinline__ Vec2 scale2(Vec2 point, float scale) {
  return Vec2{fmul(point.x, scale), fmul(point.y, scale)};
}

__device__ __forceinline__ float length2(Vec2 vector) {
  return fsqrt(fadd(fmul(vector.x, vector.x), fmul(vector.y, vector.y)));
}

__device__ __forceinline__ float minps(float left, float right) {
  if (isnan(left) || isnan(right)) return right;
  return right < left ? right : left;
}

__device__ __forceinline__ Vec2 read_mesh(const float* values, unsigned int index) {
  return Vec2{values[index * 2u], values[index * 2u + 1u]};
}

__device__ __forceinline__ void write_stage(
    float* output,
    const Vec2* mesh,
    unsigned int stage) {
  float* target = output + stage * kMeshPoints * 2u;
  for (unsigned int index = 0; index < kMeshPoints; ++index) {
    target[index * 2u] = mesh[index].x;
    target[index * 2u + 1u] = mesh[index].y;
  }
}

__device__ __forceinline__ void stage0_weights(
    const float* raw,
    float* weights) {
  for (unsigned int index = 0; index < 28u; ++index) {
    weights[index] = fmul(raw[index], 1.5f);
  }
  const float step = fdiv(1.0f, 7.0f);
  const float a = fmul(fsub(1.0f, 0.25f), step);
  const float b = fmul(step, fsub(1.0f, a));
  weights[0] = fmul(weights[0], b);
  weights[14] = fmul(weights[14], b);
  float rem = fsub(fsub(1.0f, a), a);
  float t = fadd(step, step);
  const unsigned int lefts[3] = {1u, 3u, 5u};
  for (unsigned int round = 0; round < 3u; ++round) {
    const unsigned int left = lefts[round];
    const unsigned int right = 28u - left;
    weights[left] = fmul(weights[left], t);
    const float sum = fadd(step, t);
    weights[right] = fmul(weights[right], fmul(t, rem));
    rem = fsub(rem, a);
    weights[left + 1u] = fmul(weights[left + 1u], sum);
    t = fadd(step, sum);
    weights[right - 1u] = fmul(weights[right - 1u], fmul(sum, rem));
    rem = fsub(rem, a);
  }
  t = fsub(t, step);
  rem = fadd(rem, a);
  weights[7] = fmul(weights[7], t);
  weights[21] = fmul(weights[21], fmul(rem, t));
  for (unsigned int index = 8u; index < 14u; ++index) {
    weights[index] = fmul(weights[index], t);
    const unsigned int right = 28u - index;
    weights[right] = fmul(weights[right], fmul(rem, t));
    t = fsub(t, step);
    rem = fadd(rem, a);
  }
}

__device__ __forceinline__ void stage0(
    const Vec2* a,
    Vec2* b,
    float p_left,
    float p_right,
    const Uniforms& uniforms,
    const float* raw_weights) {
  for (unsigned int index = 0; index < 15u; ++index) {
    const Vec2 left = a[1u + index];
    const Vec2 right = a[73u + index];
    const Vec2 left_delta = scale2(sub2(left, uniforms.center_left), p_left);
    const Vec2 right_delta = scale2(sub2(right, uniforms.center_right), p_right);
    b[1u + index] = add2(left, scale2(left_delta, 0.2f));
    b[73u + index] = add2(right, scale2(right_delta, 0.2f));
  }

  float weights[28];
  stage0_weights(raw_weights, weights);
  for (unsigned int index = 0; index < 28u; ++index) {
    const Vec2 left = a[16u + index];
    const Vec2 right = a[88u + index];
    const Vec2 left_delta =
        scale2(sub2(uniforms.mid_left, left), weights[index]);
    const Vec2 right_delta =
        scale2(sub2(uniforms.mid_right, right), weights[index]);
    b[16u + index] = sub2(left, scale2(left_delta, p_left));
    b[88u + index] = sub2(right, scale2(right_delta, p_right));
  }
}

__device__ __forceinline__ float point_polygon_test(
    const Vec2* polygon,
    unsigned int count,
    Vec2 point) {
  if (count == 0u) return -1.0f;
  const float px = point.x;
  const float py = point.y;
  bool inside = false;
  Vec2 previous = polygon[count - 1u];
  for (unsigned int index = 0; index < count; ++index) {
    const Vec2 current = polygon[index];
    if (py == current.y && px == current.x) return 0.0f;
    if (py == previous.y && py == current.y &&
        fminf(previous.x, current.x) <= px &&
        px <= fmaxf(previous.x, current.x)) {
      return 0.0f;
    }
    if ((current.y > py) != (previous.y > py)) {
      const double cross =
          ((double)current.x - (double)previous.x) *
              ((double)py - (double)previous.y) -
          ((double)current.y - (double)previous.y) *
              ((double)px - (double)previous.x);
      if (cross == 0.0) return 0.0f;
      if ((previous.y < current.y && cross < 0.0) ||
          (previous.y > current.y && cross > 0.0)) {
        inside = !inside;
      }
    }
    previous = current;
  }
  return inside ? 1.0f : -1.0f;
}

__device__ __forceinline__ void commit_contour_candidates(
    Vec2* b,
    float p_left,
    float p_right,
    const Uniforms& uniforms) {
  for (unsigned int index = 0; index < 15u; ++index) {
    const Vec2 left_delta = scale2(
        sub2(b[1u + index], uniforms.center_left), 0.1599999964237213f);
    const Vec2 right_delta = scale2(
        sub2(b[73u + index], uniforms.center_right), 0.1599999964237213f);
    const Vec2 candidate_left = add2(b[1u + index], scale2(left_delta, p_left));
    const Vec2 candidate_right = add2(b[73u + index], scale2(right_delta, p_right));
    const float left_gate = point_polygon_test(b + 16u, 28u, candidate_left);
    const float right_gate = point_polygon_test(b + 88u, 28u, candidate_right);
    if (left_gate == 1.0f) b[1u + index] = candidate_left;
    if (right_gate == 1.0f) b[73u + index] = candidate_right;
  }
}

__device__ __forceinline__ void stage1(
    Vec2* b,
    float p_left,
    float p_right,
    const Uniforms& uniforms) {
  float min_diff[4] = {10000.0f, 10000.0f, 0.0f, 0.0f};
  for (unsigned int index = 0; index < 28u; ++index) {
    const float diff0 =
        fsub(length2(sub2(uniforms.center_right, b[116u + index])),
             length2(sub2(uniforms.center_right, b[88u + index])));
    const float diff1 =
        fsub(length2(sub2(uniforms.center_left, b[44u + index])),
             length2(sub2(uniforms.center_left, b[16u + index])));
    min_diff[0] = minps(diff0, min_diff[0]);
    min_diff[1] = minps(diff1, min_diff[1]);
    min_diff[2] = minps(0.0f, min_diff[2]);
    min_diff[3] = minps(0.0f, min_diff[3]);
  }

  const Vec2 axis_left = scale2(uniforms.axis1_left, 0.5f);
  const Vec2 axis_right = scale2(uniforms.axis1_right, 0.5f);
  const float profile[15] = {
      1.0004600286483765f,
      0.9006100296974182f,
      0.7004799842834473f,
      0.5005099773406982f,
      0.40018001198768616f,
      0.300819993019104f,
      0.20074999332427979f,
      0.10027000308036804f,
      0.20054000616073608f,
      0.3002699911594391f,
      0.40081000328063965f,
      0.5002700090408325f,
      0.7004899978637695f,
      0.9007200002670288f,
      1.0004600286483765f,
  };
  float min_ratio[4] = {1.0f, 1.0f, 0.0f, 0.0f};
  for (unsigned int index = 0; index < 15u; ++index) {
    const float weighted_right = fmul(profile[index], p_right);
    const float weighted_left = fmul(profile[index], p_left);
    const float horizontal_right = fmul(axis_right.y, weighted_right);
    const float vertical_right = fmul(weighted_right, axis_right.x);
    const float horizontal_left = fmul(axis_left.y, weighted_left);
    const float vertical_left = fmul(weighted_left, axis_left.x);
    const float norm_right = fsqrt(fadd(
        fmul(horizontal_right, horizontal_right),
        fmul(vertical_right, vertical_right)));
    const float norm_left = fsqrt(fadd(
        fmul(horizontal_left, horizontal_left),
        fmul(vertical_left, vertical_left)));
    min_ratio[0] = minps(fdiv(min_diff[0], norm_right), min_ratio[0]);
    min_ratio[1] = minps(fdiv(min_diff[1], norm_left), min_ratio[1]);
  }

  const float coeff_left = fmul(p_left, min_ratio[1]);
  const float coeff_right = fmul(p_right, min_ratio[0]);
  const Vec2 head_left =
      scale2(axis_left, fmul(fmul(profile[0], -1.0f), coeff_left));
  const Vec2 head_right =
      scale2(axis_right, fmul(fmul(profile[0], -1.0f), coeff_right));
  b[16u] = add2(b[16u], head_left);
  b[88u] = add2(b[88u], head_right);
  const Vec2 tail_left =
      scale2(axis_left, fmul(fmul(profile[14], 1.0f), coeff_left));
  const Vec2 tail_right =
      scale2(axis_right, fmul(fmul(profile[14], 1.0f), coeff_right));
  b[30u] = add2(b[30u], tail_left);
  b[102u] = add2(b[102u], tail_right);

  for (unsigned int index = 1u; index < 14u; ++index) {
    const float sign = index < 7u ? -1.0f : 1.0f;
    const Vec2 left = scale2(
        scale2(scale2(axis_left, sign), profile[index]), coeff_left);
    const Vec2 right = scale2(
        scale2(scale2(axis_right, sign), profile[index]), coeff_right);
    b[16u + index] = add2(b[16u + index], left);
    b[44u - index] = add2(b[44u - index], left);
    b[88u + index] = add2(b[88u + index], right);
    b[116u - index] = add2(b[116u - index], right);
  }
  commit_contour_candidates(b, p_left, p_right, uniforms);
}

__device__ __forceinline__ void stage2(
    Vec2* b,
    float p_left,
    float p_right,
    const Uniforms& uniforms,
    const float* weights) {
  const Vec2 left_step = scale2(uniforms.axis2_left, p_left);
  const Vec2 right_step = scale2(uniforms.axis2_right, p_right);
  for (unsigned int index = 0; index < 28u; ++index) {
    const Vec2 left = scale2(left_step, weights[index]);
    const Vec2 right = scale2(right_step, weights[index]);
    if (index < 15u) {
      b[16u + index] = add2(b[16u + index], left);
      b[88u + index] = add2(b[88u + index], right);
    } else {
      b[16u + index] = sub2(b[16u + index], left);
      b[88u + index] = sub2(b[88u + index], right);
    }
  }
  commit_contour_candidates(b, p_left, p_right, uniforms);
}

__device__ __forceinline__ void stage3(
    Vec2* b,
    const Uniforms& uniforms) {
  const Vec2 centers[2] = {uniforms.center_left, uniforms.center_right};
  const float cosines[2] = {uniforms.cos_left, uniforms.cos_right};
  const float sines[2] = {uniforms.sin_left, uniforms.sin_right};
  for (unsigned int side = 0; side < 2u; ++side) {
    const unsigned int base = side * 72u;
    const Vec2 center = centers[side];
    const float cosine = cosines[side];
    const float sine = sines[side];
    for (unsigned int index = 1u; index < 44u; ++index) {
      const Vec2 point = b[base + index];
      const float dx = fsub(point.x, center.x);
      const float dy = fsub(point.y, center.y);
      b[base + index] = Vec2{
          fadd(center.x, fadd(fmul(dx, cosine), fmul(dy, sine))),
          fadd(center.y, fadd(fmul(fmul(-1.0f, dx), sine), fmul(dy, cosine)))};
    }
  }
}

__device__ __forceinline__ void distance_gate_min4(
    const Vec2* b,
    Vec2 center_left,
    Vec2 center_right,
    float* lanes) {
  const unsigned int ind14[14] = {
      0u, 1u, 2u, 3u, 11u, 12u, 13u, 14u, 27u, 26u, 25u, 17u, 16u, 15u};
  lanes[0] = 10000.0f;
  lanes[1] = 10000.0f;
  lanes[2] = 0.0f;
  lanes[3] = 0.0f;
  for (unsigned int order = 0; order < 14u; ++order) {
    const unsigned int index = ind14[order];
    const float diff0 =
        fsub(length2(sub2(center_left, b[44u + index])),
             length2(sub2(center_left, b[16u + index])));
    const float diff1 =
        fsub(length2(sub2(center_right, b[116u + index])),
             length2(sub2(center_right, b[88u + index])));
    lanes[0] = minps(diff0, lanes[0]);
    lanes[1] = minps(diff1, lanes[1]);
    lanes[2] = minps(0.0f, lanes[2]);
    lanes[3] = minps(0.0f, lanes[3]);
  }
}

__device__ __forceinline__ void stage4(
    Vec2* b,
    float p_left,
    float p_right,
    Vec2* center_left,
    Vec2* center_right) {
  const Vec2 axis = sub2(*center_right, *center_left);
  const float axis_len = length2(axis);
  float lanes[4];
  distance_gate_min4(b, *center_left, *center_right, lanes);
  const Vec2 v_left = Vec2{
      fdiv(fmul(fmul(-p_left, axis.x), lanes[0]), axis_len),
      fdiv(fmul(fmul(-p_left, axis.y), lanes[0]), axis_len)};
  const Vec2 v_right = Vec2{
      fdiv(fmul(fmul(p_right, axis.x), lanes[1]), axis_len),
      fdiv(fmul(fmul(p_right, axis.y), lanes[1]), axis_len)};
  for (unsigned int index = 0; index < 44u; ++index) {
    b[index] = add2(b[index], v_left);
  }
  for (unsigned int index = 72u; index < 116u; ++index) {
    b[index] = add2(b[index], v_right);
  }
  *center_left = add2(*center_left, v_left);
  *center_right = add2(*center_right, v_right);
}

__device__ __forceinline__ void stage5(
    Vec2* b,
    float p_left,
    float p_right,
    Vec2* center_left,
    Vec2* center_right,
    const Uniforms& uniforms) {
  const Vec2 direction =
      sub2(uniforms.face_direction_point, uniforms.face_center);
  const float direction_len = length2(direction);
  const Vec2 face_direction = Vec2{
      fdiv(direction.x, direction_len),
      fdiv(direction.y, direction_len)};
  float lanes[4];
  distance_gate_min4(b, *center_left, *center_right, lanes);
  const Vec2 amp_left = Vec2{
      fmul(fmul(p_left, face_direction.x), lanes[0]),
      fmul(fmul(p_left, face_direction.y), lanes[0])};
  const Vec2 amp_right = Vec2{
      fmul(fmul(p_right, face_direction.x), lanes[1]),
      fmul(fmul(p_right, face_direction.y), lanes[1])};
  for (unsigned int index = 0; index < 44u; ++index) {
    b[index] = add2(b[index], amp_left);
  }
  for (unsigned int index = 72u; index < 116u; ++index) {
    b[index] = add2(b[index], amp_right);
  }
  *center_left = add2(*center_left, amp_left);
  *center_right = add2(*center_right, amp_right);
}

}  // namespace

extern "C" __global__ void pf8050_six_stage_kernel(
    const float* mesh_a,
    const float* raw_weights,
    const float* stage2_weights,
    Uniforms uniforms,
    float* output) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  Vec2 a[kMeshPoints];
  Vec2 b[kMeshPoints];
  for (unsigned int index = 0; index < kMeshPoints; ++index) {
    a[index] = read_mesh(mesh_a, index);
    b[index] = a[index];
  }

  const float p0_left = fmul(fmul(uniforms.degrees[0], 0.25f), 0.2f);
  const float p0_right = fmul(fmul(uniforms.degrees[1], 0.25f), 0.2f);
  const float p1_left = fmul(uniforms.degrees[2], 0.12000010162591934f);
  const float p1_right = fmul(uniforms.degrees[3], 0.8000010251998901f);
  const float p2_left = fmul(uniforms.degrees[4], 0.06000009924173355f);
  const float p2_right = fmul(uniforms.degrees[5], 0.4500001072883606f);
  const float p4_left = fmul(uniforms.degrees[8], 0.4500001072883606f);
  const float p4_right = fmul(uniforms.degrees[9], 0.4500001072883606f);
  const float p5_left = fmul(uniforms.degrees[10], 0.4500001072883606f);
  const float p5_right = fmul(uniforms.degrees[11], 0.4500001072883606f);

  stage0(
      a,
      b,
      p0_left,
      p0_right,
      uniforms,
      raw_weights);
  write_stage(output, b, 0u);

  stage1(b, p1_left, p1_right, uniforms);
  write_stage(output, b, 1u);

  stage2(b, p2_left, p2_right, uniforms, stage2_weights);
  write_stage(output, b, 2u);

  stage3(b, uniforms);
  write_stage(output, b, 3u);

  Vec2 center_left = uniforms.center_left;
  Vec2 center_right = uniforms.center_right;
  stage4(b, p4_left, p4_right, &center_left, &center_right);
  write_stage(output, b, 4u);

  stage5(b, p5_left, p5_right, &center_left, &center_right, uniforms);
  write_stage(output, b, 5u);
}
