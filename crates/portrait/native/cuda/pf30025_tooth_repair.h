#pragma once

#include <cstdint>

struct Pf30025ToothRepairUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t stride;
  float degree;
  uint32_t roi_x;
  uint32_t roi_y;
  uint32_t roi_w;
  uint32_t roi_h;
};
