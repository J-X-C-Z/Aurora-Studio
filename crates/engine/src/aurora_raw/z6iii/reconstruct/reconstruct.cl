#pragma OPENCL FP_CONTRACT OFF

kernel void adaptive_gg(global const ushort *source, global ushort *destination,
                        int4 roi, int source_stride, int destination_stride,
                        float8 settings, global const float *unused_a, int4 ignored_a,
                        global const float *unused_b, int4 ignored_b) {
    const int row = get_global_id(1);
    const int column = 2 * get_global_id(0);
    if (row >= roi.w || column >= roi.z) return;
    const int green = 1 - (row & 1);
    const int point = (roi.y + row) * source_stride + roi.x + column;
    const int out = (roi.y + row) * destination_stride + roi.x + column;
    const float original = source[point + green];
    const float signal = fmax(original - settings.s5, 0.0f);
    ushort corrected = source[point + green];
    if (signal < 4.0f * settings.s0) {
        const int center = point + green;
        const float northwest = source[center - source_stride - 1];
        const float northeast = source[center - source_stride + 1];
        const float southwest = source[center + source_stride - 1];
        const float southeast = source[center + source_stride + 1];
        const float base = fma(signal, 0.25f, settings.s1);
        const float w_nw = 1.0f / (base + fabs(original - northwest));
        const float w_ne = 1.0f / (base + fabs(original - northeast));
        const float w_sw = 1.0f / (base + fabs(original - southwest));
        const float w_se = 1.0f / (base + fabs(original - southeast));
        float color = fma(northwest, w_nw, northeast * w_ne);
        color = fma(southwest, w_sw, color);
        color = fma(southeast, w_se, color);
        const float sum = ((w_nw + w_ne) + w_sw) + w_se;
        const float dark = fma(color / sum, 0.5f, original * 0.5f);
        float output = dark;
        if (signal >= settings.s0) {
            const float mix = (signal - settings.s0) / (3.0f * settings.s0);
            output = fma(dark, 1.0f - mix, original * mix);
        }
        corrected = convert_ushort_rtz(output);
    }
    destination[out + green] = corrected;
    destination[out + 1 - green] = source[point + 1 - green];
}
// AURORA_PROGRAM
#pragma OPENCL FP_CONTRACT OFF
#ifndef DEBUG_DIRECTION
#define DEBUG_DIRECTION 0
#endif

static int tap(global const ushort *center, int stride, int row, int column) {
    return center[row * stride + column];
}

static int4 residual(global const ushort *center, int stride, int y, int x, int predicted) {
    return (int4)(tap(center, stride, y, x - 1), tap(center, stride, y, x + 1),
                  tap(center, stride, y - 1, x), tap(center, stride, y + 1, x)) - predicted;
}

static int choose(global const ushort *center, int stride, float gain, float black) {
    const int qgain = convert_int_rtz(gain * 1024.0f);
    const uint curvature_gain = convert_uint_rtz((gain * 1024.0f) * 1.6f);
    const int bias = convert_int_rtz(fma(-black, gain, black));
    const int center_prediction = convert_int_rtn((float)center[0] * gain) + bias;
    const int4 reference = residual(center, stride, 0, 0, center_prediction);
    uint horizontal = (abs(tap(center,stride,0,-2) + tap(center,stride,0,2) - 2*center[0]) * curvature_gain) >> 10;
    uint vertical = (abs(tap(center,stride,-2,0) + tap(center,stride,2,0) - 2*center[0]) * curvature_gain) >> 10;
    for (int sign = -1; sign <= 1; sign += 2) {
        for (int side = -2; side <= 2; side += 4) {
            horizontal += abs(tap(center,stride,sign,0) - tap(center,stride,sign,side));
            vertical += abs(tap(center,stride,0,sign) - tap(center,stride,side,sign));
        }
    }
    uint4 cost = (uint4)(horizontal, horizontal, vertical, vertical);
    for (int offset = -4; offset <= 4; offset += 8) {
        int hprediction = ((tap(center,stride,0,offset) * qgain) >> 10) + bias;
        int vprediction = ((tap(center,stride,offset,0) * qgain) >> 10) + bias;
        int4 across = residual(center,stride,0,offset,hprediction);
        int4 along = residual(center,stride,offset,0,vprediction);
        cost += abs((int4)(across.xy, along.zw) - reference);
        horizontal += abs(across.x - across.y);
        vertical += abs(along.z - along.w);
    }
    for (int y = -2; y <= 2; y += 2) {
        for (int x = -2; x <= 2; x += 2) {
            const int predicted = ((tap(center,stride,y,x) * qgain) >> 10) + bias;
            const int4 nearby = residual(center,stride,y,x,predicted);
            cost += abs(nearby - reference);
            horizontal += abs(nearby.x - nearby.y);
            vertical += abs(nearby.z - nearby.w);
        }
    }
    if (max(cost.x,cost.y) < min(cost.z,cost.w) && horizontal < vertical) return 1;
    if (max(cost.z,cost.w) < min(cost.x,cost.y) && vertical < horizontal) return 2;
    const uint curvature = abs(reference.x + reference.y - reference.z - reference.w) / 2;
    const int signal = tap(center,stride,0,-1) + tap(center,stride,0,1) + tap(center,stride,-1,0)
        + tap(center,stride,1,0) - convert_int_rtz(black * 4.0f);
    if (as_int(curvature * curvature) < signal) return 3;
    cost = (uint4)(0);
    for (int y = -4; y <= 4; y += 2) {
        for (int x = -4; x <= 4; x += 2) {
            const int predicted = ((tap(center,stride,y,x) * qgain) >> 10) + bias;
            cost += abs(residual(center,stride,y,x,predicted) - reference);
        }
    }
    if (max(cost.x,cost.y) < min(cost.z,cost.w)) return 1;
    if (max(cost.z,cost.w) < min(cost.x,cost.y)) return 2;
    return 3;
}

static float axis_weight(global const ushort *center, int along, int across, float gain, bool scaled) {
    const uint neighbors = abs(center[-along] - center[along]) + abs(center[-across + 2*along] - center[-across])
        + abs(center[across + 2*along] - center[across]) + abs(center[3*along] - center[along]);
    const uint difference = abs(center[2*along] - center[0]);
    if (scaled) return (float)(1 + neighbors + difference) * gain;
    return fma((float)difference, gain, (float)(1 + neighbors));
}

static float midpoint(global const ushort *center, int offset) { return (center[0] + center[offset] + 1) / 2; }

static float4 green_value(global const ushort *center, int stride, float gain, float floor_code, float black, int direction) {
    float weight, numer, denom;
    if (direction == 3) {
        const float neg_x = axis_weight(center, -1, stride, gain, false);
        const float pos_x = axis_weight(center, 1, stride, gain, false);
        const float neg_y = axis_weight(center, -stride, 1, gain, false);
        const float pos_y = axis_weight(center, stride, 1, gain, false);
        const float pair_x = pos_x * neg_x;
        const float pair_y = pos_y * neg_y;
        const float down = neg_y * pair_x;
        const float up = pos_y * pair_x;
        const float right = pair_y * neg_x;
        const float left = pair_y * pos_x;
        weight = down + ((right + left) + up);
        numer = fma(right, (float)center[1], fma(left,(float)center[-1], fma(up,(float)center[-stride],down*(float)center[stride])));
        denom = fma(right, midpoint(center,2),fma(left,midpoint(center,-2),fma(up,midpoint(center,-2*stride),down*midpoint(center,2*stride))));
    } else {
        const int along = direction == 1 ? 1 : stride;
        const int across = direction == 1 ? stride : 1;
        const float neg = axis_weight(center, -along, across, gain, false);
        const float pos = axis_weight(center, along, across, gain, direction == 1);
        weight = pos + neg;
        numer = fma(pos, (float)center[-along], neg * (float)center[along]);
        denom = fma(pos, midpoint(center,-2*along), neg * midpoint(center,2*along));
    }
    const float floor_value = weight * floor_code;
    const float top = fmax(fma(-weight, black, numer + floor_value), floor_value);
    const float bottom = fmax(fma(-weight, black, denom + floor_value), floor_value);
    const float ratio = top / bottom;
    return (float4)((ratio * (((float)center[0] - black) + floor_code) - floor_code) + black,
                    top, bottom, ratio);
}

kernel void bayer_stage1(global const ushort *source, global float4 *output, int4 roi,
                         int stride, int pitch_words, float2 gains, float floor_code, float black) {
    const int x = 2 * get_global_id(0);
    const int y = get_global_id(1);
    if (x + 1 >= roi.z || y >= roi.w) return;
    const int odd = y & 1;
    const int start = (roi.y + y) * stride + roi.x + x;
    const int destination = (roi.y + y) * (pitch_words / 4) + roi.x + x;
    global const ushort *center = source + start + odd;
    const float gain = odd ? gains.y : gains.x;
    const int direction = choose(center, stride, gain, black);
    const float4 terms = green_value(center, stride, gain, floor_code, black, direction);
    const float green = terms.x;
    const float tag = DEBUG_DIRECTION ? (float)direction : 0.0f;
    if (odd) {
        output[destination] = (float4)(0, source[start], 0, 0);
        output[destination+1] = DEBUG_DIRECTION == 2 ? terms : (float4)(0, green, center[0], tag);
    } else {
        output[destination] = DEBUG_DIRECTION == 2 ? terms : (float4)(center[0], green, 0, tag);
        output[destination+1] = (float4)(0, source[start+1], 0, 0);
    }
}
// AURORA_PROGRAM
#pragma OPENCL FP_CONTRACT OFF

static float channel(float4 pixel, int component) {
    return component == 0 ? pixel.x : pixel.z;
}

static float color_cost(float difference, float denominator) {
    return 0.03f + fabs(difference) / fmax(1.0f, denominator);
}

kernel void reconstruct_diagonal(global const float4 *source, global float4 *target,
                                 int width, int height, float floor_code) {
    const int column = 2 * get_global_id(0) + 2;
    const int row = get_global_id(1) + 2;
    if (column + 1 >= width - 2 || row >= height - 2) return;
    const int odd = row & 1;
    const int position = row * width + column + odd;
    global const float4 *center = source + position;
    const int missing = odd ? 0 : 2;
    const int native = 2 - missing;
    const float4 nw = center[-width-1], ne = center[-width+1];
    const float4 sw = center[width-1], se = center[width+1];
    const float cnw = channel(nw, missing), cne = channel(ne, missing);
    const float csw = channel(sw, missing), cse = channel(se, missing);
    const bool main = fabs(cnw-cse) < fabs(csw-cne);
    const bool chroma = (((cnw+cne)+csw)+cse)*0.75f < (((nw.y+se.y)+ne.y)+sw.y);
    const int first = main ? -width-1 : width-1;
    const int second = -first;
    const float c_first = channel(center[first], missing), c_second = channel(center[second], missing);
    const float g_first = center[first].y, g_second = center[second].y;
    const float native_center = channel(center[0], native);
    float w_first = (1.0f+fabs(g_first-center[0].y)) + fabs(channel(center[2*first],native)-native_center);
    float w_second = (1.0f+fabs(g_second-center[0].y)) + fabs(channel(center[2*second],native)-native_center);
    if (chroma) {
        w_first *= color_cost(c_first-g_first, (1.0f+c_first)+g_first);
        w_second *= color_cost(c_second-g_second, (1.0f+c_second)+g_second);
    }
    const float weight = w_first+w_second;
    const float numerator = fma(weight,floor_code,fma(w_second,c_first,w_first*c_second));
    const float denominator = fma(weight,floor_code,fma(w_second,g_first,w_first*g_second));
    const float ratio = numerator/fmax(weight*floor_code,denominator);
#ifdef DIAGONAL_FUSED_TAIL
    const float value = fma(ratio,floor_code+center[0].y,-floor_code);
#else
    const float value = ratio*(floor_code+center[0].y)-floor_code;
#endif
    float4 result = center[0];
    if (odd) result.x = value;
    else result.z = value;
    target[position] = result;
    target[position + (odd ? -1 : 1)] = center[odd ? -1 : 1];
}

static float at_green(global const float4 *center, int width, int component, float floor_code) {
    const float4 up = center[-width], down = center[width], left = center[-1], right = center[1];
    const float cu = channel(up,component), cd = channel(down,component);
    const float cl = channel(left,component), cr = channel(right,component);
    const bool chroma = (((cu+cl)+cr)+cd)*0.75f < (((up.y+down.y)+right.y)+left.y);
    const float vertical = 1.0f + fabs(up.y-down.y), horizontal = 1.0f + fabs(left.y-right.y);
    const float green = center[0].y;
    float wu = vertical + fabs(center[-2*width].y-green), wd = vertical + fabs(center[2*width].y-green);
    float wl = horizontal + fabs(center[-2].y-green), wr = horizontal + fabs(center[2].y-green);
    if (chroma) {
        wu *= color_cost(up.y-cu, (1.0f+up.y)+cu);
        wd *= color_cost(down.y-cd, (1.0f+down.y)+cd);
        wl *= color_cost(left.y-cl, (1.0f+left.y)+cl);
        wr *= color_cost(right.y-cr, (1.0f+right.y)+cr);
    }
    wu = 1.0f/wu; wd = 1.0f/wd; wl = 1.0f/wl; wr = 1.0f/wr;
    const float weight = ((wu+wd)+wr)+wl;
    const float numerator = fma(weight,floor_code,fma(wr,cr,fma(wl,cl,fma(wu,cu,wd*cd))));
    const float denominator = fma(weight,floor_code,fma(wr,right.y,fma(wl,left.y,fma(wu,up.y,wd*down.y))));
    return fma(numerator/fmax(weight*floor_code,denominator),floor_code+green,-floor_code);
}

kernel void reconstruct_green(global const float4 *source, global float4 *target,
                              int width, int height, float floor_code) {
    const int column = 2 * get_global_id(0) + 4;
    const int row = get_global_id(1) + 4;
    if (column + 1 >= width - 4 || row >= height - 4) return;
    const int odd = row & 1;
    const int position = row * width + column + (1-odd);
    global const float4 *center = source + position;
    target[position] = (float4)(at_green(center,width,0,floor_code),center[0].y,
                               at_green(center,width,2,floor_code),center[0].w);
    target[position + (odd ? 1 : -1)] = center[odd ? 1 : -1];
}
// AURORA_PROGRAM
#pragma OPENCL FP_CONTRACT OFF

static float guide_value(float4 pixel, float4 weights) {
    const float green = (pixel.y + pixel.y) * weights.y;
    return fmax(1.0f, fma(pixel.z, weights.z, fma(pixel.x, weights.x, green)) + 1.0f);
}

kernel void gg_compensate(global const float4 *input, global float4 *output,
                          int width, int margin, int output_width, int phase,
                          float4 weights, float2 params) {
    const int column = get_global_id(0) + margin;
    const int row = get_global_id(1) + margin;
    global const float4 *center = input + row * width + column;
    float peak = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if ((row + dy + column + dx + phase) & 1) {
                peak = fmax(peak, center[dy * width + dx].y * weights.y);
            }
        }
    }
    float4 result = *center;
    if (peak < 4.0f * params.y) {
        float rolloff = 1.0f;
        if (peak >= params.y) {
            rolloff = fma(params.y, 4.0f, -peak) * (1.0f / (3.0f * params.y));
        }
        const float middle = guide_value(result, weights);
        const float nw = guide_value(center[-width-1], weights);
        const float ne = guide_value(center[-width+1], weights);
        const float sw = guide_value(center[width-1], weights);
        const float se = guide_value(center[width+1], weights);
        const bool main = fabs(fma(middle, 2.0f, -nw) - se) < fabs(fma(middle, 2.0f, -ne) - sw);
        const float diag = main ? nw + se : ne + sw;
        const float delta = fma(fma(middle, 2.0f, diag), -0.25f, middle) * 20.0f;
        const float limit = fmin(middle, params.x);
        const float limited = (delta * limit) / (limit + fabs(delta));
        result *= fma(limited * rolloff, 0.75f, middle) / middle;
    }
    output[(row - margin) * output_width + column - margin] = result;
}
// AURORA_PROGRAM
#pragma OPENCL FP_CONTRACT OFF

static float median_chroma(global const float4 *center, int width, bool blue) {
    float values[25];
    int count = 0;
    for (int row = -2; row <= 2; ++row) {
        for (int col = -2; col <= 2; ++col) {
            const float4 pixel = center[row * width + col];
            const float chroma = (blue ? pixel.z : pixel.x) - pixel.y;
            int index = count;
            while (index > 0 && values[index - 1] > chroma) {
                values[index] = values[index - 1];
                --index;
            }
            values[index] = chroma;
            ++count;
        }
    }
    return values[12];
}

static float blend(float original, float green, float chroma, float factor, float gain) {
    const float z = chroma * factor;
    const float gate = fmax(0.0f, fma(-z, z, 1.0f));
    const float weight = (gate * gate) * gain;
    return fma(weight, chroma + green, (1.0f - weight) * original);
}

kernel void anti_alias(global const float4 *input, global float4 *output, int width, float2 params) {
    const int column = get_global_id(0), row = get_global_id(1);
    global const float4 *center = input + (row + 2) * width + column + 2;
    const float4 pixel = *center;
    const float factor = params.x / fmax(1.0f, pixel.y + 1.0f);
    output[row * (width - 4) + column] = (float4)(
        blend(pixel.x, pixel.y, median_chroma(center, width, false), factor, params.y),
        pixel.y,
        blend(pixel.z, pixel.y, median_chroma(center, width, true), factor, params.y),
        pixel.w);
}
