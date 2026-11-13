#include "teeth_lut.h"

// Source-order translation of kernels::face_detail::{teeth_whitening, teeth_lut_sample}.
// Compile with -fmad=false. The recovered store rule is not plain rounding: it
// compares the fraction against a 256-entry threshold table, which is kept in
// double precision exactly as the Rust oracle does.

__device__ __constant__ double kStoreThresholds[256] = {
    0.5605493055190891, 0.556803893763572, 0.553058541379869, 0.5493133077397943,
    0.545567836612463, 0.5418230779469013, 0.5380773693323135, 0.5330834276974201,
    0.5293377190828323, 0.5255929604172707, 0.5218472518026829, 0.5181024931371212,
    0.5143567845225334, 0.5106110759079456, 0.5056180842220783, 0.5018733255565166,
    0.5605487674474716, 0.5568040087819099, 0.5530592501163483, 0.5493125915527344,
    0.5455678328871727, 0.5405748412013054, 0.5368281826376915, 0.5330834239721298,
    0.5293386653065681, 0.5255920067429543, 0.5218472480773926, 0.5181024894118309,
    0.514355830848217, 0.5093628391623497, 0.505618080496788, 0.5018733218312263,
    0.5605516135692596, 0.5568049550056458, 0.5530582964420319, 0.549311637878418,
    0.5443186461925507, 0.5405719876289368, 0.5368253290653229, 0.5330862700939178,
    0.529339611530304, 0.5255929529666901, 0.5218462944030762, 0.5168533027172089,
    0.513106644153595, 0.5093637853860855, 0.505620926618576, 0.5018742680549622,
    0.5605506598949432, 0.5568040013313293, 0.551811009645462, 0.5480643510818481,
    0.5443176925182343, 0.5405748337507248, 0.5368281751871109, 0.5330853164196014,
    0.5293386578559875, 0.5255919992923737, 0.5205990076065063, 0.5168523490428925,
    0.5131056904792786, 0.5093666315078735, 0.5056199729442596, 0.5018733143806458,
    0.5605497062206268, 0.5555567145347595, 0.5518100559711456, 0.5480633974075317,
    0.5443167388439178, 0.540570080280304, 0.5368234217166901, 0.533084362745285,
    0.5280913710594177, 0.5243447124958038, 0.5205980539321899, 0.516851395368576,
    0.5131047368049622, 0.5093580782413483, 0.5056190192699432, 0.5018723607063293,
    0.559302419424057, 0.5555557608604431, 0.5518091022968292, 0.5480624437332153,
    0.5443157851696014, 0.5405767261981964, 0.5368300676345825, 0.5318370759487152,
    0.5280904173851013, 0.5243437588214874, 0.5205971002578735, 0.5168504416942596,
    0.5131037831306458, 0.5093647241592407, 0.5056180655956268, 0.5006250739097595,
    0.5593014657497406, 0.5555548071861267, 0.5518081486225128, 0.5480614900588989,
    0.5443224310874939, 0.5393294394016266, 0.5355827808380127, 0.5318361222743988,
    0.5280894637107849, 0.524342805147171, 0.5205961465835571, 0.5168570876121521,
    0.5131104290485382, 0.5081174373626709, 0.504370778799057, 0.5006241202354431,
    0.5593005120754242, 0.5555538535118103, 0.5518147945404053, 0.5480681359767914,
    0.5430675446987152, 0.5393284857273102, 0.5355818271636963, 0.5318351686000824,
    0.5280885100364685, 0.5243418514728546, 0.5206027925014496, 0.5168561339378357,
    0.5118631422519684, 0.5081164836883545, 0.5043698251247406, 0.5630462169647217,
    0.5592919588088989, 0.5555528998374939, 0.5505675077438354, 0.5468132495880127,
    0.5430741906166077, 0.5393199324607849, 0.5355808734893799, 0.5318266153335571,
    0.5280875563621521, 0.5243332982063293, 0.5193479061126709, 0.5156088471412659,
    0.5118545889854431, 0.5081155300140381, 0.5043612718582153, 0.5630452632904053,
    0.5592910051345825, 0.5543056130409241, 0.550566554069519, 0.5468122959136963,
    0.5430732369422913, 0.5393189787864685, 0.5355799198150635, 0.5318256616592407,
    0.5280866026878357, 0.5231012105941772, 0.5193469524383545, 0.5156078934669495,
    0.5118536353111267, 0.5081145763397217, 0.5043603181838989, 0.5617979764938354,
    0.5580589175224304, 0.5543046593666077, 0.5505656003952026, 0.5468113422393799,
    0.5430722832679749, 0.5393180251121521, 0.5355789661407471, 0.5305935740470886,
    0.5268393158912659, 0.5231002569198608, 0.5193459987640381, 0.5156069397926331,
    0.5118526816368103, 0.5081136226654053, 0.5043593645095825, 0.561797022819519,
    0.558057963848114, 0.5543037056922913, 0.5505646467208862, 0.5468103885650635,
    0.5430713295936584, 0.5393170714378357, 0.5343316793441772, 0.5305926203727722,
    0.5268383622169495, 0.5230993032455444, 0.5193450450897217, 0.5156059861183167,
    0.5118517279624939, 0.5068663358688354, 0.5031272768974304, 0.5617960691452026,
    0.5580570101737976, 0.5543027520179749, 0.5505636930465698, 0.5468094348907471,
    0.5418240427970886, 0.5380697846412659, 0.5343307256698608, 0.5305916666984558,
    0.5268374085426331, 0.523098349571228, 0.5193440914154053, 0.5156050324440002,
    0.5106044411659241, 0.506865382194519, 0.503126323223114, 0.5617951154708862,
    0.5580560564994812, 0.5543017983436584, 0.54931640625, 0.5455621480941772,
    0.5418230891227722, 0.5380688309669495, 0.5343297719955444, 0.5305907130241394,
    0.5268364548683167, 0.5230973958969116, 0.5180968046188354, 0.5143577456474304,
    0.5106034874916077, 0.5068644285202026, 0.5031253695487976, 0.5617941617965698,
    0.5580551028251648, 0.5530545115470886, 0.5493154525756836, 0.5455611944198608,
    0.5418221354484558, 0.5380678772926331, 0.534328818321228, 0.530589759349823,
    0.5268355011940002, 0.5218501091003418, 0.518095850944519, 0.514356791973114,
    0.5106025338172913, 0.5068634748458862, 0.5031244158744812, 0.560546875,
    0.556807816028595, 0.5530535578727722, 0.5493144989013672, 0.5455602407455444,
    0.5418211817741394, 0.5380669236183167, 0.5343278646469116, 0.5293424725532532,
    0.5255882143974304, 0.5218491554260254, 0.5180948972702026, 0.5143558382987976,
    0.5106015801429749, 0.5068625211715698, 0.5031234622001648, 255.0,
};

__device__ __forceinline__ float t_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float t_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float t_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float t_div(float a, float b) { return __fdiv_rn(a, b); }

__device__ __forceinline__ float clamp01(float value) {
  if (isnan(value)) return 0.0f;
  return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ __forceinline__ int clamp_int(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

// Rust store_unorm8: threshold table, ties go up only when fraction >= table.
__device__ __forceinline__ unsigned char store_unorm8(float value) {
  double v = (double)clamp01(value) * 255.0;
  double floor_v = floor(v);
  int index = (int)floor_v;
  double fraction = v - floor_v;
  int result = index + (fraction >= kStoreThresholds[index] ? 1 : 0);
  return (unsigned char)clamp_int(result, 0, 255);
}

// One quad fetch of the 512x512x3 atlas.
__device__ __forceinline__ void teeth_fetch(
    const unsigned char* table,
    float quad_x,
    float quad_y,
    float r,
    float g,
    float* out) {
  float sx = t_add(t_add(t_mul(quad_x, 0.125f), 0.0009765625f), t_mul(0.123046875f, r));
  float sy = t_add(t_add(t_mul(quad_y, 0.125f), 0.0009765625f), t_mul(0.123046875f, g));
  float xs = t_sub(t_mul(sx, 512.0f), 0.5f);
  float ys = t_sub(t_mul(sy, 512.0f), 0.5f);
  float xf = floorf(xs);
  float yf = floorf(ys);
  int xi = (int)xf;
  int yi = (int)yf;
  float ax = t_div(floorf(t_add(t_mul(t_sub(xs, xf), 256.0f), 0.5f)), 256.0f);
  float ay = t_div(floorf(t_add(t_mul(t_sub(ys, yf), 256.0f), 0.5f)), 256.0f);
  int x0 = clamp_int(xi, 0, 511);
  int x1 = clamp_int(xi + 1, 0, 511);
  int y0 = clamp_int(yi, 0, 511);
  int y1 = clamp_int(yi + 1, 0, 511);
  for (int channel = 0; channel < 3; ++channel) {
    float top = t_add(
        t_mul(t_div((float)table[(y0 * 512 + x0) * 3 + channel], 255.0f), t_sub(1.0f, ax)),
        t_mul(t_div((float)table[(y0 * 512 + x1) * 3 + channel], 255.0f), ax));
    float bottom = t_add(
        t_mul(t_div((float)table[(y1 * 512 + x0) * 3 + channel], 255.0f), t_sub(1.0f, ax)),
        t_mul(t_div((float)table[(y1 * 512 + x1) * 3 + channel], 255.0f), ax));
    out[channel] = t_add(t_mul(top, t_sub(1.0f, ay)), t_mul(bottom, ay));
  }
}

__device__ __forceinline__ void teeth_sample(
    const unsigned char* table,
    float r,
    float g,
    float b,
    float degree,
    unsigned char* out) {
  float blue = t_mul(b, 63.0f);
  float fb = floorf(blue);
  float cb = ceilf(blue);
  float q1y = floorf(t_div(fb, 8.0f));
  float q1x = t_sub(fb, t_mul(q1y, 8.0f));
  float q2y = floorf(t_div(cb, 8.0f));
  float q2x = t_sub(cb, t_mul(q2y, 8.0f));
  float c1[3];
  float c2[3];
  teeth_fetch(table, q1x, q1y, r, g, c1);
  teeth_fetch(table, q2x, q2y, r, g, c2);
  float fraction = t_sub(blue, fb);
  float source[3] = {r, g, b};
  for (int channel = 0; channel < 3; ++channel) {
    float sampled = t_add(t_mul(c1[channel], t_sub(1.0f, fraction)), t_mul(c2[channel], fraction));
    out[channel] = store_unorm8(
        t_add(t_mul(source[channel], t_sub(1.0f, degree)), t_mul(sampled, degree)));
  }
}

extern "C" __global__ void teeth_whiten_bright_kernel(
    const unsigned char* source_rgba,
    const unsigned char* mask_g,
    const unsigned char* whiten_lut,
    const unsigned char* bright_lut,
    unsigned char* out_rgba,
    TeethUniforms uniforms) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (int)uniforms.width || y >= (int)uniforms.height) return;

  unsigned int index = (unsigned int)y * uniforms.stride + (unsigned int)x * 4u;
  unsigned int mask_index = (unsigned int)y * uniforms.mask_row_bytes + (unsigned int)x;

  unsigned char source_r = source_rgba[index + 0u];
  unsigned char source_g = source_rgba[index + 1u];
  unsigned char source_b = source_rgba[index + 2u];
  unsigned char alpha = source_rgba[index + 3u];

  float mask = clamp01(t_div((float)mask_g[mask_index], 255.0f));
  float dz = fminf(fmaxf(uniforms.dz, 0.0f), 1.0f);
  float ea = fminf(fmaxf(uniforms.ea, 0.0f), 1.0f);
  float whiten_weight = t_div((float)store_unorm8(t_mul(mask, dz)), 255.0f);
  float bright_weight = t_div((float)store_unorm8(t_mul(mask, ea)), 255.0f);

  float r = t_div((float)source_r, 255.0f);
  float g = t_div((float)source_g, 255.0f);
  float b = t_div((float)source_b, 255.0f);

  if (whiten_weight > 0.001f || bright_weight > 0.001f) {
    unsigned char sampled[3];
    if (whiten_weight > 0.001f) {
      teeth_sample(whiten_lut, r, g, b, whiten_weight, sampled);
      r = t_div((float)sampled[0], 255.0f);
      g = t_div((float)sampled[1], 255.0f);
      b = t_div((float)sampled[2], 255.0f);
    }
    if (bright_weight > 0.001f) {
      teeth_sample(bright_lut, r, g, b, bright_weight, sampled);
      r = t_div((float)sampled[0], 255.0f);
      g = t_div((float)sampled[1], 255.0f);
      b = t_div((float)sampled[2], 255.0f);
    }
    source_r = store_unorm8(r);
    source_g = store_unorm8(g);
    source_b = store_unorm8(b);
  }

  out_rgba[index + 0u] = source_r;
  out_rgba[index + 1u] = source_g;
  out_rgba[index + 2u] = source_b;
  out_rgba[index + 3u] = alpha;
}
