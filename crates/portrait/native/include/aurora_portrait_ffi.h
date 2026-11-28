/*
 * Aurora Portrait stable C ABI.
 *
 * Mirrors crates/portrait/src/mobile_ffi.rs. All handles and tokens are opaque
 * uint64 values; all pixel memory is owned by the caller. Every entry point
 * catches Rust panics and returns a status code. No unwind crosses this ABI.
 *
 * Evidence level: DESIGNED / SYNTHETIC_OK on the Windows host. No GPU, NPU,
 * photo, or vendor-parity claim is implied by this header.
 */
#ifndef AURORA_PORTRAIT_FFI_H
#define AURORA_PORTRAIT_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AURORA_ABI_VERSION 1u
#define AURORA_MAX_STAGES 64u

/* Execution modes */
#define AURORA_MODE_SESSION_DEFAULT (-1)
#define AURORA_MODE_STRICT 0
#define AURORA_MODE_AUTO_FALLBACK 1

/* Pixel stage tags */
#define AURORA_STAGE_BRIGHTNESS 0u
#define AURORA_STAGE_WHITEN 1u
#define AURORA_STAGE_ROSY 2u
#define AURORA_STAGE_TEMPERATURE 3u
#define AURORA_STAGE_SATURATION 4u
#define AURORA_STAGE_LUT1D 5u
#define AURORA_STAGE_LUT3D 6u

/* Status codes (must match AuroraStatus in mobile_ffi.rs) */
#define AURORA_STATUS_OK 0
#define AURORA_STATUS_NULL_POINTER 1
#define AURORA_STATUS_INVALID_HANDLE 2
#define AURORA_STATUS_INVALID_ARGUMENT 3
#define AURORA_STATUS_BUFFER_TOO_SMALL 4
#define AURORA_STATUS_STRICT_REJECTED 5
#define AURORA_STATUS_CANCELLED 6
#define AURORA_STATUS_MODE_UNSUPPORTED 7
#define AURORA_STATUS_PANIC 8
#define AURORA_STATUS_INTERNAL 9
#define AURORA_STATUS_BUDGET_EXHAUSTED 10

typedef enum AuroraThermalState {
    AURORA_THERMAL_NOMINAL = 0,
    AURORA_THERMAL_FAIR = 1,
    AURORA_THERMAL_SERIOUS = 2,
    AURORA_THERMAL_CRITICAL = 3
} AuroraThermalState;

typedef enum AuroraMemoryPressure {
    AURORA_MEMORY_NORMAL = 0,
    AURORA_MEMORY_WARNING = 1,
    AURORA_MEMORY_CRITICAL = 2
} AuroraMemoryPressure;

typedef struct AuroraPixelStage {
    uint32_t kind;
    float a;
    float b;
    const uint8_t *lut_ptr; /* borrowed for the duration of the call only */
    uint32_t lut_len;
    uint32_t lut_size;      /* 3D LUT size; 0 for other stages */
    uint32_t reserved;
} AuroraPixelStage;

typedef struct AuroraFusionRequest {
    uint32_t width;
    uint32_t height;
    uint32_t stage_count;
    const AuroraPixelStage *stages;
    uint64_t cancel_token;  /* 0 means no cancellation token */
    int32_t mode;           /* AURORA_MODE_* */
    uint32_t reserved;
} AuroraFusionRequest;

typedef struct AuroraBudget {
    uint64_t max_vram_bytes;
    uint64_t max_staging_bytes;
    uint32_t max_shared_mem_bytes;
    uint32_t max_threads_per_group;
    uint32_t max_concurrent_dispatches;
    uint32_t reserved;
} AuroraBudget;

typedef struct AuroraReceipt {
    uint32_t fused_stages;
    uint64_t pixel_count;
    uint32_t memory_passes;
    uint32_t tiled_fallback;
    uint32_t backend;       /* 0 = CPU reference */
    int32_t status_code;
    uint32_t reserved;
} AuroraReceipt;

/* Session lifecycle */
int32_t aurora_session_create(int32_t mode, uint64_t *out_handle);
int32_t aurora_session_destroy(uint64_t handle);
int32_t aurora_session_set_mode(uint64_t handle, int32_t mode);

/* Cancellation tokens */
int32_t aurora_cancel_token_create(uint64_t *out_token);
int32_t aurora_cancel_token_cancel(uint64_t token);
int32_t aurora_cancel_token_is_cancelled(uint64_t token, uint32_t *out_flag);
int32_t aurora_cancel_token_destroy(uint64_t token);

/* Thermal / memory pressure notifications */
int32_t aurora_notify_thermal_state(uint64_t handle, int32_t state);
int32_t aurora_notify_memory_pressure(uint64_t handle, int32_t pressure);
int32_t aurora_budget_snapshot(uint64_t handle, AuroraBudget *out_budget);
int32_t aurora_last_receipt(uint64_t handle, AuroraReceipt *out_receipt);

/* Capability probe: call with NULL/0 to learn required size in out_len. */
int32_t aurora_capabilities_json(uint64_t handle, char *out_json,
                                 size_t capacity, size_t *out_len);

/* Single-pass fused RGBA8 processing. src and dst must each be
 * width * height * 4 bytes and must not alias. */
int32_t aurora_process_rgba8(uint64_t handle,
                             const AuroraFusionRequest *request,
                             const uint8_t *src_rgba8,
                             uint8_t *dst_rgba8);

/* Static NUL-terminated status name; never freed. */
const char *aurora_status_name(int32_t code);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AURORA_PORTRAIT_FFI_H */
