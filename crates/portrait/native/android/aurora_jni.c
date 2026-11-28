/*
 * Aurora Portrait Android JNI bridge.
 *
 * Implements the native methods declared by AuroraPortraitBridge.kt. Owns no
 * pixel buffers: Java byte arrays are pinned only for the duration of a call
 * and copied back to the caller's dst array. Handles and tokens are uint64
 * values from the Rust C ABI. ADPF is dynamically loaded so API < 31 devices
 * keep working with thermal/memory notifications only.
 *
 * Evidence level: DESIGNED. This file does not prove GPU/NPU execution.
 */
#include <jni.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>
#include "../include/aurora_portrait_ffi.h"

#define AURORA_JNI_OK 0

/* ---- stage stream parser (little-endian, Kotlin DataOutputStream) ---- */

typedef struct {
    uint32_t kind;
    float a;
    float b;
    const uint8_t *lut;
    int32_t lut_len;
    uint32_t lut_size;
} aurora_stage_native;

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static float rd_f32(const uint8_t *p) {
    uint32_t bits = rd_u32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Returns stage count or negative status on malformed input. */
static int parse_stages(JNIEnv *env, jbyteArray packed, uint32_t **out_integers,
                        aurora_stage_native **out_stages, AuroraPixelStage **out_c_stages) {
    if (packed == NULL) return AURORA_STATUS_NULL_POINTER;
    const jsize total = (*env)->GetArrayLength(env, packed);
    if (total < 4) return AURORA_STATUS_INVALID_ARGUMENT;
    uint8_t *raw = (uint8_t *)malloc((size_t)total);
    if (raw == NULL) return AURORA_STATUS_INTERNAL;
    (*env)->GetByteArrayRegion(env, packed, 0, total, (jbyte *)raw);
    if ((*env)->ExceptionCheck(env)) { free(raw); return AURORA_STATUS_INTERNAL; }
    if ((*env)->GetArrayLength(env, packed) < 4) { free(raw); return AURORA_STATUS_INVALID_ARGUMENT; }

    const uint32_t count = rd_u32(raw);
    if (count == 0 || count > AURORA_MAX_STAGES) { free(raw); return AURORA_STATUS_INVALID_ARGUMENT; }
    size_t need = 4;
    for (uint32_t i = 0; i < count; ++i) {
        if (need + 16 > (size_t)total) { free(raw); return AURORA_STATUS_INVALID_ARGUMENT; }
        uint32_t lut_len = rd_u32(raw + need + 12);
        if ((size_t)lut_len > (size_t)total - need - 16) { free(raw); return AURORA_STATUS_INVALID_ARGUMENT; }
        need += 16 + lut_len;
    }

    aurora_stage_native *native = (aurora_stage_native *)calloc(count, sizeof(*native));
    AuroraPixelStage *c_stages = (AuroraPixelStage *)calloc(count, sizeof(*c_stages));
    if (native == NULL || c_stages == NULL) {
        free(raw); free(native); free(c_stages); return AURORA_STATUS_INTERNAL;
    }
    size_t offset = 4;
    for (uint32_t i = 0; i < count; ++i) {
        native[i].kind = rd_u32(raw + offset);
        native[i].a = rd_f32(raw + offset + 4);
        native[i].b = rd_f32(raw + offset + 8);
        native[i].lut_len = (int32_t)rd_u32(raw + offset + 12);
        native[i].lut_size = rd_u32(raw + offset + 16);
        offset += 16;
        native[i].lut = native[i].lut_len > 0 ? (raw + offset) : NULL;
        offset += (size_t)native[i].lut_len;

        c_stages[i].kind = native[i].kind;
        c_stages[i].a = native[i].a;
        c_stages[i].b = native[i].b;
        c_stages[i].lut_ptr = native[i].lut;
        c_stages[i].lut_len = (uint32_t)native[i].lut_len;
        c_stages[i].lut_size = native[i].lut_size;
        c_stages[i].reserved = 0;
    }
    /* `raw` remains owned by the caller until after aurora_process_rgba8. */
    *out_integers = (uint32_t *)raw;
    *out_stages = native;
    *out_c_stages = c_stages;
    return (int)count;
}

/* ---- ADPF dynamic loader (API 31+) ---- */

typedef struct APerformanceHintManager APerformanceHintManager;
typedef struct APerformanceHintSession APerformanceHintSession;

typedef APerformanceHintManager *(*PFN_AHINT_GET_MANAGER)(void);
typedef APerformanceHintSession *(*PFN_AHINT_CREATE_SESSION)(
    APerformanceHintManager *, const int32_t *, size_t, int64_t);
typedef void (*PFN_AHINT_UPDATE_TARGET)(APerformanceHintSession *, int64_t);
typedef void (*PFN_AHINT_REPORT_WORK)(APerformanceHintSession *, bool);
typedef void (*PFN_AHINT_CLOSE)(APerformanceHintSession *);

typedef struct {
    void *lib;
    APerformanceHintManager *manager;
    APerformanceHintSession *session;
    PFN_AHINT_UPDATE_TARGET update_target;
    PFN_AHINT_REPORT_WORK report_work;
    PFN_AHINT_CLOSE close_session;
} aurora_adpf;

static void *load_symbol(void *lib, const char *name) {
    return lib == NULL ? NULL : dlsym(lib, name);
}

/* ---- JNI exports ---- */

JNIEXPORT jlong JNICALL
Java_com_aurora_portrait_AuroraPortraitSession_nativeCreate(JNIEnv *env, jclass cls, jint mode) {
    (void)env; (void)cls;
    uint64_t handle = 0;
    if (aurora_session_create((int32_t)mode, &handle) != AURORA_STATUS_OK) return 0;
    return (jlong)handle;
}

JNIEXPORT void JNICALL
Java_com_aurora_portrait_AuroraPortraitSession_nativeDestroy(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    aurora_session_destroy((uint64_t)handle);
}

JNIEXPORT jint JNICALL
Java_com_aurora_portrait_AuroraPortraitSession_nativeCreateToken(JNIEnv *env, jclass cls, jlongArray out) {
    (void)cls;
    if (out == NULL || (*env)->GetArrayLength(env, out) < 1) return AURORA_STATUS_NULL_POINTER;
    uint64_t token = 0;
    int32_t code = aurora_cancel_token_create(&token);
    if (code == AURORA_STATUS_OK) {
        jlong value = (jlong)token;
        (*env)->SetLongArrayRegion(env, out, 0, 1, &value);
    }
    return code;
}

JNIEXPORT void JNICALL
Java_com_aurora_portrait_AuroraPortraitSession_nativeCancel(JNIEnv *env, jclass cls, jlong token) {
    (void)env; (void)cls;
    aurora_cancel_token_cancel((uint64_t)token);
}

JNIEXPORT void JNICALL
Java_com_aurora_portrait_AuroraPortraitSession_nativeIsCancelled(
    JNIEnv *env, jclass cls, jlong token, jintArray out) {
    (void)cls;
    if (out == NULL || (*env)->GetArrayLength(env, out) < 1) return;
    uint32_t flag = 0;
    if (aurora_cancel_token_is_cancelled((uint64_t)token, &flag) == AURORA_STATUS_OK) {
        jint value = (jint)flag;
        (*env)->SetIntArrayRegion(env, out, 0, 1, &value);
    }
}

JNIEXPORT void JNICALL
Java_com_aurora_portrait_AuroraPortraitSession_nativeDestroyToken(
    JNIEnv *env, jclass cls, jlong token) {
    (void)env; (void)cls;
    aurora_cancel_token_destroy((uint64_t)token);
}

JNIEXPORT void JNICALL
Java_com_aurora_portrait_AuroraPortraitSession_nativeNotifyThermal(
    JNIEnv *env, jclass cls, jlong handle, jint status, jfloat headroom, jboolean lowMemory) {
    (void)env; (void)cls;
    /* AThermalStatus already matches AuroraThermalState 0..3 for the first four
     * statuses; anything hotter is clamped to critical. */
    int32_t state = status <= 3 ? status : 3;
    if (status < 0) state = 0;
    aurora_notify_thermal_state((uint64_t)handle, state);
    int32_t pressure = lowMemory == JNI_TRUE ? 2 : (headroom < 0.2f ? 1 : 0);
    aurora_notify_memory_pressure((uint64_t)handle, pressure);
}

JNIEXPORT void JNICALL
Java_com_aurora_portrait_AuroraPortraitSession_nativeNotifyPressure(
    JNIEnv *env, jclass cls, jlong handle, jint pressure) {
    (void)env; (void)cls;
    aurora_notify_memory_pressure((uint64_t)handle, (int32_t)pressure);
}

JNIEXPORT jint JNICALL
Java_com_aurora_portrait_AuroraPortraitSession_nativeProcessRgba8(
    JNIEnv *env, jclass cls, jlong handle, jint width, jint height,
    jbyteArray packed, jbyteArray src, jbyteArray dst, jint mode, jlong token) {
    (void)cls;
    uint32_t *raw = NULL;
    aurora_stage_native *native_stages = NULL;
    AuroraPixelStage *stages = NULL;
    int count = parse_stages(env, packed, &raw, &native_stages, &stages);
    if (count < 0) return count;
    if (src == NULL || dst == NULL) {
        free(raw); free(native_stages); free(stages);
        return AURORA_STATUS_NULL_POINTER;
    }
    const jsize expected = width * height * 4;
    if (expected <= 0 ||
        (*env)->GetArrayLength(env, src) != expected ||
        (*env)->GetArrayLength(env, dst) != expected) {
        free(raw); free(native_stages); free(stages);
        return AURORA_STATUS_INVALID_ARGUMENT;
    }

    jbyte *src_ptr = (*env)->GetByteArrayElements(env, src, NULL);
    jbyte *dst_ptr = (*env)->GetByteArrayElements(env, dst, NULL);
    int32_t code = AURORA_STATUS_INTERNAL;
    if (src_ptr != NULL && dst_ptr != NULL) {
        AuroraFusionRequest request;
        memset(&request, 0, sizeof(request));
        request.width = (uint32_t)width;
        request.height = (uint32_t)height;
        request.stage_count = (uint32_t)count;
        request.stages = stages;
        request.cancel_token = (uint64_t)token;
        request.mode = (int32_t)mode;
        code = aurora_process_rgba8((uint64_t)handle, &request,
                                    (const uint8_t *)src_ptr, (uint8_t *)dst_ptr);
    }
    if (dst_ptr != NULL) {
        (*env)->ReleaseByteArrayElements(env, dst, dst_ptr, code == AURORA_STATUS_OK ? 0 : JNI_ABORT);
    }
    if (src_ptr != NULL) (*env)->ReleaseByteArrayElements(env, src, src_ptr, JNI_ABORT);
    free(raw); free(native_stages); free(stages);
    return code;
}

JNIEXPORT jint JNICALL
Java_com_aurora_portrait_AuroraPortraitSession_nativeReceipt(
    JNIEnv *env, jclass cls, jlong handle, jlongArray out) {
    (void)cls;
    if (out == NULL || (*env)->GetArrayLength(env, out) < 7) return AURORA_STATUS_NULL_POINTER;
    AuroraReceipt receipt;
    memset(&receipt, 0, sizeof(receipt));
    int32_t code = aurora_last_receipt((uint64_t)handle, &receipt);
    if (code == AURORA_STATUS_OK) {
        jlong values[7] = {
            (jlong)receipt.fused_stages, (jlong)receipt.pixel_count,
            (jlong)receipt.memory_passes, (jlong)receipt.tiled_fallback,
            (jlong)receipt.backend, (jlong)receipt.status_code, 0,
        };
        (*env)->SetLongArrayRegion(env, out, 0, 7, values);
    }
    return code;
}

JNIEXPORT jint JNICALL
Java_com_aurora_portrait_AuroraPortraitSession_nativeCapabilitiesJson(
    JNIEnv *env, jclass cls, jlong handle, jbyteArray out, jint capacity, jintArray outLen) {
    (void)cls;
    size_t needed = 0;
    if (out == NULL) {
        int32_t code = aurora_capabilities_json((uint64_t)handle, NULL, 0, &needed);
        if (outLen != NULL && needed <= INT32_MAX) {
            jint value = (jint)needed;
            (*env)->SetIntArrayRegion(env, outLen, 0, 1, &value);
        }
        return code;
    }
    jsize length = (*env)->GetArrayLength(env, out);
    if (length < 0 || (size_t)length > (size_t)capacity) return AURORA_STATUS_INVALID_ARGUMENT;
    jbyte *ptr = (*env)->GetByteArrayElements(env, out, NULL);
    if (ptr == NULL) return AURORA_STATUS_NULL_POINTER;
    int32_t code = aurora_capabilities_json((uint64_t)handle, (char *)ptr, (size_t)capacity, &needed);
    (*env)->ReleaseByteArrayElements(env, out, ptr, code == AURORA_STATUS_OK ? 0 : JNI_ABORT);
    if (outLen != NULL && needed <= INT32_MAX) {
        jint value = (jint)needed;
        (*env)->SetIntArrayRegion(env, outLen, 0, 1, &value);
    }
    return code;
}

/* ---- ADPF hint lifecycle ---- */

JNIEXPORT jlong JNICALL
Java_com_aurora_portrait_AuroraPortraitSession_nativeAdpfCreate(JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    void *lib = dlopen("libadaptiveloadhint.so", RTLD_NOW);
    if (lib == NULL) return 0;
    PFN_AHINT_GET_MANAGER get_manager =
        (PFN_AHINT_GET_MANAGER)load_symbol(lib, "APerformanceHintManager_get");
    PFN_AHINT_CREATE_SESSION create_session =
        (PFN_AHINT_CREATE_SESSION)load_symbol(lib, "APerformanceHintManager_createSession");
    if (get_manager == NULL || create_session == NULL) { dlclose(lib); return 0; }
    aurora_adpf *adpf = (aurora_adpf *)calloc(1, sizeof(*adpf));
    if (adpf == NULL) { dlclose(lib); return 0; }
    adpf->lib = lib;
    adpf->manager = get_manager();
    if (adpf->manager == NULL) { dlclose(lib); free(adpf); return 0; }
    const int32_t tid = (int32_t)gettid();
    adpf->session = create_session(adpf->manager, &tid, 1, 16 * 1000 * 1000LL);
    adpf->update_target =
        (PFN_AHINT_UPDATE_TARGET)load_symbol(lib, "APerformanceHintSession_updateTarget");
    adpf->report_work =
        (PFN_AHINT_REPORT_WORK)load_symbol(lib, "APerformanceHintSession_reportActualWorkDuration");
    adpf->close_session =
        (PFN_AHINT_CLOSE)load_symbol(lib, "APerformanceHintSession_close");
    if (adpf->session == NULL) { dlclose(lib); free(adpf); return 0; }
    return (jlong)(uintptr_t)adpf;
}

JNIEXPORT void JNICALL
Java_com_aurora_portrait_AuroraPortraitSession_nativeAdpfUpdate(
    JNIEnv *env, jclass cls, jlong handle, jint status, jfloat headroom) {
    (void)env; (void)cls;
    aurora_adpf *adpf = (aurora_adpf *)(uintptr_t)handle;
    if (adpf == NULL || adpf->update_target == NULL) return;
    /* Conservative target: 16ms nominal, shortened only at high headroom. */
    int64_t target_ns = 16 * 1000 * 1000LL;
    if (status == 0 && headroom >= 0.7f) target_ns = 12 * 1000 * 1000LL;
    adpf->update_target(adpf->session, target_ns);
    if (adpf->report_work != NULL) adpf->report_work(adpf->session, true);
}

JNIEXPORT void JNICALL
Java_com_aurora_portrait_AuroraPortraitSession_nativeAdpfDestroy(
    JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    aurora_adpf *adpf = (aurora_adpf *)(uintptr_t)handle;
    if (adpf == NULL) return;
    if (adpf->close_session != NULL && adpf->session != NULL) adpf->close_session(adpf->session);
    if (adpf->lib != NULL) dlclose(adpf->lib);
    free(adpf);
}
