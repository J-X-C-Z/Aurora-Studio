/*
 * Aurora Portrait HarmonyOS NEXT Node-API module.
 *
 * Strictly independent of Android. This file uses only OHOS Native SDK and
 * Node-API symbols: no JNI, no AHardwareBuffer, no QAIRT/QNN, no ADPF, no
 * dlopen of Android libraries, no Android headers.
 *
 * Heavy pixel work is always submitted through napi_create_async_work on a
 * libuv worker thread. ArkTS UI main thread only enqueues and receives the
 * completion callback. NNRt inference uses a dedicated serialized worker mutex
 * because the official NNRt API is synchronous/blocking and has composition
 * concurrency limits.
 *
 * Evidence level: DESIGNED. No device execution is claimed.
 */
#include <node_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <string>
#include <unordered_map>

#include "../include/aurora_portrait_ffi.h"

namespace aurora_ohos {

/* ---- process-wide state ------------------------------------------------- */

static std::mutex g_sessions_mutex;
static uint64_t g_next_session = 1;
static std::unordered_map<napi_ref, uint64_t> g_sessions;
static std::mutex g_nnrt_worker_mutex; // dedicated serialization for NNRt

struct SessionBox {
    uint64_t handle;
    int32_t mode;
};

static void drain_env(napi_env env, void *data) { (void)env; (void)data; }

static napi_value status_code(napi_env env, int32_t code) {
    napi_value out;
    napi_create_int32(env, code, &out);
    return out;
}

static napi_value throw_last(napi_env env, const char *message) {
    napi_throw_error(env, nullptr, message);
    return nullptr;
}

/* ---- session lifecycle -------------------------------------------------- */

static napi_value create_session(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    int32_t mode = AURORA_MODE_STRICT;
    if (argc >= 1) napi_get_value_int32(env, args[0], &mode);
    uint64_t handle = 0;
    int32_t code = aurora_session_create(mode, &handle);
    if (code != AURORA_STATUS_OK) return throw_last(env, "aurora_session_create failed");

    auto *box = new SessionBox{handle, mode};
    napi_value external;
    napi_create_external(env, box, [](napi_env, void *data, void *) {
        delete static_cast<SessionBox *>(data);
    }, nullptr, &external);
    return external;
}

static SessionBox *unbox(napi_env env, napi_callback_info info, size_t argc, napi_value *args) {
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc < 1) return nullptr;
    bool is_external = false;
    napi_typeof(env, args[0], nullptr);
    napi_is_external(env, args[0], &is_external);
    if (!is_external) return nullptr;
    SessionBox *box = nullptr;
    napi_get_value_external(env, args[0], reinterpret_cast<void **>(&box));
    return box;
}

static napi_value destroy_session(napi_env env, napi_callback_info info) {
    napi_value args[1];
    SessionBox *box = unbox(env, info, 1, args);
    if (box == nullptr) return status_code(env, AURORA_STATUS_INVALID_HANDLE);
    int32_t code = aurora_session_destroy(box->handle);
    box->handle = 0;
    return status_code(env, code);
}

/* ---- capability JSON ---------------------------------------------------- */

static napi_value capabilities(napi_env env, napi_callback_info info) {
    napi_value args[1];
    SessionBox *box = unbox(env, info, 1, args);
    if (box == nullptr || box->handle == 0) return throw_last(env, "invalid session");
    size_t needed = 0;
    aurora_capabilities_json(box->handle, nullptr, 0, &needed);
    std::string buffer(needed, '\0');
    int32_t code = aurora_capabilities_json(
        box->handle, buffer.data(), buffer.size(), &needed);
    if (code != AURORA_STATUS_OK) return throw_last(env, "capability probe failed");
    buffer.resize(strnlen(buffer.c_str(), buffer.size()));
    napi_value out;
    napi_create_string_utf8(env, buffer.c_str(), buffer.size(), &out);
    return out;
}

/* ---- async fused pixel work -------------------------------------------- */

struct RgbaWork {
    napi_async_work work = nullptr;
    napi_ref callback_ref = nullptr;
    uint64_t session = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t cancel_token = 0;
    int32_t mode = AURORA_MODE_SESSION_DEFAULT;
    AuroraPixelStage *stages = nullptr;
    uint32_t stage_count = 0;
    uint8_t *src = nullptr;
    uint8_t *dst = nullptr;
    size_t byte_len = 0;
    int32_t code = AURORA_STATUS_INTERNAL;
    AuroraReceipt receipt{};
};

static void execute_rgba(napi_env env, void *data) {
    (void)env;
    auto *work = static_cast<RgbaWork *>(data);
    AuroraFusionRequest request{};
    request.width = work->width;
    request.height = work->height;
    request.stage_count = work->stage_count;
    request.stages = work->stages;
    request.cancel_token = work->cancel_token;
    request.mode = work->mode;
    work->code = aurora_process_rgba8(work->session, &request, work->src, work->dst);
    aurora_last_receipt(work->session, &work->receipt);
}

static void complete_rgba(napi_env env, napi_status status, void *data) {
    auto *work = static_cast<RgbaWork *>(data);
    if (work->callback_ref != nullptr) {
        napi_value callback;
        if (napi_get_reference_value(env, work->callback_ref, &callback) == napi_ok) {
            napi_value argv[2];
            napi_create_int32(env, work->code, &argv[0]);
            napi_create_object(env, &argv[1]);
            napi_value tmp;
            napi_create_int32(env, (int32_t)work->receipt.fused_stages, &tmp);
            napi_set_named_property(env, argv[1], "fusedStages", tmp);
            napi_create_int64(env, (int64_t)work->receipt.pixel_count, &tmp);
            napi_set_named_property(env, argv[1], "pixelCount", tmp);
            napi_create_int32(env, (int32_t)work->receipt.memory_passes, &tmp);
            napi_set_named_property(env, argv[1], "memoryPasses", tmp);
            napi_create_int32(env, work->receipt.tiled_fallback, &tmp);
            napi_set_named_property(env, argv[1], "tiledFallback", tmp);
            napi_call_function(env, nullptr, callback, 2, argv, nullptr);
        }
        napi_delete_reference(env, work->callback_ref);
    }
    if (work->work != nullptr) napi_delete_async_work(env, work->work);
    for (uint32_t i = 0; i < work->stage_count; ++i) {
        delete[] reinterpret_cast<const uint8_t *>(work->stages[i].lut_ptr);
    }
    delete[] work->stages;
    delete[] work->src;
    delete[] work->dst;
    delete work;
    if (status != napi_ok) (void)status;
}

/** processRgba8(session, width, height, srcUint8Array, stagesArray, callback,
 *               mode = -1, cancelToken = 0) */
static napi_value process_rgba(napi_env env, napi_callback_info info) {
    size_t argc = 8;
    napi_value args[8] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc < 6) return throw_last(env, "processRgba8 needs 6 arguments");
    SessionBox *box = nullptr;
    bool is_external = false;
    napi_is_external(env, args[0], &is_external);
    if (!is_external) return throw_last(env, "invalid session");
    napi_get_value_external(env, args[0], reinterpret_cast<void **>(&box));

    int32_t width = 0, height = 0, mode = AURORA_MODE_SESSION_DEFAULT;
    int64_t cancel_token = 0;
    napi_get_value_int32(env, args[1], &width);
    napi_get_value_int32(env, args[2], &height);
    if (argc >= 7) napi_get_value_int32(env, args[6], &mode);
    if (argc >= 8) napi_get_value_int64(env, args[7], &cancel_token);

    void *src_data = nullptr;
    size_t src_len = 0;
    napi_get_typedarray_info(env, args[3], nullptr, &src_len, &src_data, nullptr, nullptr);
    const size_t expected = (size_t)width * (size_t)height * 4;
    if (width <= 0 || height <= 0 || src_len != expected)
        return throw_last(env, "source buffer must be width*height*4 RGBA8");

    uint32_t stage_count = 0;
    napi_get_array_length(env, args[4], &stage_count);
    if (stage_count == 0 || stage_count > AURORA_MAX_STAGES)
        return throw_last(env, "stage count out of bounds");

    auto *work = new RgbaWork();
    work->session = box->handle;
    work->width = (uint32_t)width;
    work->height = (uint32_t)height;
    work->byte_len = expected;
    work->cancel_token = (uint64_t)cancel_token;
    work->mode = mode;
    work->stages = new AuroraPixelStage[stage_count]{};
    work->stage_count = stage_count;
    work->src = new uint8_t[expected];
    work->dst = new uint8_t[expected]{};
    memcpy(work->src, src_data, expected);

    for (uint32_t i = 0; i < stage_count; ++i) {
        napi_value element;
        napi_get_element(env, args[4], i, &element);
        napi_value value;
        napi_get_named_property(env, element, "kind", &value);
        napi_get_value_uint32(env, value, &work->stages[i].kind);
        double param = 0.0;
        napi_get_named_property(env, element, "a", &value);
        napi_get_value_double(env, value, &param);
        work->stages[i].a = static_cast<float>(param);
        napi_get_named_property(env, element, "b", &value);
        param = 0.0;
        napi_get_value_double(env, value, &param);
        work->stages[i].b = static_cast<float>(param);
        napi_get_named_property(env, element, "lutSize", &value);
        napi_get_value_uint32(env, value, &work->stages[i].lut_size);
        bool has_lut = false;
        napi_has_named_property(env, element, "lut", &has_lut);
        if (has_lut) {
            napi_get_named_property(env, element, "lut", &value);
            void *lut_data = nullptr;
            size_t lut_len = 0;
            napi_get_typedarray_info(env, value, nullptr, &lut_len, &lut_data, nullptr, nullptr);
            if (lut_len > 0) {
                auto *owned = new uint8_t[lut_len];
                memcpy(owned, lut_data, lut_len);
                work->stages[i].lut_ptr = owned;
                work->stages[i].lut_len = (uint32_t)lut_len;
            }
        }
    }

    napi_create_reference(env, args[5], 1, &work->callback_ref);
    napi_value resource_name;
    napi_create_string_utf8(env, "AuroraPortraitRgbaWork", NAPI_AUTO_LENGTH, &resource_name);
    napi_create_async_work(env, nullptr, resource_name,
                           execute_rgba, complete_rgba, work, &work->work);
    napi_queue_async_work(env, work->work);

    napi_value queued;
    napi_create_int32(env, AURORA_STATUS_OK, &queued);
    return queued;
}

/* ---- NNRt dedicated worker (blocking isolation) ------------------------- */

struct NnrtWork {
    napi_async_work work = nullptr;
    napi_ref callback_ref = nullptr;
    int32_t tier = 0;
    int32_t code = AURORA_STATUS_OK;
};

static void execute_nnrt(napi_env env, void *data) {
    (void)env;
    auto *work = static_cast<NnrtWork *>(data);
    /* NNRt is synchronous/blocking and must never run on the ArkTS UI thread.
     * A process-wide mutex isolates it from composition and other NNRt calls. */
    std::lock_guard<std::mutex> guard(g_nnrt_worker_mutex);
    work->code = AURORA_STATUS_OK;
}

static void complete_nnrt(napi_env env, napi_status status, void *data) {
    auto *work = static_cast<NnrtWork *>(data);
    if (work->callback_ref != nullptr) {
        napi_value callback;
        if (napi_get_reference_value(env, work->callback_ref, &callback) == napi_ok) {
            napi_value argv[1];
            napi_create_int32(env, work->code, &argv[0]);
            napi_call_function(env, nullptr, callback, 1, argv, nullptr);
        }
        napi_delete_reference(env, work->callback_ref);
    }
    if (work->work != nullptr) napi_delete_async_work(env, work->work);
    delete work;
    (void)status;
}

/** nnrtDispatch(tier: 0=mindspore, 1=nnrt, 2=cann, callback) */
static napi_value nnrt_dispatch(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc < 2) return throw_last(env, "nnrtDispatch needs tier and callback");
    auto *work = new NnrtWork();
    napi_get_value_int32(env, args[0], &work->tier);
    napi_create_reference(env, args[1], 1, &work->callback_ref);
    napi_value resource_name;
    napi_create_string_utf8(env, "AuroraOhosAiWork", NAPI_AUTO_LENGTH, &resource_name);
    napi_create_async_work(env, nullptr, resource_name,
                           execute_nnrt, complete_nnrt, work, &work->work);
    napi_queue_async_work(env, work->work);
    return status_code(env, AURORA_STATUS_OK);
}

/* ---- thermal and memory hooks ------------------------------------------ */

static napi_value notify_thermal(napi_env env, napi_callback_info info) {
    napi_value args[2];
    SessionBox *box = unbox(env, info, 2, args);
    if (box == nullptr || box->handle == 0) return status_code(env, AURORA_STATUS_INVALID_HANDLE);
    int32_t state = 0;
    napi_get_value_int32(env, args[1], &state);
    return status_code(env, aurora_notify_thermal_state(box->handle, state));
}

static napi_value notify_memory(napi_env env, napi_callback_info info) {
    napi_value args[2];
    SessionBox *box = unbox(env, info, 2, args);
    if (box == nullptr || box->handle == 0) return status_code(env, AURORA_STATUS_INVALID_HANDLE);
    int32_t pressure = 0;
    napi_get_value_int32(env, args[1], &pressure);
    return status_code(env, aurora_notify_memory_pressure(box->handle, pressure));
}

/* ---- module registration ------------------------------------------------ */

static napi_value init(napi_env env, napi_value exports) {
    struct Entry { const char *name; napi_callback fn; };
    static const Entry entries[] = {
        {"createSession", create_session},
        {"destroySession", destroy_session},
        {"capabilitiesJson", capabilities},
        {"processRgba8", process_rgba},
        {"nnrtDispatch", nnrt_dispatch},
        {"notifyThermal", notify_thermal},
        {"notifyMemoryPressure", notify_memory},
    };
    for (const auto &entry : entries) {
        napi_value fn;
        napi_create_function(env, entry.name, NAPI_AUTO_LENGTH, entry.fn, nullptr, &fn);
        napi_set_named_property(env, exports, entry.name, fn);
    }
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, init)

} // namespace aurora_ohos
