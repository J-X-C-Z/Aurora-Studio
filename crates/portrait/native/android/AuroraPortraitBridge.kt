/*
 * Aurora Portrait Android bridge (Kotlin).
 *
 * Owns the native session/token handles, streams SAF/MediaStore URI bytes
 * without writing to the source, and forwards ADPF thermal/hint callbacks.
 * The Kotlin layer never holds raw native pointers in persistent state and
 * never mutates user photos, sidecars or export targets.
 *
 * Evidence level: DESIGNED. GPU/NPU execution is a caller-supplied fact;
 * this bridge only transports bytes and system signals.
 */
package com.aurora.portrait

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.PowerManager
import android.system.Os
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

object AuroraStatus {
    const val OK = 0
    const val NULL_POINTER = 1
    const val INVALID_HANDLE = 2
    const val INVALID_ARGUMENT = 3
    const val BUFFER_TOO_SMALL = 4
    const val STRICT_REJECTED = 5
    const val CANCELLED = 6
    const val MODE_UNSUPPORTED = 7
    const val PANIC = 8
    const val INTERNAL = 9
    const val BUDGET_EXHAUSTED = 10
}

class AuroraFfiException(val code: Int) : RuntimeException("aurora ffi status=$code")

enum class AuroraMode(val code: Int) {
    STRICT(0),
    AUTO_FALLBACK(1),
}

data class AuroraReceipt(
    val fusedStages: Int,
    val pixelCount: Long,
    val memoryPasses: Int,
    val tiledFallback: Boolean,
    val backend: Int,
    val statusCode: Int,
)

/** Opaque cancellation token; cancellation is idempotent. */
class AuroraCancelToken internal constructor(
    internal val handle: Long,
) : AutoCloseable {
    private val closed = AtomicBoolean(false)

    fun cancel() {
        if (!closed.get()) nativeCancel(handle)
    }

    fun isCancelled(): Boolean {
        val flag = IntArray(1)
        nativeIsCancelled(handle, flag)
        return flag[0] != 0
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) nativeDestroyToken(handle)
    }
}

/**
 * One Rust FFI session. All native calls are marshalled through the companion
 * object's JNI entry points; handles are Long values only.
 */
class AuroraPortraitSession(
    mode: AuroraMode = AuroraMode.STRICT,
) : AutoCloseable {
    private val handle: Long = nativeCreate(mode.code)
    private val closed = AtomicBoolean(false)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val adpfHandle: Long = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        nativeAdpfCreate()
    } else {
        0L
    }

    init {
        require(handle != 0L) { "aurora_session_create returned null handle" }
    }

    fun capabilitiesJson(): String {
        val length = IntArray(1)
        nativeCapabilitiesJson(handle, null, 0, length)
        val size = length[0]
        require(size > 0) { "capability JSON length missing" }
        val bytes = ByteArray(size)
        checkCode(nativeCapabilitiesJson(handle, bytes, bytes.size, length))
        return String(bytes, 0, maxOf(0, length[0] - 1), Charsets.UTF_8)
    }

    fun createCancelToken(): AuroraCancelToken {
        val token = LongArray(1)
        checkCode(nativeCreateToken(token))
        return AuroraCancelToken(token[0])
    }

    fun notifyThermalStatus(status: Int, headroom: Float, lowMemory: Boolean) {
        checkOpen()
        nativeNotifyThermal(handle, status, headroom, lowMemory)
        if (adpfHandle != 0L) nativeAdpfUpdate(adpfHandle, status, headroom)
    }

    fun notifyMemoryPressure(pressure: Int) {
        checkOpen()
        nativeNotifyPressure(handle, pressure)
    }

    /**
     * Process a caller-owned RGBA8 frame. `strict` selects the per-call mode;
     * null uses the session mode. Returns a receipt, never a native pointer.
     */
    fun processRgba8(
        width: Int,
        height: Int,
        src: ByteArray,
        stages: List<AuroraStage>,
        strict: Boolean? = null,
        token: AuroraCancelToken? = null,
    ): AuroraReceipt {
        checkOpen()
        require(width > 0 && height > 0 && stages.isNotEmpty() && stages.size <= 64)
        require(src.size == width * height * 4) { "source must be RGBA8" }
        val dst = ByteArray(src.size)
        val packed = serializeStages(stages)
        val mode = when (strict) {
            null -> -1
            true -> AuroraMode.STRICT.code
            false -> AuroraMode.AUTO_FALLBACK.code
        }
        val code = nativeProcessRgba8(
            handle, width, height, packed, src, dst, mode, token?.handle ?: 0L,
        )
        checkCode(code)
        return receipt()
    }

    fun receipt(): AuroraReceipt {
        val values = LongArray(7)
        checkCode(nativeReceipt(handle, values))
        return AuroraReceipt(
            fusedStages = values[0].toInt(),
            pixelCount = values[1],
            memoryPasses = values[2].toInt(),
            tiledFallback = values[3] != 0L,
            backend = values[4].toInt(),
            statusCode = values[5].toInt(),
        )
    }

    /**
     * SAF/MediaStore URI import: reads bytes only. The source document and its
     * metadata are never modified; EXIF/GPS remains in the original file.
     */
    suspend fun loadRgba8FromUri(
        context: Context,
        uri: Uri,
        maxDimension: Int = 4096,
    ): Pair<Int, Int, ByteArray> = withContext(scope.coroutineContext) {
        val resolver: ContentResolver = context.contentResolver
        val bytes = resolver.openInputStream(uri)?.use { input ->
            val out = ByteArrayOutputStream()
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                out.write(buffer, 0, read)
            }
            out.toByteArray()
        } ?: throw IllegalArgumentException("cannot open SAF URI")
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "not a decodable image" }
        val scale = maxOf(1, (maxOf(bounds.outWidth, bounds.outHeight) + maxDimension - 1) / maxDimension)
        val decoded = BitmapFactory.Options().apply { inSampleSize = scale }
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, decoded)
            ?: throw IllegalArgumentException("bitmap decode failed")
        val width = bitmap.width
        val height = bitmap.height
        val rgba = ByteArray(width * height * 4)
        bitmap.copyPixelsToBuffer(ByteBuffer.wrap(rgba).order(ByteOrder.nativeOrder()))
        bitmap.recycle()
        Triple(width, height, rgba)
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            if (adpfHandle != 0L) nativeAdpfDestroy(adpfHandle)
            nativeDestroy(handle)
            scope.cancel()
        }
    }

    private fun checkOpen() {
        check(!closed.get()) { "session closed" }
    }

    private companion object {
        init {
            System.loadLibrary("aurora_portrait_ffi")
        }

        @JvmStatic external fun nativeCreate(mode: Int): Long
        @JvmStatic external fun nativeDestroy(handle: Long)
        @JvmStatic external fun nativeCreateToken(out: LongArray): Int
        @JvmStatic external fun nativeCancel(token: Long)
        @JvmStatic external fun nativeIsCancelled(token: Long, out: IntArray)
        @JvmStatic external fun nativeDestroyToken(token: Long)
        @JvmStatic external fun nativeNotifyThermal(handle: Long, status: Int, headroom: Float, lowMemory: Boolean)
        @JvmStatic external fun nativeNotifyPressure(handle: Long, pressure: Int)
        @JvmStatic external fun nativeProcessRgba8(
            handle: Long, width: Int, height: Int, stages: ByteArray,
            src: ByteArray, dst: ByteArray, mode: Int, token: Long,
        ): Int
        @JvmStatic external fun nativeReceipt(handle: Long, out: LongArray): Int
        @JvmStatic external fun nativeCapabilitiesJson(handle: Long, out: ByteArray?, capacity: Int, outLen: IntArray): Int

        @JvmStatic external fun nativeAdpfCreate(): Long
        @JvmStatic external fun nativeAdpfUpdate(handle: Long, status: Int, headroom: Float)
        @JvmStatic external fun nativeAdpfDestroy(handle: Long)

        fun checkCode(code: Int) {
            if (code != AuroraStatus.OK) throw AuroraFfiException(code)
        }

        /** Little-endian stage stream; must match aurora_jni.c exactly. */
        fun serializeStages(stages: List<AuroraStage>): ByteArray {
            val out = java.io.ByteArrayOutputStream()
            val writer = java.io.DataOutputStream(out)
            writer.writeInt(stages.size)
            for (stage in stages) {
                writer.writeInt(stage.kind)
                writer.writeFloat(stage.a)
                writer.writeFloat(stage.b)
                writer.writeInt(stage.lut?.size ?: 0)
                writer.writeInt(stage.lutSize)
                stage.lut?.let { writer.write(it) }
            }
            writer.flush()
            return out.toByteArray()
        }
    }
}

/** Mirror of the C `AuroraPixelStage`. `lut` is copied into the native call. */
data class AuroraStage(
    val kind: Int,
    val a: Float = 0f,
    val b: Float = 0f,
    val lut: ByteArray? = null,
    val lutSize: Int = 0,
) {
    companion object {
        fun brightness(delta: Float) = AuroraStage(0, a = delta)
        fun whiten(strength: Float) = AuroraStage(1, a = strength)
        fun rosy(strength: Float) = AuroraStage(2, a = strength)
        fun temperature(warmth: Float, tint: Float) = AuroraStage(3, a = warmth, b = tint)
        fun saturation(amount: Float) = AuroraStage(4, a = amount)
        fun lut1D(table: ByteArray): AuroraStage {
            require(table.size == 768)
            return AuroraStage(5, lut = table.copyOf())
        }
        fun lut3D(size: Int, table: FloatArray): AuroraStage {
            require(size >= 2 && table.size == size * size * size * 3)
            val byteBuffer = ByteBuffer.allocate(table.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            table.forEach { byteBuffer.putFloat(it) }
            return AuroraStage(6, lut = byteBuffer.array(), lutSize = size)
        }
    }
}
