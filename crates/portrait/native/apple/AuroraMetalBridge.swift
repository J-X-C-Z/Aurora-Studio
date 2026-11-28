// Aurora Portrait iOS/iPadOS bridge.
//
// Imports the stable C ABI from `native/include/aurora_portrait_ffi.h` and
// provides PhotoKit-safe image import, ProcessInfo thermal/memory hooks,
// MTKView/CAMetalLayer presentation and Core ML compute-unit selection.
//
// Evidence level: DESIGNED. No GPU/Core ML/ANE execution claim is made here;
// the Rust CPU reference is the only verified pixel path. Do not treat a
// compiled bridge as vendor parity or real-photo acceptance.

import Foundation
import Metal
import MetalKit
import Photos
import CoreML
import UIKit

// MARK: - C ABI status

public enum AuroraBridgeError: Error, Equatable {
    case nullPointer
    case invalidHandle
    case invalidArgument
    case bufferTooSmall
    case strictRejected
    case cancelled
    case modeUnsupported
    case panic
    case internalError
    case budgetExhausted
    case unknown(Int32)

    init(code: Int32) {
        switch code {
        case 1: self = .nullPointer
        case 2: self = .invalidHandle
        case 3: self = .invalidArgument
        case 4: self = .bufferTooSmall
        case 5: self = .strictRejected
        case 6: self = .cancelled
        case 7: self = .modeUnsupported
        case 8: self = .panic
        case 9: self = .internalError
        case 10: self = .budgetExhausted
        default: self = .unknown(code)
        }
    }
}

@inline(__always)
private func auroraCheck(_ code: Int32) throws {
    if code != 0 { throw AuroraBridgeError(code: code) }
}

// MARK: - Stage descriptors

public enum AuroraStage {
    case brightness(Float)
    case whiten(Float)
    case rosy(Float)
    case temperature(warmth: Float, tint: Float)
    case saturation(Float)
    case lut1D([UInt8])                 // exactly 768 bytes: R,G,B
    case lut3D(size: Int, table: [Float]) // size^3 * 3 little-endian floats

    var kind: UInt32 {
        switch self {
        case .brightness: return 0
        case .whiten: return 1
        case .rosy: return 2
        case .temperature: return 3
        case .saturation: return 4
        case .lut1D: return 5
        case .lut3D: return 6
        }
    }
}

// MARK: - Options / receipts

public struct AuroraFusionOptions: Sendable {
    public var strict: Bool
    public var cancelToken: UInt64

    public init(strict: Bool = true, cancelToken: UInt64 = 0) {
        self.strict = strict
        self.cancelToken = cancelToken
    }
}

public struct AuroraProcessReceipt: Sendable {
    public let fusedStages: UInt32
    public let pixelCount: UInt64
    public let memoryPasses: UInt32
    public let tiledFallback: Bool
    public let backend: UInt32
    public let statusCode: Int32
}

// MARK: - Session

public final class AuroraPortraitSession {
    private var handle: UInt64 = 0
    private let queue = DispatchQueue(label: "aurora.portrait.ffi", qos: .userInitiated)
    private var thermalObserver: NSObjectProtocol?
    private var memoryObserver: NSObjectProtocol?

    public init(strict: Bool = true) throws {
        var h: UInt64 = 0
        try auroraCheck(aurora_session_create(strict ? 0 : 1, &h))
        self.handle = h
        installSystemHooks()
    }

    deinit {
        if handle != 0 { _ = aurora_session_destroy(handle) }
        if let thermalObserver { NotificationCenter.default.removeObserver(thermalObserver) }
        if let memoryObserver { NotificationCenter.default.removeObserver(memoryObserver) }
    }

    private func installSystemHooks() {
        // ProcessInfo.thermalState is polled per frame as well; the notification
        // is only a hint, not a guarantee of state change.
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil, queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            let code: Int32
            switch ProcessInfo.processInfo.thermalState {
            case .nominal: code = 0
            case .fair: code = 1
            case .serious: code = 2
            case .critical: code = 3
            @unknown default: code = 3
            }
            self.queue.async { _ = aurora_notify_thermal_state(self.handle, code) }
        }
        memoryObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil, queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            self.queue.async { _ = aurora_notify_memory_pressure(self.handle, 2) }
        }
    }

    public func capabilitiesJSON() throws -> String {
        var length = 0
        _ = aurora_capabilities_json(handle, nil, 0, &length)
        guard length > 0 else { throw AuroraBridgeError.bufferTooSmall }
        var buffer = [CChar](repeating: 0, count: length)
        let code = buffer.withUnsafeMutableBufferPointer { raw -> Int32 in
            aurora_capabilities_json(handle, raw.baseAddress, length, &length)
        }
        try auroraCheck(code)
        return String(cString: buffer)
    }

    public func processRGBA(
        width: Int,
        height: Int,
        src: UnsafePointer<UInt8>,
        dst: UnsafeMutablePointer<UInt8>,
        stages: [AuroraStage],
        options: AuroraFusionOptions = AuroraFusionOptions()
    ) throws -> AuroraProcessReceipt {
        guard width > 0, height > 0, !stages.isEmpty, stages.count <= 64 else {
            throw AuroraBridgeError.invalidArgument
        }
        // Flat little-endian LUT arena. Every LUT pointer borrows from this
        // buffer, which stays alive for the entire FFI call.
        var lutArena = [UInt8]()
        var lutOffsets = [Int]()
        var rawStages = [AuroraPixelStage]()
        for stage in stages {
            var raw = AuroraPixelStage()
            raw.kind = stage.kind
            switch stage {
            case .brightness(let v): raw.a = v
            case .whiten(let v): raw.a = v
            case .rosy(let v): raw.a = v
            case .temperature(let w, let t): raw.a = w; raw.b = t
            case .saturation(let v): raw.a = v
            case .lut1D(let bytes):
                guard bytes.count == 768 else { throw AuroraBridgeError.invalidArgument }
                lutOffsets.append(lutArena.count)
                lutArena.append(contentsOf: bytes)
                raw.lut_len = 768
            case .lut3D(let size, let table):
                guard size >= 2, table.count == size * size * size * 3 else {
                    throw AuroraBridgeError.invalidArgument
                }
                lutOffsets.append(lutArena.count)
                for value in table {
                    withUnsafeBytes(of: value.littleEndian) { lutArena.append(contentsOf: $0) }
                }
                raw.lut_size = UInt32(size)
                raw.lut_len = UInt32(table.count * 4)
            }
            rawStages.append(raw)
        }

        return try lutArena.withUnsafeBytes { arena in
            let base = arena.baseAddress?.assumingMemoryBound(to: UInt8.self)
            var cursor = 0
            for index in rawStages.indices {
                if rawStages[index].lut_len > 0 {
                    rawStages[index].lut_ptr = base?.advanced(by: lutOffsets[cursor])
                    cursor += 1
                }
            }
            return try rawStages.withUnsafeBufferPointer { stageBuffer in
                var request = AuroraFusionRequest()
                request.width = UInt32(width)
                request.height = UInt32(height)
                request.stage_count = UInt32(rawStages.count)
                request.stages = stageBuffer.baseAddress
                request.cancel_token = options.cancelToken
                request.mode = options.strict ? 0 : 1
                let code = aurora_process_rgba8(handle, &request, src, dst)
                try auroraCheck(code)
                var receipt = AuroraReceipt()
                try auroraCheck(aurora_last_receipt(handle, &receipt))
                return AuroraProcessReceipt(
                    fusedStages: receipt.fused_stages,
                    pixelCount: receipt.pixel_count,
                    memoryPasses: receipt.memory_passes,
                    tiledFallback: receipt.tiled_fallback != 0,
                    backend: receipt.backend,
                    statusCode: receipt.status_code
                )
            }
        }
    }
}

// MARK: - PhotoKit safe import boundary

public enum AuroraPhotoImport {
    /// Loads image data from PHAsset without mutating the asset, library, or
    /// sidecars. Only in-memory RGBA8 bytes are handed to the Rust ABI.
    public static func loadRGBA8(
        from asset: PHAsset,
        targetSize: CGSize,
        completion: @escaping (Result<(width: Int, height: Int, bytes: [UInt8]), Error>) -> Void
    ) {
        let options = PHImageRequestOptions()
        options.isSynchronous = false
        options.deliveryMode = .highQualityFormat
        options.resizeMode = .exact
        options.isNetworkAccessAllowed = false
        PHImageManager.default().requestImage(
            for: asset,
            targetSize: targetSize,
            contentMode: .aspectFit,
            options: options
        ) { image, info in
            if let info, let error = info[PHImageErrorKey] as? Error {
                completion(.failure(error))
                return
            }
            guard let image,
                  let cg = image.cgImage else {
                completion(.failure(AuroraBridgeError.invalidArgument))
                return
            }
            let width = cg.width, height = cg.height
            var bytes = [UInt8](repeating: 0, count: width * height * 4)
            guard let context = CGContext(
                data: &bytes,
                width: width, height: height,
                bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue |
                            CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                completion(.failure(AuroraBridgeError.internalError))
                return
            }
            context.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
            completion(.success((width, height, bytes)))
        }
    }
}

// MARK: - Metal presentation

/// MTKView-backed preview. Uses a regular CAMetalLayer/MTLDevice path; no
/// Core Animation or PhotoKit object is mutated by the Rust bridge.
public final class AuroraMetalPreviewView: MTKView {
    public override init(frame frameRect: CGRect, device: MTLDevice?) {
        super.init(frame: frameRect, device: device ?? MTLCreateSystemDefaultDevice())
        self.framebufferOnly = true
        self.isPaused = true
        self.enableSetNeedsDisplay = true
    }

    required init(coder: NSCoder) {
        super.init(coder: coder)
        self.device = MTLCreateSystemDefaultDevice()
        self.framebufferOnly = true
        self.isPaused = true
        self.enableSetNeedsDisplay = true
    }
}

/// Core ML compute-unit selection. ANE is selected only when the model
/// explicitly supports it; this does not claim neural-engine execution.
public enum AuroraCoreMLPolicy {
    public static func computeUnits(allowNeuralEngine: Bool) -> MLComputeUnits {
        allowNeuralEngine ? .all : .cpuAndGPU
    }
}
