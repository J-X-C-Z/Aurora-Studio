/*
 * Aurora Portrait HarmonyOS NEXT ArkTS bridge.
 *
 * Strictly independent of Android: no JNI, no AHardwareBuffer, no QAIRT/QNN,
 * no ADPF. UI code only enqueues work through Node-API; heavy computation runs
 * on libuv workers inside aurora_napi.cpp. XComponent provides the OHOS native
 * surface; this file never writes to user photos, sidecars or export targets.
 *
 * Evidence level: DESIGNED. Tier availability is supplied by the native host;
 * no NNRt/CANN/MindSpore execution is claimed by this wrapper.
 */

import { BusinessError } from '@kit.BasicServicesKit';
import { image } from '@kit.ImageKit';
import { emitter } from '@kit.BasicServicesKit';
import { taskpool } from '@kit.ArkTS';
import auroraNapi from 'libaurora_portrait_ohos.so';

export const AURORA_STATUS_OK: number = 0;
export const AURORA_MODE_STRICT: number = 0;
export const AURORA_MODE_AUTO_FALLBACK: number = 1;

export interface AuroraPixelStage {
  kind: number;
  a: number;
  b: number;
  lutSize: number;
  lut?: Uint8Array;
}

export interface AuroraReceipt {
  fusedStages: number;
  pixelCount: bigint;
  memoryPasses: number;
  tiledFallback: number;
}

export enum OhosAiTier {
  MINDSPORE_LITE_KIT = 0,
  NNRT = 1,
  CANN_KIT = 2,
}

/** Native session owner. `nativeSession` is an opaque Node-API external. */
export class AuroraPortraitBridge {
  private nativeSession: object | undefined;
  private cancelled: boolean = false;

  constructor(strict: boolean = true) {
    this.nativeSession = auroraNapi.createSession(
      strict ? AURORA_MODE_STRICT : AURORA_MODE_AUTO_FALLBACK
    );
  }

  close(): void {
    if (this.nativeSession !== undefined) {
      auroraNapi.destroySession(this.nativeSession);
      this.nativeSession = undefined;
    }
  }

  capabilitiesJson(): string {
    this.assertOpen();
    return auroraNapi.capabilitiesJson(this.nativeSession!) as string;
  }

  /**
   * Queue a fused RGBA8 pass on a native worker. The ArkTS UI thread returns
   * immediately; `onDone` runs on the event loop after native completion.
   */
  processRgba8(
    width: number,
    height: number,
    src: Uint8Array,
    stages: AuroraPixelStage[],
    onDone: (code: number, receipt: AuroraReceipt) => void,
    mode: number = -1,
    cancelToken: bigint = 0n,
  ): void {
    this.assertOpen();
    if (width <= 0 || height <= 0 || stages.length === 0 || stages.length > 64) {
      throw new Error('invalid Aurora fusion request');
    }
    if (src.byteLength !== width * height * 4) {
      throw new Error('source buffer must be width*height*4 RGBA8');
    }
    auroraNapi.processRgba8(
      this.nativeSession!, width, height, src, stages, onDone, mode, cancelToken
    );
  }

  /** NNRt calls are serialized on a dedicated native worker because the
   * official API is synchronous/blocking and has composition concurrency
   * limits. Never call inference directly on the UI thread. */
  dispatchAi(tier: OhosAiTier, onDone: (code: number) => void): void {
    auroraNapi.nnrtDispatch(tier as number, onDone);
  }

  notifyThermal(state: number): void {
    this.assertOpen();
    auroraNapi.notifyThermal(this.nativeSession!, state);
  }

  notifyMemoryPressure(pressure: number): void {
    this.assertOpen();
    auroraNapi.notifyMemoryPressure(this.nativeSession!, pressure);
  }

  /** Decode a caller-provided image bytes to RGBA8 in memory. No source file
   * is modified, and no EXIF/GPS is copied into the processing buffer. */
  static decodeRgba8(bytes: Uint8Array, maxDimension: number): Promise<Uint8Array> {
    return taskpool.execute(async (): Promise<Uint8Array> => {
      const source: image.ImageSource = image.createImageSource(bytes.buffer.slice(
        bytes.byteOffset, bytes.byteOffset + bytes.byteLength
      ) as ArrayBuffer);
      const opts: image.DecodingOptions = {
        desiredSize: { width: maxDimension, height: maxDimension },
        desiredPixelFormat: image.PixelMapFormat.RGBA_8888,
      };
      const pixelMap: image.PixelMap = await source.createPixelMap(opts);
      const info = await pixelMap.getImageInfo();
      const out = new Uint8Array(info.size.width * info.size.height * 4);
      await pixelMap.readPixelsToBuffer(out.buffer);
      pixelMap.release();
      source.release();
      return out;
    }) as Promise<Uint8Array>;
  }

  private assertOpen(): void {
    if (this.nativeSession === undefined) {
      throw new Error('Aurora session is closed');
    }
  }
}

/**
 * ArkUI component that embeds an OHOS XComponent surface. The native bridge
 * receives only buffer bytes; presentation/event handling stays on ArkUI.
 */
@Entry
@Component
export struct AuroraXComponentPreview {
  @State private status: string = 'DESIGNED';
  private bridge: AuroraPortraitBridge = new AuroraPortraitBridge(true);

  aboutToAppear(): void {
    const handler = emitter.createRunner('aurora.preview', 1);
    emitter.emit({ eventId: 1 }, {
      data: { message: 'aurora native surface ready' }
    });
    handler.cancel();
  }

  aboutToDisappear(): void {
    this.bridge.close();
  }

  build() {
    Column() {
      XComponent({
        id: 'aurora_surface',
        type: XComponentType.SURFACE,
        libraryname: 'aurora_portrait_ohos'
      })
        .width('100%')
        .height('70%')
        .onLoad(() => {
          this.status = 'DESIGNED';
        })
      Text(`Aurora Portrait: ${this.status}`)
        .fontSize(14)
        .margin(8)
    }
    .width('100%')
    .height('100%')
  }
}
