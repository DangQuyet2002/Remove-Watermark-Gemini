import { WatermarkEngine } from '../core/watermarkEngine.js';
import { canvasToBlob } from '../core/canvasBlob.js';
import { createAllenkFdncnnOnnxRuntime } from '../core/allenkFdncnnOnnxRuntime.js';
import { resolveAllenkFdncnnRuntimeProfile } from '../video/videoDenoiseRuntimePolicy.js';

let enginePromise = null;
const allenkFdncnnRuntimePromises = new Map();

function getEngine() {
    if (!enginePromise) {
        enginePromise = WatermarkEngine.create();
    }
    return enginePromise;
}

function asErrorPayload(error) {
    if (!error) return { message: 'Unknown error' };
    return {
        message: error.message || String(error),
        stack: error.stack || null
    };
}

function getWorkerRootBaseUrl() {
    try {
        if (typeof self?.location?.href === 'string' && self.location.href.length > 0) {
            return new URL('../', self.location.href).href;
        }
    } catch {
        // Fallback to relative resolution below.
    }
    return '../';
}

function resolveAssetUrl(relativePath) {
    const clean = String(relativePath || '').replace(/^\.\//, '');
    const base = getWorkerRootBaseUrl();
    try {
        return new URL(clean, base).href;
    } catch {
        return `../${clean}`;
    }
}

function isLikelyJavascriptMime(contentType) {
    const mime = String(contentType || '').split(';')[0].trim().toLowerCase();
    return mime === 'application/javascript'
        || mime === 'text/javascript'
        || mime === 'application/ecmascript'
        || mime === 'text/ecmascript'
        || mime.endsWith('+javascript');
}

async function preflightWebGpuRuntimeAssets(paths) {
    try {
        const response = await fetch(paths.mjs, { cache: 'no-store' });
        if (!response.ok) return { ok: false };
        const contentType = response.headers.get('content-type') || '';
        return { ok: isLikelyJavascriptMime(contentType) };
    } catch {
        return { ok: false };
    }
}

async function loadAllenkFdncnnRuntime(runtimeProfile = resolveAllenkFdncnnRuntimeProfile()) {
    const profile = runtimeProfile || resolveAllenkFdncnnRuntimeProfile();
    if (!allenkFdncnnRuntimePromises.has(profile.id)) {
        const runtimePromise = (async () => {
            const modelUrl = resolveAssetUrl(profile.modelUrl);
            const wasmPaths = {
                mjs: resolveAssetUrl('onnxruntime/ort-wasm-simd-threaded.js'),
                wasm: resolveAssetUrl('onnxruntime/ort-wasm-simd-threaded.wasm')
            };
            const webgpuWasmPaths = {
                mjs: resolveAssetUrl('onnxruntime/ort-wasm-simd-threaded.asyncify.mjs'),
                wasm: resolveAssetUrl('onnxruntime/ort-wasm-simd-threaded.asyncify.wasm')
            };

            const response = await fetch(modelUrl);
            if (!response.ok) {
                throw new Error(`Failed to fetch AI model: ${response.status}`);
            }
            const modelBytes = new Uint8Array(await response.arrayBuffer());

            if (self.navigator?.gpu && self.__gwrDisableWebGpuDenoise !== true) {
                try {
                    const preflight = await preflightWebGpuRuntimeAssets(webgpuWasmPaths);
                    if (preflight.ok) {
                        const webgpuOrt = await import('onnxruntime-web/webgpu');
                        return await createAllenkFdncnnOnnxRuntime({
                            ort: webgpuOrt,
                            modelBytes,
                            executionProvider: 'webgpu',
                            wasmPaths: webgpuWasmPaths,
                            inputName: 'fdncnn_input',
                            outputName: 'fdncnn_output',
                            inputShape: profile.inputShape,
                            outputShape: profile.outputShape,
                            numThreads: 1
                        });
                    }
                } catch (error) {
                    console.warn('Worker WebGPU AI runtime unavailable, falling back to WASM:', error);
                }
            }

            return createAllenkFdncnnOnnxRuntime({
                modelBytes,
                executionProvider: 'wasm',
                wasmPaths,
                inputName: 'fdncnn_input',
                outputName: 'fdncnn_output',
                inputShape: profile.inputShape,
                outputShape: profile.outputShape,
                numThreads: 1
            });
        })();

        allenkFdncnnRuntimePromises.set(profile.id, runtimePromise);
        runtimePromise.catch(() => {
            if (allenkFdncnnRuntimePromises.get(profile.id) === runtimePromise) {
                allenkFdncnnRuntimePromises.delete(profile.id);
            }
        });
    }

    return allenkFdncnnRuntimePromises.get(profile.id);
}

self.addEventListener('message', async (event) => {
    const payload = event.data;
    if (!payload || typeof payload.type !== 'string') return;

    if (payload.type === 'ping') {
        self.postMessage({
            id: payload.id,
            ok: true,
            result: {
                ready: true
            }
        });
        return;
    }

    if (payload.type !== 'process-image') return;

    const { id, inputBuffer, mimeType, options } = payload;
    try {
        const engine = await getEngine();
        const inputBlob = new Blob([inputBuffer], { type: mimeType || 'image/png' });
        const imageBitmap = await createImageBitmap(inputBlob);
        const resolvedOptions = {
            ...(options || {}),
            ...(options?.thoroughMode && options?.disableWebGpu !== true
                ? { resolveAiDenoiseRuntime: loadAllenkFdncnnRuntime }
                : options?.thoroughMode
                    ? {
                        resolveAiDenoiseRuntime: (profile) => {
                            self.__gwrDisableWebGpuDenoise = true;
                            return loadAllenkFdncnnRuntime(profile);
                        }
                    }
                    : {})
        };
        const canvas = await engine.removeWatermarkFromImage(imageBitmap, resolvedOptions);
        if (typeof imageBitmap.close === 'function') {
            imageBitmap.close();
        }

        const pngBlob = await canvasToBlob(canvas, 'image/png', {
            nullBlobMessage: 'Failed to encode PNG blob'
        });
        const processedBuffer = await pngBlob.arrayBuffer();

        self.postMessage({
            id,
            ok: true,
            result: {
                processedBuffer,
                mimeType: 'image/png',
                meta: canvas.__watermarkMeta || null
            }
        }, [processedBuffer]);
    } catch (error) {
        self.postMessage({
            id,
            ok: false,
            error: asErrorPayload(error)
        });
    }
});
