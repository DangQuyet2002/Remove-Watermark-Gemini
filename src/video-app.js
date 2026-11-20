import {
    DEFAULT_ADAPTIVE_ALPHA,
    DEFAULT_ALPHA_GAIN,
    DEFAULT_DENOISE_BACKEND,
    DEFAULT_EDGE_DENOISE_STRENGTH,
    DEFAULT_HIGH_QUALITY_CLEANUP,
    DEFAULT_RESIDUAL_CLEANUP_STRENGTH,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_VIDEO_BITRATE,
    VIDEO_DENOISE_BACKENDS,
    detectGeminiVideoWatermark,
    inspectGeminiVideoFile,
    removeGeminiVideoWatermark
} from './video/videoExport.js';
import { isReferenceGeminiVideoSize } from './video/videoWatermarkCatalog.js';
import {
    getAutomaticVideoPresetConfig,
    getRelocatedReviewPresetConfig
} from './video/videoPresetPolicy.js';
import { resolveAllenkFdncnnRuntimeProfile } from './video/videoDenoiseRuntimePolicy.js';
import {
    applyVideoAdaptiveAlphaDebugOverride,
    applyVideoBitrateDebugOverride
} from './video/videoDebugControlOverrides.js';
import {
    consumeDebugFileHandoff,
    getDebugFileKind,
    pickDebugUploadFile,
    pickDebugUploadFiles,
    saveDebugFileHandoff
} from './shared/debugFileHandoff.js';
import { createAllenkFdncnnOnnxRuntime } from './core/allenkFdncnnOnnxRuntime.js';

const $ = (id) => document.getElementById(id);
const ALLENK_FDNCNN_WASM_PATHS = Object.freeze({
    mjs: './onnxruntime/ort-wasm-simd-threaded.js',
    wasm: './onnxruntime/ort-wasm-simd-threaded.wasm'
});
const ALLENK_FDNCNN_WEBGPU_WASM_PATHS = Object.freeze({
    mjs: './onnxruntime/ort-wasm-simd-threaded.asyncify.mjs',
    wasm: './onnxruntime/ort-wasm-simd-threaded.asyncify.wasm'
});

const state = {
    file: null,
    originalUrl: null,
    processedUrl: null,
    metadata: null,
    detection: null,
    running: false,
    jobId: 0,
    syncingPlayback: false,

    // Queue & Batch Processing State
    queue: [],
    activeId: null,
    batchRunning: false,
    cancelRequested: false
};

const allenkFdncnnRuntimePromises = new Map();

const els = {
    dropzone: $('dropzone'),
    fileInput: $('fileInput'),
    comparePlayer: $('comparePlayer'),
    afterBadge: $('afterBadge'),
    playPauseBtn: $('playPauseBtn'),
    scrubber: $('scrubber'),
    timeLabel: $('timeLabel'),
    originalVideo: $('originalVideo'),
    processedVideo: $('processedVideo'),
    originalEmpty: $('originalEmpty'),
    processedEmpty: $('processedEmpty'),
    metadata: $('metadata'),
    detection: $('detection'),
    progressBar: $('progressBar'),
    progressText: $('progressText'),
    status: $('status'),
    alphaGain: $('alphaGain'),
    alphaGainValue: $('alphaGainValue'),
    adaptiveAlpha: $('adaptiveAlpha'),
    highQualityCleanup: $('highQualityCleanup'),
    denoiseBackend: $('denoiseBackend'),
    edgeDenoiseStrength: $('edgeDenoiseStrength'),
    edgeDenoiseStrengthValue: $('edgeDenoiseStrengthValue'),
    residualCleanup: $('residualCleanup'),
    residualCleanupValue: $('residualCleanupValue'),
    videoBitrateMbps: $('videoBitrateMbps'),
    sampleCount: $('sampleCount'),
    allowLowConfidence: $('allowLowConfidence'),
    autoPresetSummary: $('autoPresetSummary'),
    processBtn: $('processBtn'),
    detectBtn: $('detectBtn'),
    downloadBtn: $('downloadBtn'),
    resetBtn: $('resetBtn'),
    relocatedReviewPresetBtn: $('relocatedReviewPresetBtn'),

    // Batch Queue elements
    queuePanel: $('queuePanel'),
    queueBadge: $('queueBadge'),
    processAllBtn: $('processAllBtn'),
    cancelBatchBtn: $('cancelBatchBtn'),
    downloadAllBtn: $('downloadAllBtn'),
    clearQueueBtn: $('clearQueueBtn'),
    batchProgressWrapper: $('batchProgressWrapper'),
    batchProgressBar: $('batchProgressBar'),
    batchProgressLabel: $('batchProgressLabel'),
    batchProgressPct: $('batchProgressPct'),
    videoQueueList: $('videoQueueList'),
    emptyQueueMsg: $('emptyQueueMsg')
};

function setStatus(message, tone = 'info') {
    els.status.textContent = message || '';
    els.status.dataset.tone = tone;
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
        if (!response.ok) {
            return {
                ok: false,
                reason: `WebGPU runtime module unavailable: ${response.status}`
            };
        }
        const contentType = response.headers.get('content-type') || '';
        if (!isLikelyJavascriptMime(contentType)) {
            return {
                ok: false,
                reason: `WebGPU runtime module is served as ${contentType || 'unknown MIME'}`
            };
        }
        return { ok: true };
    } catch (error) {
        return {
            ok: false,
            reason: error?.message || String(error)
        };
    }
}

async function loadAllenkFdncnnRuntime(runtimeProfile = resolveAllenkFdncnnRuntimeProfile()) {
    const profile = runtimeProfile || resolveAllenkFdncnnRuntimeProfile();
    if (!allenkFdncnnRuntimePromises.has(profile.id)) {
        const runtimePromise = (async () => {
            const response = await fetch(profile.modelUrl);
            if (!response.ok) {
                throw new Error(`Không thể tải mô hình AI: ${response.status}`);
            }
            const modelBytes = new Uint8Array(await response.arrayBuffer());
            if (navigator.gpu && window.__gwrDisableWebGpuDenoise !== true) {
                try {
                    const preflight = await preflightWebGpuRuntimeAssets(ALLENK_FDNCNN_WEBGPU_WASM_PATHS);
                    if (!preflight.ok) {
                        console.warn('WebGPU AI runtime skipped:', preflight.reason);
                        throw new Error(preflight.reason);
                    }
                    setStatus('Đang bật xóa watermark bằng AI (WebGPU)...');
                    const webgpuOrt = await import('onnxruntime-web/webgpu');
                    return await createAllenkFdncnnOnnxRuntime({
                        ort: webgpuOrt,
                        modelBytes,
                        executionProvider: 'webgpu',
                        wasmPaths: ALLENK_FDNCNN_WEBGPU_WASM_PATHS,
                        inputName: 'fdncnn_input',
                        outputName: 'fdncnn_output',
                        inputShape: profile.inputShape,
                        outputShape: profile.outputShape
                    });
                } catch (error) {
                    console.warn('WebGPU AI runtime unavailable, falling back to WASM:', error);
                }
            }
            return createAllenkFdncnnOnnxRuntime({
                modelBytes,
                executionProvider: 'wasm',
                wasmPaths: ALLENK_FDNCNN_WASM_PATHS,
                inputName: 'fdncnn_input',
                outputName: 'fdncnn_output',
                inputShape: profile.inputShape,
                outputShape: profile.outputShape
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

async function resolveExportDenoiseRuntime(denoiseBackend, runtimeProfile = resolveAllenkFdncnnRuntimeProfile()) {
    if (denoiseBackend !== VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE) {
        return null;
    }
    setStatus('Đang tải mô hình AI FDnCNN ONNX, lần đầu sẽ hơi chậm...');
    return loadAllenkFdncnnRuntime(runtimeProfile);
}

function getAllenkFdncnnTemporalReuseConfig(runtime) {
    if (!runtime || runtime.executionProvider !== 'wasm') {
        return null;
    }
    const hasThreadedWasm = Boolean(crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined');
    return {
        maxFrames: hasThreadedWasm ? 1 : 2,
        threshold: hasThreadedWasm ? 4.5 : 6.5
    };
}

function resolveDetectionAllenkFdncnnSigma(detection) {
    if (Number.isFinite(window.__gwrVideoOverrideAllenkFdncnnSigma)) {
        return Math.max(0, Math.min(150, window.__gwrVideoOverrideAllenkFdncnnSigma));
    }
    if (detection?.watermarkKind === 'veo-text') {
        return detection.template?.cleanup?.runtimeFdncnnSigma ?? 75;
    }
    return 75;
}

function resolveDetectionAllenkFdncnnPadding(detection, runtimeProfile) {
    if (Number.isFinite(window.__gwrVideoOverrideAllenkFdncnnPadding)) {
        return Math.max(0, Math.round(window.__gwrVideoOverrideAllenkFdncnnPadding));
    }
    if (detection?.watermarkKind === 'veo-text' && Number.isFinite(detection.template?.cleanup?.allenkFdncnnPadding)) {
        return detection.template.cleanup.allenkFdncnnPadding;
    }
    return runtimeProfile.padding;
}

function setProgress(progress, label) {
    const pct = Number.isFinite(progress) ? Math.max(0, Math.min(100, Math.round(progress * 100))) : 0;
    els.progressBar.style.width = `${pct}%`;
    els.progressText.textContent = label || `${pct}%`;
}

function yieldToBrowserFrame() {
    return new Promise((resolve) => {
        const finish = () => setTimeout(resolve, 0);
        if (typeof requestAnimationFrame === 'function') {
            requestAnimationFrame(finish);
        } else {
            finish();
        }
    });
}

function createDetectionProgressHandler(jobId, { start = 0, span = 1 } = {}) {
    return ({ progress = 0, step = 'detect', sampledFrames = 0, sampleCount = 0 } = {}) => {
        if (jobId !== state.jobId) return;
        const safeProgress = Number.isFinite(progress) ? Math.max(0, Math.min(1, progress)) : 0;
        const labelByStep = {
            metadata: 'Đọc video',
            sample: sampleCount > 0 ? `Lấy mẫu khung ${sampledFrames}/${sampleCount}` : 'Lấy mẫu khung',
            score: 'Khớp watermark',
            done: 'Phát hiện xong'
        };
        setProgress(start + safeProgress * span, labelByStep[step] || 'Đang phát hiện');
        if (step === 'sample') {
            setStatus(sampleCount > 0
                ? `Đang lấy mẫu khung để phát hiện watermark: ${sampledFrames}/${sampleCount}`
                : 'Đang lấy mẫu khung để phát hiện watermark...');
        } else if (step === 'score') {
            setStatus('Đang khớp các ứng viên watermark, trang vẫn phản hồi bình thường...');
        }
    };
}

function formatSeconds(value) {
    if (!Number.isFinite(value)) return 'Không rõ';
    return `${value.toFixed(2)}s`;
}

function formatBitrate(value) {
    if (!Number.isFinite(value)) return 'Không rõ';
    return `${(value / 1000 / 1000).toFixed(2)} Mbps`;
}

function formatPlaybackTime(value) {
    if (!Number.isFinite(value) || value < 0) return '0:00';
    const totalSeconds = Math.floor(value);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

function updateButtons() {
    const hasFile = Boolean(state.file);
    const isBusy = state.running || state.batchRunning;
    els.detectBtn.disabled = !hasFile || isBusy;
    els.processBtn.disabled = !hasFile || isBusy;
    const canDownload = Boolean(state.processedUrl) && !isBusy;
    els.downloadBtn.setAttribute('aria-disabled', canDownload ? 'false' : 'true');
    els.downloadBtn.tabIndex = canDownload ? 0 : -1;
    els.resetBtn.disabled = isBusy;
    updatePlaybackControls();
    updateBatchControls();
}

function hasPlayableOriginal() {
    return Boolean(state.originalUrl) && Number.isFinite(els.originalVideo.duration);
}

function hasPlayableProcessed() {
    return Boolean(state.processedUrl);
}

function updatePlaybackControls() {
    const canPlay = Boolean(state.originalUrl);
    els.playPauseBtn.disabled = !canPlay;
    els.scrubber.disabled = !canPlay;
    els.playPauseBtn.dataset.playing = els.originalVideo.paused ? 'false' : 'true';
    els.playPauseBtn.setAttribute('aria-label', els.originalVideo.paused ? 'Phát' : 'Tạm dừng');

    const duration = Number.isFinite(els.originalVideo.duration) ? els.originalVideo.duration : 0;
    const currentTime = Number.isFinite(els.originalVideo.currentTime) ? els.originalVideo.currentTime : 0;
    if (duration > 0 && !els.scrubber.matches(':active')) {
        els.scrubber.value = String(Math.round((currentTime / duration) * 1000));
    }
    els.timeLabel.textContent = duration > 0
        ? `${formatPlaybackTime(currentTime)} / ${formatPlaybackTime(duration)}`
        : formatPlaybackTime(currentTime);
}

function updateCompareMode() {
    const hasAfter = hasPlayableProcessed();
    els.afterBadge.hidden = !hasAfter;
    els.processedEmpty.hidden = hasAfter;
}

function syncProcessedToOriginal({ force = false } = {}) {
    if (!hasPlayableProcessed() || state.syncingPlayback) return;
    const targetTime = Number(els.originalVideo.currentTime) || 0;
    if (!force && Math.abs((Number(els.processedVideo.currentTime) || 0) - targetTime) < 0.08) return;

    state.syncingPlayback = true;
    try {
        els.processedVideo.currentTime = targetTime;
    } catch (error) {
        console.warn('sync processed video failed:', error);
    } finally {
        state.syncingPlayback = false;
    }
}

async function playComparison() {
    if (!state.originalUrl) return;
    syncProcessedToOriginal({ force: true });
    try {
        await els.originalVideo.play();
        if (hasPlayableProcessed()) {
            await els.processedVideo.play().catch((error) => {
                console.warn('processed video play failed:', error);
            });
        }
    } catch (error) {
        console.warn('original video play failed:', error);
        setStatus('Trình duyệt đã chặn phát video, hãy nhấn nút phát thêm một lần.', 'warn');
    } finally {
        updatePlaybackControls();
    }
}

function pauseComparison() {
    els.originalVideo.pause();
    els.processedVideo.pause();
    updatePlaybackControls();
}

function togglePlayback() {
    if (els.originalVideo.paused) {
        playComparison();
    } else {
        pauseComparison();
    }
}

function seekComparison(value) {
    const duration = Number(els.originalVideo.duration);
    if (!Number.isFinite(duration) || duration <= 0) return;
    const currentTime = (Number(value) / 1000) * duration;
    els.originalVideo.currentTime = currentTime;
    syncProcessedToOriginal({ force: true });
    updatePlaybackControls();
}

function renderAutoPresetSummary(preset = null) {
    if (!els.autoPresetSummary) return;
    if (!preset) {
        els.autoPresetSummary.innerHTML = `
            <strong>Xử lý tự động bằng AI</strong>
            <span>Tự động phát hiện watermark sau khi chọn video, dùng mô hình AI cục bộ để làm sạch khi xuất.</span>
        `;
        return;
    }

    els.autoPresetSummary.innerHTML = `
        <strong>${preset.label}</strong>
        <span>${preset.description}</span>
    `;
}

function renderMetadata(metadata) {
    if (!metadata) {
        els.metadata.innerHTML = '<p class="muted">Chờ tải video</p>';
        return;
    }
    const reference = isReferenceGeminiVideoSize(metadata.width, metadata.height);
    els.metadata.innerHTML = `
        <dl>
            <div><dt>Kích thước</dt><dd>${metadata.width} x ${metadata.height}</dd></div>
            <div><dt>Thời lượng</dt><dd>${formatSeconds(metadata.duration)}</dd></div>
            <div><dt>Tốc độ khung hình</dt><dd>${metadata.frameRate.toFixed(2)} fps</dd></div>
            <div><dt>Bitrate video</dt><dd>${formatBitrate(metadata.averageBitrate)}</dd></div>
            <div><dt>Quy cách watermark</dt><dd>${reference ? '1920x1080 đã xác nhận' : 'Suy ra theo tỷ lệ, thử nghiệm'}</dd></div>
        </dl>
    `;
}

function renderDetection(detection) {
    if (!detection) {
        els.detection.innerHTML = '<p class="muted">Hãy phát hiện trước hoặc xuất trực tiếp</p>';
        return;
    }

    const best = detection.summary?.best || {};
    const bestLabel = detection.watermarkKind === 'veo-text'
        ? (best.templateId || detection.template?.id || 'Veo text')
        : (best.label || best.candidateId || detection.candidate?.label || 'unknown');
    const bestScore = Number.isFinite(best.meanConfidence)
        ? best.meanConfidence
        : Number.isFinite(best.meanNcc)
            ? best.meanNcc
            : null;
    els.detection.innerHTML = `
        <dl>
            <div><dt>Ứng viên</dt><dd>${bestLabel}</dd></div>
            <div><dt>Vị trí</dt><dd>${detection.position.x}, ${detection.position.y}</dd></div>
            <div><dt>Kích thước</dt><dd>${detection.position.width} x ${detection.position.height}</dd></div>
            <div><dt>Điểm trung bình</dt><dd>${Number.isFinite(bestScore) ? bestScore.toFixed(3) : '-'}</dd></div>
            <div><dt>Phiếu bầu</dt><dd>${best.votes || 0}/${detection.summary?.frameCount || 0}</dd></div>
            <div><dt>Trạng thái</dt><dd>${detection.isConfident ? 'Có thể xuất' : 'Độ tin cậy thấp'}</dd></div>
        </dl>
    `;
}

function formatFileSize(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function createQueueItem(file) {
    return {
        id: `vid_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
        file,
        name: file.name || 'video.mp4',
        size: file.size || 0,
        formattedSize: formatFileSize(file.size),
        originalUrl: URL.createObjectURL(file),
        processedUrl: null,
        processedBlob: null,
        metadata: null,
        detection: null,
        status: 'idle',
        progress: 0,
        progressText: 'Chờ xử lý',
        errorMessage: null
    };
}

function getActiveItem() {
    return state.queue.find((item) => item.id === state.activeId) || null;
}

function updateBatchControls() {
    const count = state.queue.length;
    const hasPending = state.queue.some((item) => item.status !== 'done');
    const doneCount = state.queue.filter((item) => item.status === 'done' && item.processedUrl).length;

    els.processAllBtn.disabled = !hasPending || state.running || state.batchRunning;
    els.processAllBtn.style.display = state.batchRunning ? 'none' : 'inline-flex';
    els.cancelBatchBtn.style.display = state.batchRunning ? 'inline-flex' : 'none';

    els.downloadAllBtn.disabled = doneCount === 0 || state.batchRunning;
    els.clearQueueBtn.disabled = count === 0 || state.batchRunning;
}

function setBatchProgress(progress, label) {
    const pct = Number.isFinite(progress) ? Math.max(0, Math.min(100, Math.round(progress * 100))) : 0;
    els.batchProgressBar.style.width = `${pct}%`;
    els.batchProgressPct.textContent = `${pct}%`;
    if (label) {
        els.batchProgressLabel.textContent = label;
    }
}

function renderQueueList() {
    const queue = state.queue;
    els.queueBadge.textContent = `${queue.length} video`;
    const doneCount = queue.filter((item) => item.status === 'done').length;
    if (doneCount > 0) {
        els.queueBadge.textContent = `${doneCount}/${queue.length} hoàn thành`;
    }

    if (queue.length === 0) {
        els.videoQueueList.innerHTML = `
            <div class="empty-queue" id="emptyQueueMsg">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="17" x2="22" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/></svg>
                <p><strong>Chưa có video nào trong hàng đợi</strong></p>
                <span>Chọn hoặc kéo thả nhiều video (MP4, WebM, MOV) để bắt đầu xử lý hàng loạt.</span>
            </div>
        `;
        updateBatchControls();
        return;
    }

    els.videoQueueList.innerHTML = queue.map((item, index) => {
        const isActive = item.id === state.activeId;
        const statusMap = {
            idle: { text: 'Chờ xử lý', class: 'status-idle' },
            detecting: { text: 'Phát hiện', class: 'status-detecting' },
            processing: { text: item.progressText || 'Đang xử lý', class: 'status-processing' },
            done: { text: 'Hoàn thành', class: 'status-done' },
            error: { text: item.errorMessage || 'Lỗi', class: 'status-error' }
        };
        const statusInfo = statusMap[item.status] || statusMap.idle;
        const progressPct = Math.round((item.progress || 0) * 100);

        const metaParts = [];
        metaParts.push(item.formattedSize);
        if (item.metadata) {
            metaParts.push(`${item.metadata.width}x${item.metadata.height}`);
            metaParts.push(formatSeconds(item.metadata.duration));
        }

        return `
            <div class="queue-item ${isActive ? 'active' : ''}" data-id="${item.id}">
                <div class="queue-item-thumb">
                    <span>${index + 1}</span>
                </div>
                <div class="queue-item-info">
                    <div class="queue-item-title" title="${item.name}">${item.name}</div>
                    <div class="queue-item-meta">
                        <span class="queue-item-status-badge ${statusInfo.class}">${statusInfo.text}</span>
                        ${metaParts.map((p) => `<span>• ${p}</span>`).join('')}
                    </div>
                    ${item.status === 'processing' ? `
                        <div class="queue-item-progress-track">
                            <div class="queue-item-progress-fill" style="width: ${progressPct}%"></div>
                        </div>
                    ` : ''}
                </div>
                <div class="queue-item-actions">
                    <button class="queue-item-btn view-item-btn ${isActive ? 'primary' : ''}" data-id="${item.id}" type="button" title="Xem video này trên player">
                        ${isActive ? 'Đang xem' : 'Xem'}
                    </button>
                    ${item.status === 'done' && item.processedUrl ? `
                        <a class="queue-item-btn download-link" href="${item.processedUrl}" download="${item.name.replace(/\.[^.]+$/, '')}_clean.mp4" title="Tải xuống video đã xử lý">
                            Tải về
                        </a>
                    ` : `
                        <button class="queue-item-btn process-item-btn" data-id="${item.id}" type="button" ${state.running || state.batchRunning ? 'disabled' : ''} title="Xử lý video này">
                            Xử lý
                        </button>
                    `}
                    <button class="queue-item-btn delete-item-btn" data-id="${item.id}" type="button" ${state.batchRunning ? 'disabled' : ''} title="Xóa khỏi danh sách">
                        ✕
                    </button>
                </div>
            </div>
        `;
    }).join('');

    updateBatchControls();
}

function renderQueueItemProgress(item) {
    const itemEl = els.videoQueueList.querySelector(`.queue-item[data-id="${item.id}"]`);
    if (!itemEl) return;
    const progressFill = itemEl.querySelector('.queue-item-progress-fill');
    if (progressFill) {
        progressFill.style.width = `${Math.round((item.progress || 0) * 100)}%`;
    }
    const badge = itemEl.querySelector('.queue-item-status-badge');
    if (badge && item.progressText) {
        badge.textContent = item.progressText;
    }
}

function setActiveVideo(id) {
    const item = state.queue.find((q) => q.id === id);
    if (!item) return;

    state.activeId = id;
    state.file = item.file;
    state.originalUrl = item.originalUrl;
    state.processedUrl = item.processedUrl;
    state.metadata = item.metadata;
    state.detection = item.detection;

    els.originalVideo.src = item.originalUrl;
    els.originalVideo.currentTime = 0;
    els.originalEmpty.hidden = true;

    if (item.processedUrl) {
        els.processedVideo.src = item.processedUrl;
        els.processedVideo.load();
        els.processedEmpty.hidden = true;
        els.downloadBtn.href = item.processedUrl;
        els.downloadBtn.download = `${item.name.replace(/\.[^.]+$/, '')}_clean.mp4`;
    } else {
        els.processedVideo.removeAttribute('src');
        els.processedVideo.load();
        els.processedEmpty.hidden = false;
        els.downloadBtn.removeAttribute('href');
        els.downloadBtn.removeAttribute('download');
    }

    updateCompareMode();
    renderMetadata(item.metadata);
    renderDetection(item.detection);
    if (item.detection && item.metadata) {
        applyAutomaticPreset(item.detection, item.metadata, { silent: true });
    }
    setProgress(item.progress || 0, item.progressText || (item.status === 'done' ? 'Hoàn thành' : 'Sẵn sàng'));
    updateButtons();
    renderQueueList();
}

function removeVideoFromQueue(id) {
    const index = state.queue.findIndex((q) => q.id === id);
    if (index === -1) return;

    const [removed] = state.queue.splice(index, 1);
    if (removed.originalUrl) URL.revokeObjectURL(removed.originalUrl);
    if (removed.processedUrl) URL.revokeObjectURL(removed.processedUrl);

    if (state.activeId === id) {
        if (state.queue.length > 0) {
            const nextIndex = Math.min(index, state.queue.length - 1);
            setActiveVideo(state.queue[nextIndex].id);
        } else {
            reset();
        }
    } else {
        renderQueueList();
        updateBatchControls();
    }
}

function clearQueue() {
    if (state.batchRunning) return;
    for (const item of state.queue) {
        if (item.originalUrl) URL.revokeObjectURL(item.originalUrl);
        if (item.processedUrl) URL.revokeObjectURL(item.processedUrl);
    }
    state.queue = [];
    state.activeId = null;
    els.batchProgressWrapper.style.display = 'none';
    reset();
    renderQueueList();
}

async function addVideosToQueue(files) {
    const validFiles = Array.from(files || []).filter((f) => getDebugFileKind(f) === 'video');
    if (validFiles.length === 0) {
        setStatus('Không tìm thấy tệp video hợp lệ (MP4, WebM, MOV).', 'warn');
        return;
    }

    const newItems = validFiles.map((file) => createQueueItem(file));
    state.queue.push(...newItems);
    renderQueueList();

    if (!state.activeId && newItems.length > 0) {
        setActiveVideo(newItems[0].id);
    }

    setStatus(`Đã thêm ${newItems.length} video vào hàng đợi. Nhấn "Xử lý tất cả" hoặc chọn video để xử lý.`, 'success');

    for (const item of newItems) {
        inspectGeminiVideoFile(item.file).then((metadata) => {
            item.metadata = metadata;
            renderQueueList();
            if (state.activeId === item.id) {
                state.metadata = metadata;
                renderMetadata(metadata);
                applyAutomaticPreset(null, metadata, { silent: true });
            }
        }).catch((err) => {
            console.warn('Lỗi đọc metadata:', item.name, err);
        });
    }
}

async function addFiles(files) {
    const list = Array.from(files || []).filter(Boolean);
    const videoFiles = list.filter((f) => getDebugFileKind(f) === 'video');
    const imageFiles = list.filter((f) => getDebugFileKind(f) === 'image');

    if (imageFiles.length > 0 && videoFiles.length === 0) {
        await routeImageFile(imageFiles[0]);
        return;
    }

    if (videoFiles.length > 0) {
        await addVideosToQueue(videoFiles);
        return;
    }

    setStatus('Vui lòng chọn tệp ảnh hoặc video. Video được xử lý tại trang này, ảnh sẽ chuyển về trang so sánh ảnh.', 'warn');
}

function cleanupUrls() {
    for (const item of state.queue) {
        if (item.originalUrl) URL.revokeObjectURL(item.originalUrl);
        if (item.processedUrl) URL.revokeObjectURL(item.processedUrl);
    }
    if (state.originalUrl) URL.revokeObjectURL(state.originalUrl);
    if (state.processedUrl) URL.revokeObjectURL(state.processedUrl);
    state.originalUrl = null;
    state.processedUrl = null;
}

async function setFile(file) {
    await addFiles([file]);
}

async function routeImageFile(file) {
    try {
        setStatus('Đang chuyển sang quy trình gỡ lỗi ảnh...');
        await saveDebugFileHandoff(file, 'image');
        window.location.assign('./dev-preview.html?fileHandoff=1');
    } catch (error) {
        console.error(error);
        setStatus(error.message || 'Không thể chuyển sang quy trình gỡ lỗi ảnh, hãy mở trang ảnh rồi chọn lại tệp.', 'warn');
    }
}

function getDebugAlphaOptions() {
    return {
        alphaProfile: typeof window.__gwrVideoAlphaProfile === 'string'
            ? window.__gwrVideoAlphaProfile
            : undefined,
        alphaLowScale: Number.isFinite(window.__gwrVideoAlphaLowScale)
            ? window.__gwrVideoAlphaLowScale
            : undefined,
        alphaBodyScale: Number.isFinite(window.__gwrVideoAlphaBodyScale)
            ? window.__gwrVideoAlphaBodyScale
            : undefined,
        alphaEdgeBoost: Number.isFinite(window.__gwrVideoAlphaEdgeBoost)
            ? window.__gwrVideoAlphaEdgeBoost
            : undefined,
        alphaLocalRegion: typeof window.__gwrVideoAlphaLocalRegion === 'string'
            ? window.__gwrVideoAlphaLocalRegion
            : undefined,
        alphaLocalLowScale: Number.isFinite(window.__gwrVideoAlphaLocalLowScale)
            ? window.__gwrVideoAlphaLocalLowScale
            : undefined,
        alphaLocalBodyScale: Number.isFinite(window.__gwrVideoAlphaLocalBodyScale)
            ? window.__gwrVideoAlphaLocalBodyScale
            : undefined
    };
}

async function processSingleItem(item) {
    if (!item) return;
    const jobId = ++state.jobId;
    item.status = 'processing';
    item.progress = 0.04;
    item.progressText = 'Bắt đầu...';
    renderQueueList();

    try {
        let detectionPayload = item.detection ? { metadata: item.metadata, detection: item.detection } : null;
        if (!detectionPayload) {
            item.progressText = 'Đang phát hiện watermark...';
            renderQueueItemProgress(item);
            await yieldToBrowserFrame();
            const detected = await detectGeminiVideoWatermark(item.file, {
                ...getDebugAlphaOptions(),
                sampleCount: Number(els.sampleCount.value) || DEFAULT_SAMPLE_COUNT,
                onProgress: createDetectionProgressHandler(jobId, { start: 0.04, span: 0.08 }),
                yieldToMainThread: yieldToBrowserFrame
            });
            if (jobId !== state.jobId && !state.batchRunning) return;
            item.metadata = detected.metadata;
            item.detection = detected.detection;
            detectionPayload = { metadata: detected.metadata, detection: detected.detection };
        }

        if (state.activeId === item.id) {
            state.metadata = item.metadata;
            state.detection = item.detection;
            renderMetadata(item.metadata);
            renderDetection(item.detection);
            applyAutomaticPreset(item.detection, item.metadata, { silent: true });
        }

        applyDebugControlOverrides();
        const denoiseBackend = els.denoiseBackend.value || DEFAULT_DENOISE_BACKEND;
        const allenkFdncnnRuntimeProfile = resolveAllenkFdncnnRuntimeProfile(detectionPayload?.detection?.position);
        const allenkFdncnnSigma = resolveDetectionAllenkFdncnnSigma(detectionPayload?.detection);
        const allenkFdncnnPadding = resolveDetectionAllenkFdncnnPadding(
            detectionPayload?.detection,
            allenkFdncnnRuntimeProfile
        );
        const allenkFdncnnRuntime = await resolveExportDenoiseRuntime(denoiseBackend, allenkFdncnnRuntimeProfile);
        const allenkFdncnnTemporalReuse = getAllenkFdncnnTemporalReuseConfig(allenkFdncnnRuntime);
        const debugAlphaOptions = getDebugAlphaOptions();

        item.progressText = 'Đang xử lý khung hình...';
        renderQueueItemProgress(item);

        const result = await removeGeminiVideoWatermark(item.file, {
            alphaGain: Number(els.alphaGain.value) || DEFAULT_ALPHA_GAIN,
            adaptiveAlpha: els.adaptiveAlpha.checked,
            highQualityCleanup: els.highQualityCleanup.checked,
            denoiseBackend,
            edgeDenoiseStrength: Number(els.edgeDenoiseStrength.value) || 0,
            residualCleanupStrength: Number(els.residualCleanup.value) || 0,
            videoBitrate: Number(els.videoBitrateMbps.value) > 0
                ? Number(els.videoBitrateMbps.value) * 1000 * 1000
                : DEFAULT_VIDEO_BITRATE,
            ...debugAlphaOptions,
            sampleCount: Number(els.sampleCount.value) || DEFAULT_SAMPLE_COUNT,
            detection: detectionPayload,
            allowLowConfidence: els.allowLowConfidence.checked,
            allenkFdncnnRuntime,
            allenkFdncnnSigma,
            allenkFdncnnPadding,
            allenkFdncnnTemporalReuse,
            yieldToMainThread: yieldToBrowserFrame,
            onProgress: ({ phase, progress, processedFrames, frameEstimate }) => {
                if (phase === 'detect') {
                    item.progress = Math.min(0.12, progress * 0.12);
                    item.progressText = 'Phát hiện';
                } else if (phase === 'export') {
                    item.progress = 0.12 + progress * 0.88;
                    const frameStr = Number.isFinite(processedFrames)
                        ? `${processedFrames}${Number.isFinite(frameEstimate) ? `/${frameEstimate}` : ''} frame`
                        : `${Math.round(progress * 100)}%`;
                    item.progressText = `Xuất video (${frameStr})`;
                }
                renderQueueItemProgress(item);
                if (state.activeId === item.id) {
                    setProgress(item.progress, item.progressText);
                    setStatus(`Đang xử lý ${item.name}: ${item.progressText}`);
                }
            }
        });

        if (item.processedUrl) URL.revokeObjectURL(item.processedUrl);
        item.processedBlob = result.blob;
        item.processedUrl = URL.createObjectURL(result.blob);
        item.status = 'done';
        item.progress = 1;
        item.progressText = 'Hoàn thành';

        if (state.activeId === item.id) {
            state.processedUrl = item.processedUrl;
            els.processedVideo.src = item.processedUrl;
            els.processedVideo.load();
            els.processedEmpty.hidden = true;
            updateCompareMode();
            syncProcessedToOriginal({ force: true });
            els.downloadBtn.href = item.processedUrl;
            els.downloadBtn.download = `${item.name.replace(/\.[^.]+$/, '')}_clean.mp4`;
            setProgress(1, 'Hoàn thành');
            const audioNote = result.audioCopied
                ? `Giữ nguyên âm thanh (${result.audioCodec || 'mp4a'}).`
                : '';
            setStatus(`Xử lý thành công ${item.name}, đã xuất ${result.processedFrames} khung hình. ${audioNote}`, 'success');
        }
    } catch (error) {
        console.error('Lỗi khi xử lý video:', error);
        item.status = 'error';
        item.progress = 0;
        item.errorMessage = error.message || 'Lỗi xử lý';
        item.progressText = error.message || 'Lỗi';
        if (state.activeId === item.id) {
            setProgress(0, 'Lỗi');
            setStatus(error.message || 'Lỗi xử lý video', 'error');
        }
        throw error;
    } finally {
        renderQueueList();
        updateButtons();
        updateBatchControls();
    }
}

async function processAllQueue() {
    if (state.batchRunning || state.queue.length === 0) return;
    state.batchRunning = true;
    state.cancelRequested = false;
    updateBatchControls();
    updateButtons();

    const pendingItems = state.queue.filter((item) => item.status !== 'done');
    if (pendingItems.length === 0) {
        setStatus('Tất cả video trong hàng đợi đều đã được xử lý xong.', 'success');
        state.batchRunning = false;
        updateBatchControls();
        updateButtons();
        return;
    }

    const totalToProcess = pendingItems.length;
    let completedInBatch = 0;
    els.batchProgressWrapper.style.display = 'block';
    setBatchProgress(0, `Bắt đầu xử lý ${totalToProcess} video...`);

    for (let i = 0; i < pendingItems.length; i++) {
        if (state.cancelRequested) {
            setStatus(`Đã dừng xử lý hàng loạt (${completedInBatch}/${totalToProcess} video).`, 'warn');
            break;
        }

        const item = pendingItems[i];
        setActiveVideo(item.id);
        setBatchProgress(completedInBatch / totalToProcess, `Đang xử lý ${i + 1}/${totalToProcess}: ${item.name}`);

        try {
            await processSingleItem(item);
            completedInBatch++;
        } catch (error) {
            console.error(`Lỗi khi xử lý ${item.name}:`, error);
        }

        setBatchProgress(completedInBatch / totalToProcess, `Đã xong ${completedInBatch}/${totalToProcess} video`);
    }

    state.batchRunning = false;
    updateBatchControls();
    updateButtons();
    renderQueueList();

    if (!state.cancelRequested) {
        setStatus(`Hoàn tất xử lý hàng loạt: ${completedInBatch}/${totalToProcess} video thành công.`, 'success');
    }
}

function cancelBatch() {
    if (!state.batchRunning) return;
    state.cancelRequested = true;
    setStatus('Đang dừng sau khi xử lý xong video hiện tại...', 'warn');
}

async function downloadAllProcessed() {
    const doneItems = state.queue.filter((item) => item.status === 'done' && item.processedUrl);
    if (doneItems.length === 0) return;

    setStatus(`Đang tải xuống ${doneItems.length} video đã hoàn thành...`);
    for (let i = 0; i < doneItems.length; i++) {
        const item = doneItems[i];
        const a = document.createElement('a');
        a.href = item.processedUrl;
        a.download = `${item.name.replace(/\.[^.]+$/, '')}_clean.mp4`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        if (i < doneItems.length - 1) {
            await new Promise((resolve) => setTimeout(resolve, 350));
        }
    }
    setStatus(`Đã tải xuống thành công ${doneItems.length} video.`, 'success');
}

async function runDetection() {
    const activeItem = getActiveItem();
    if (!activeItem || state.running || state.batchRunning) return;
    const jobId = ++state.jobId;
    state.running = true;
    activeItem.status = 'detecting';
    activeItem.progressText = 'Đang phát hiện...';
    renderQueueList();
    updateButtons();
    setProgress(0.05, 'Đang phát hiện');
    setStatus('Đang lấy mẫu khung để phát hiện watermark góc dưới bên phải...');

    try {
        await yieldToBrowserFrame();
        const result = await detectGeminiVideoWatermark(activeItem.file, {
            ...getDebugAlphaOptions(),
            sampleCount: Number(els.sampleCount.value) || DEFAULT_SAMPLE_COUNT,
            onProgress: createDetectionProgressHandler(jobId, { start: 0.05, span: 0.9 }),
            yieldToMainThread: yieldToBrowserFrame
        });
        if (jobId !== state.jobId) return;
        activeItem.metadata = result.metadata;
        activeItem.detection = result.detection;
        activeItem.status = 'idle';
        activeItem.progressText = result.detection.isConfident ? 'Sẵn sàng' : 'Độ tin cậy thấp';
        if (state.activeId === activeItem.id) {
            state.metadata = result.metadata;
            state.detection = result.detection;
            renderMetadata(result.metadata);
            renderDetection(result.detection);
            applyAutomaticPreset(result.detection, result.metadata, { silent: true });
        }
        setProgress(1, result.detection.isConfident ? 'Phát hiện xong' : 'Độ tin cậy thấp');
        const preset = applyAutomaticPreset(result.detection, result.metadata, { silent: true });
        if (preset.id === 'relocated-review') {
            setStatus('Phát hiện xong, khi xuất sẽ dùng AI để xóa watermark.', result.detection.isConfident ? 'success' : 'warn');
        } else {
            setStatus(result.detection.isConfident ? 'Phát hiện xong, khi xuất sẽ dùng AI để xóa watermark.' : 'Độ tin cậy phát hiện thấp, vẫn có thể thử xuất bằng AI.', result.detection.isConfident ? 'success' : 'warn');
        }
    } catch (error) {
        console.error(error);
        activeItem.status = 'error';
        activeItem.errorMessage = error.message || 'Phát hiện thất bại';
        activeItem.progressText = 'Lỗi phát hiện';
        setStatus(error.message || 'Phát hiện thất bại', 'error');
        setProgress(0, 'Phát hiện thất bại');
    } finally {
        state.running = false;
        renderQueueList();
        updateButtons();
    }
}

async function runExport() {
    const activeItem = getActiveItem();
    if (!activeItem || state.running || state.batchRunning) return;
    state.running = true;
    updateButtons();
    setProgress(0, 'Bắt đầu');
    setStatus('Đang xử lý từng khung hình cục bộ, hãy giữ trang này mở.');

    try {
        await processSingleItem(activeItem);
    } catch (error) {
        // error already handled in processSingleItem
    } finally {
        state.running = false;
        updateButtons();
    }
}

function reset() {
    state.jobId++;
    if (state.originalUrl && !state.queue.some((q) => q.originalUrl === state.originalUrl)) {
        URL.revokeObjectURL(state.originalUrl);
    }
    if (state.processedUrl && !state.queue.some((q) => q.processedUrl === state.processedUrl)) {
        URL.revokeObjectURL(state.processedUrl);
    }
    state.file = null;
    state.metadata = null;
    state.detection = null;
    state.running = false;
    els.fileInput.value = '';
    els.originalVideo.removeAttribute('src');
    els.originalVideo.load();
    els.processedVideo.removeAttribute('src');
    els.processedVideo.load();
    els.downloadBtn.removeAttribute('href');
    els.downloadBtn.removeAttribute('download');
    els.originalEmpty.hidden = false;
    updateCompareMode();
    renderMetadata(null);
    renderDetection(null);
    renderAutoPresetSummary(null);
    setProgress(0, 'Chờ video');
    setStatus('');
    updateButtons();
}

function setNumberControl(input, value) {
    if (input.hasAttribute('max') && Number(value) > Number(input.getAttribute('max'))) {
        input.setAttribute('max', String(value));
    }
    input.value = String(value);
    if (input === els.alphaGain) {
        els.alphaGainValue.textContent = Number(input.value).toFixed(2);
    } else if (input === els.residualCleanup) {
        els.residualCleanupValue.textContent = Number(input.value).toFixed(2);
    } else if (input === els.edgeDenoiseStrength) {
        els.edgeDenoiseStrengthValue.textContent = Number(input.value).toFixed(2);
    }
    input.dispatchEvent(new Event('input', { bubbles: true }));
}

function applyPresetToControls(preset) {
    if (!preset) return;
    setNumberControl(els.alphaGain, preset.alphaGain ?? DEFAULT_ALPHA_GAIN);
    els.adaptiveAlpha.checked = preset.adaptiveAlpha ?? DEFAULT_ADAPTIVE_ALPHA;
    els.highQualityCleanup.checked = preset.highQualityCleanup ?? DEFAULT_HIGH_QUALITY_CLEANUP;
    els.denoiseBackend.value = Object.values(VIDEO_DENOISE_BACKENDS).includes(preset.denoiseBackend)
        ? preset.denoiseBackend
        : DEFAULT_DENOISE_BACKEND;
    els.denoiseBackend.dispatchEvent(new Event('change', { bubbles: true }));
    setNumberControl(els.edgeDenoiseStrength, preset.edgeDenoiseStrength ?? DEFAULT_EDGE_DENOISE_STRENGTH);
    setNumberControl(els.residualCleanup, preset.residualCleanupStrength ?? DEFAULT_RESIDUAL_CLEANUP_STRENGTH);
    els.sampleCount.value = String(preset.sampleCount ?? DEFAULT_SAMPLE_COUNT);
    els.videoBitrateMbps.value = Number(preset.videoBitrateMbps) > 0
        ? String(preset.videoBitrateMbps)
        : '';
    els.allowLowConfidence.checked = preset.allowLowConfidence === true;
    renderAutoPresetSummary(preset);
}

function applyAutomaticPreset(detection = state.detection, metadata = state.metadata, { silent = false } = {}) {
    const preset = getAutomaticVideoPresetConfig(detection, metadata);
    applyPresetToControls(preset);
    if (!silent) {
        setStatus(`Đã tự động chọn: ${preset.label}.`, preset.allowLowConfidence ? 'warn' : 'success');
    }
    return preset;
}

function applyDebugControlOverrides() {
    applyVideoAdaptiveAlphaDebugOverride({
        windowObject: window,
        adaptiveAlphaInput: els.adaptiveAlpha
    });
    if (typeof window.__gwrVideoOverrideDenoiseBackend === 'string') {
        if (els.denoiseBackend.value !== window.__gwrVideoOverrideDenoiseBackend) {
            els.denoiseBackend.value = window.__gwrVideoOverrideDenoiseBackend;
            els.denoiseBackend.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
    if (typeof window.__gwrVideoOverrideAllowLowConfidence === 'boolean') {
        els.allowLowConfidence.checked = window.__gwrVideoOverrideAllowLowConfidence;
    }
    if (Number.isFinite(window.__gwrVideoOverrideEdgeDenoiseStrength)) {
        setNumberControl(
            els.edgeDenoiseStrength,
            Math.max(0, Math.min(3, window.__gwrVideoOverrideEdgeDenoiseStrength))
        );
    }
    if (Number.isFinite(window.__gwrVideoOverrideResidualCleanupStrength)) {
        setNumberControl(
            els.residualCleanup,
            Math.max(0, Math.min(1.8, window.__gwrVideoOverrideResidualCleanupStrength))
        );
    }
    applyVideoBitrateDebugOverride({
        windowObject: window,
        videoBitrateInput: els.videoBitrateMbps,
        setNumberControl
    });
}

function applyRelocatedReviewPreset() {
    const preset = getRelocatedReviewPresetConfig();
    applyPresetToControls(preset);
    setStatus('Đã áp dụng cấu hình rà soát neo di chuyển: Canvas đánh bóng vết, 12Mbps, cho phép độ tin cậy thấp. Cấu hình này dùng để rà soát thủ công, không phải chiến lược mặc định.', 'warn');
}

function setupEvents() {
    els.dropzone.addEventListener('click', () => els.fileInput.click());
    els.dropzone.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            els.fileInput.click();
        }
    });
    els.fileInput.addEventListener('change', async (event) => {
        const files = Array.from(event.target.files || []);
        if (files.length > 0) await addFiles(files);
        els.fileInput.value = '';
    });

    const dropTargets = [els.dropzone, els.queuePanel].filter(Boolean);
    for (const dropTarget of dropTargets) {
        for (const eventName of ['dragenter', 'dragover']) {
            dropTarget.addEventListener(eventName, (event) => {
                event.preventDefault();
                els.dropzone.dataset.dragging = 'true';
            });
        }
        for (const eventName of ['dragleave', 'drop']) {
            dropTarget.addEventListener(eventName, (event) => {
                event.preventDefault();
                els.dropzone.dataset.dragging = 'false';
            });
        }
        dropTarget.addEventListener('drop', async (event) => {
            const files = Array.from(event.dataTransfer?.files || []);
            if (files.length > 0) await addFiles(files);
        });
    }

    els.processAllBtn.addEventListener('click', processAllQueue);
    els.cancelBatchBtn.addEventListener('click', cancelBatch);
    els.downloadAllBtn.addEventListener('click', downloadAllProcessed);
    els.clearQueueBtn.addEventListener('click', clearQueue);

    els.videoQueueList.addEventListener('click', async (event) => {
        const itemEl = event.target.closest('.queue-item');
        if (!itemEl) return;
        const id = itemEl.dataset.id;
        if (!id) return;

        if (event.target.closest('.delete-item-btn')) {
            event.stopPropagation();
            removeVideoFromQueue(id);
            return;
        }

        if (event.target.closest('.process-item-btn')) {
            event.stopPropagation();
            const item = state.queue.find((q) => q.id === id);
            if (item && !state.running && !state.batchRunning) {
                setActiveVideo(id);
                state.running = true;
                updateButtons();
                try {
                    await processSingleItem(item);
                } catch (err) {
                    // handled inside processSingleItem
                } finally {
                    state.running = false;
                    updateButtons();
                }
            }
            return;
        }

        if (event.target.closest('.download-link')) {
            event.stopPropagation();
            return;
        }

        setActiveVideo(id);
    });

    els.alphaGain.addEventListener('input', () => {
        els.alphaGainValue.textContent = Number(els.alphaGain.value).toFixed(2);
    });
    els.residualCleanup.addEventListener('input', () => {
        els.residualCleanupValue.textContent = Number(els.residualCleanup.value).toFixed(2);
    });
    els.edgeDenoiseStrength.addEventListener('input', () => {
        els.edgeDenoiseStrengthValue.textContent = Number(els.edgeDenoiseStrength.value).toFixed(2);
    });
    els.denoiseBackend.addEventListener('change', () => {
        if (els.denoiseBackend.value !== VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE) return;
        setNumberControl(els.edgeDenoiseStrength, 1.8);
    });
    els.detectBtn.addEventListener('click', runDetection);
    els.processBtn.addEventListener('click', runExport);
    els.resetBtn.addEventListener('click', reset);
    els.relocatedReviewPresetBtn.addEventListener('click', applyRelocatedReviewPreset);
    els.downloadBtn.addEventListener('click', (event) => {
        if (!state.processedUrl || state.running) event.preventDefault();
    });
    els.playPauseBtn.addEventListener('click', togglePlayback);
    els.scrubber.addEventListener('input', (event) => {
        seekComparison(event.target.value);
    });
    els.originalVideo.addEventListener('loadedmetadata', () => {
        updatePlaybackControls();
    });
    els.originalVideo.addEventListener('timeupdate', () => {
        if (!els.originalVideo.paused) syncProcessedToOriginal();
        updatePlaybackControls();
    });
    els.originalVideo.addEventListener('pause', () => {
        if (!els.processedVideo.paused) els.processedVideo.pause();
        updatePlaybackControls();
    });
    els.originalVideo.addEventListener('ended', () => {
        pauseComparison();
    });
    els.processedVideo.addEventListener('loadedmetadata', () => {
        syncProcessedToOriginal({ force: true });
        updateCompareMode();
    });
    window.addEventListener('beforeunload', cleanupUrls);
}

async function consumePendingVideoHandoff() {
    const params = new URLSearchParams(window.location.search);
    if (params.get('fileHandoff') !== '1') return;

    try {
        const record = await consumeDebugFileHandoff('video');
        if (!record) return;
        const files = Array.isArray(record.files) && record.files.length > 0
            ? record.files
            : record.file ? [record.file] : [];
        if (files.length > 0) {
            await addFiles(files);
        }
        window.history.replaceState(null, '', window.location.pathname);
    } catch (error) {
        console.warn('video handoff unavailable:', error);
        setStatus(error.message || 'Không đọc được video tạm, vui lòng chọn lại tệp.', 'warn');
    }
}

async function init() {
    applyPresetToControls(getAutomaticVideoPresetConfig());

    if (!('VideoDecoder' in window) || !('VideoEncoder' in window)) {
        setStatus('Trình duyệt hiện tại không có WebCodecs, hãy dùng Chrome hoặc Edge bản mới.', 'error');
    }

    renderMetadata(null);
    renderDetection(null);
    updateCompareMode();
    setProgress(0, 'Chờ video');
    setupEvents();
    updateButtons();
    await consumePendingVideoHandoff();
}

init();
