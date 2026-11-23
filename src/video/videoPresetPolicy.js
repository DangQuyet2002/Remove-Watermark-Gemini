import {
    DEFAULT_HIGH_QUALITY_CLEANUP,
    VIDEO_DENOISE_BACKENDS
} from './videoCleanupBackends.js';

const RELOCATED_MARGIN_RATIO = 1.8;
const DEFAULT_AUTO_SAMPLE_COUNT = 12;
const DEFAULT_AUTO_ALPHA_GAIN = 1;
const DEFAULT_AUTO_AI_EDGE_DENOISE_STRENGTH = 1.85;
const DEFAULT_AUTO_AI_RESIDUAL_CLEANUP_STRENGTH = 0.55;
const DEFAULT_RELOCATED_AI_EDGE_DENOISE_STRENGTH = 1.85;
const DEFAULT_RELOCATED_AI_RESIDUAL_CLEANUP_STRENGTH = 0.65;
const DEFAULT_VEO_TEXT_AI_EDGE_DENOISE_STRENGTH = 1.6;
const DEFAULT_VEO_TEXT_AI_RESIDUAL_CLEANUP_STRENGTH = 0.95;

export function isRelocatedVideoWatermarkPosition(position) {
    if (!position || !Number.isFinite(position.width) || position.width <= 0) {
        return false;
    }
    const explicitMarginRight = Number(position.marginRight);
    const explicitMarginBottom = Number(position.marginBottom);
    const inferredMarginRight = Number.isFinite(Number(position.videoWidth)) && Number.isFinite(Number(position.x))
        ? Number(position.videoWidth) - Number(position.x) - Number(position.width)
        : null;
    const inferredMarginBottom = Number.isFinite(Number(position.videoHeight)) && Number.isFinite(Number(position.y))
        ? Number(position.videoHeight) - Number(position.y) - Number(position.height || position.width)
        : null;
    const marginRight = Number.isFinite(explicitMarginRight) ? explicitMarginRight : inferredMarginRight;
    const marginBottom = Number.isFinite(explicitMarginBottom) ? explicitMarginBottom : inferredMarginBottom;
    return (
        Number.isFinite(marginRight) && marginRight >= position.width * RELOCATED_MARGIN_RATIO
    ) || (
        Number.isFinite(marginBottom) && marginBottom >= position.width * RELOCATED_MARGIN_RATIO
    );
}

function isRelocatedCandidateLabel(candidate = {}) {
    const text = `${candidate.id || ''} ${candidate.label || ''}`.toLowerCase();
    return text.includes('inset') || text.includes('relocated');
}

export function shouldUseRelocatedReviewPreset(detection, metadata = null) {
    if (!detection?.isConfident || !detection.position) {
        return false;
    }
    const position = {
        ...detection.position,
        videoWidth: detection.position.videoWidth ?? metadata?.width,
        videoHeight: detection.position.videoHeight ?? metadata?.height
    };
    return isRelocatedVideoWatermarkPosition(position) ||
        isRelocatedCandidateLabel(detection.summary?.best);
}

export function getRelocatedReviewPresetConfig() {
    return {
        id: 'relocated-review',
        label: 'Xử lý tự động bằng AI',
        description: 'Tự động phát hiện vị trí watermark (kể cả neo lùi vào trong) và dùng mô hình AI FDnCNN để xóa sạch viền.',
        alphaGain: DEFAULT_AUTO_ALPHA_GAIN,
        adaptiveAlpha: true,
        highQualityCleanup: DEFAULT_HIGH_QUALITY_CLEANUP,
        denoiseBackend: VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE,
        edgeDenoiseStrength: DEFAULT_RELOCATED_AI_EDGE_DENOISE_STRENGTH,
        residualCleanupStrength: DEFAULT_RELOCATED_AI_RESIDUAL_CLEANUP_STRENGTH,
        sampleCount: DEFAULT_AUTO_SAMPLE_COUNT,
        videoBitrateMbps: 12,
        allowLowConfidence: true
    };
}

export function getStandardAutoPresetConfig() {
    return {
        id: 'standard-auto',
        label: 'Xử lý tự động bằng AI',
        description: 'Mặc định dùng mô hình AI cục bộ kết hợp thích nghi Alpha từng khung hình để xóa triệt để watermark Gemini/Veo.',
        alphaGain: DEFAULT_AUTO_ALPHA_GAIN,
        adaptiveAlpha: true,
        highQualityCleanup: DEFAULT_HIGH_QUALITY_CLEANUP,
        denoiseBackend: VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE,
        edgeDenoiseStrength: DEFAULT_AUTO_AI_EDGE_DENOISE_STRENGTH,
        residualCleanupStrength: DEFAULT_AUTO_AI_RESIDUAL_CLEANUP_STRENGTH,
        sampleCount: DEFAULT_AUTO_SAMPLE_COUNT,
        videoBitrateMbps: '',
        allowLowConfidence: false
    };
}

export function getVeoTextAutoPresetConfig() {
    return {
        ...getStandardAutoPresetConfig(),
        id: 'veo-text-auto',
        edgeDenoiseStrength: DEFAULT_VEO_TEXT_AI_EDGE_DENOISE_STRENGTH,
        residualCleanupStrength: DEFAULT_VEO_TEXT_AI_RESIDUAL_CLEANUP_STRENGTH
    };
}

export function getAutomaticVideoPresetConfig(detection = null, metadata = null) {
    if (shouldUseRelocatedReviewPreset(detection, metadata)) {
        return getRelocatedReviewPresetConfig();
    }
    if (detection?.watermarkKind === 'veo-text') {
        return getVeoTextAutoPresetConfig();
    }
    return getStandardAutoPresetConfig();
}
