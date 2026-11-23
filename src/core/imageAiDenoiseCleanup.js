import { removeWatermark } from './blendModes.js';
import {
    computeRegionGradientCorrelation,
    computeRegionSpatialCorrelation,
    detectAdaptiveWatermarkRegion,
    interpolateAlphaMap
} from './adaptiveDetector.js';
import { detectWatermarkConfig } from './watermarkConfig.js';
import { resolveGeminiWatermarkSearchCatalogEntries } from './geminiSizeCatalog.js';
import {
    VIDEO_DENOISE_BACKENDS,
    applyVideoResidualCleanup,
    applyVideoResidualCleanupAsync
} from '../video/videoCleanupBackends.js';
import { resolveAllenkFdncnnRuntimeProfile } from '../video/videoDenoiseRuntimePolicy.js';

const DEFAULT_IMAGE_AI_SIGMA = 45;
const RESIDUAL_IMAGE_AI_SIGMA = 75;
const DEFAULT_IMAGE_AI_EDGE_STRENGTH = 1.45;
const RESIDUAL_IMAGE_AI_EDGE_STRENGTH = 1.85;
const DEFAULT_IMAGE_RESIDUAL_POLISH = 0.55;
const RESIDUAL_IMAGE_RESIDUAL_POLISH = 1.05;

const FORCED_RESCUE_MIN_SPATIAL = 0.2;
const FORCED_RESCUE_MIN_GRADIENT = 0.14;

function isValidRegionPosition(position, width, height) {
    return Boolean(
        position &&
        Number.isFinite(position.x) &&
        Number.isFinite(position.y) &&
        Number.isFinite(position.width) &&
        Number.isFinite(position.height) &&
        position.width > 0 &&
        position.height > 0 &&
        position.x >= 0 &&
        position.y >= 0 &&
        position.x + position.width <= width &&
        position.y + position.height <= height
    );
}

async function resolveRegionAlphaMap(engine, region) {
    const size = Math.round(Number(region?.config?.logoSize ?? region?.position?.width ?? 0));
    if (size <= 0) return null;
    const variant = region?.config?.alphaVariant;

    if (variant === 'v2' && size === 36) {
        return engine.getAlphaMap('36-v2');
    }
    if (variant === '20260520' && size === 96) {
        return engine.getAlphaMap('96-20260520');
    }
    if (variant === 'outline-light' && size === 96) {
        return engine.getAlphaMap('96-outline-light');
    }
    if (variant === 'outline-dark' && size === 96) {
        return engine.getAlphaMap('96-outline-dark');
    }
    return engine.getAlphaMap(size);
}

function collectTargetRegions(meta, canvasWidth, canvasHeight) {
    if (Array.isArray(meta?.watermarkRegions) && meta.watermarkRegions.length > 0) {
        return meta.watermarkRegions.filter((region) =>
            isValidRegionPosition(region?.position, canvasWidth, canvasHeight)
        );
    }

    if (meta?.applied === true && isValidRegionPosition(meta?.position, canvasWidth, canvasHeight)) {
        return [{
            position: meta.position,
            config: meta.config ?? null,
            qualityStatus: meta.qualityStatus ?? 'clean',
            source: meta.source ?? 'processed'
        }];
    }

    return [];
}

async function tryForcedThoroughRescue(ctx, engine, canvasWidth, canvasHeight) {
    const imageData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    const defaultConfig = detectWatermarkConfig(canvasWidth, canvasHeight);
    const alpha96 = await engine.getAlphaMap(96);
    const entries = resolveGeminiWatermarkSearchCatalogEntries(canvasWidth, canvasHeight, defaultConfig);

    let bestCandidate = null;
    for (const entry of entries) {
        const cfg = entry?.config;
        if (!cfg) continue;
        const size = cfg.logoSize;
        const position = {
            x: canvasWidth - cfg.marginRight - size,
            y: canvasHeight - cfg.marginBottom - size,
            width: size,
            height: size
        };
        if (!isValidRegionPosition(position, canvasWidth, canvasHeight)) continue;
        const alphaMap = await resolveRegionAlphaMap(engine, { config: cfg, position }).catch(() => null);
        if (!alphaMap || alphaMap.length !== size * size) continue;

        const spatialScore = computeRegionSpatialCorrelation({
            imageData,
            alphaMap,
            region: { x: position.x, y: position.y, size }
        });
        const gradientScore = computeRegionGradientCorrelation({
            imageData,
            alphaMap,
            region: { x: position.x, y: position.y, size }
        });
        if (spatialScore < FORCED_RESCUE_MIN_SPATIAL || gradientScore < FORCED_RESCUE_MIN_GRADIENT) {
            continue;
        }
        const combined = spatialScore * 0.65 + gradientScore * 0.35;
        if (!bestCandidate || combined > bestCandidate.combined) {
            bestCandidate = { position, config: cfg, alphaMap, spatialScore, gradientScore, combined };
        }
    }

    if (!bestCandidate) {
        const adaptive = detectAdaptiveWatermarkRegion({
            imageData,
            alpha96,
            defaultConfig,
            threshold: 0.36
        });
        if (
            adaptive?.found &&
            adaptive.region &&
            adaptive.spatialScore >= FORCED_RESCUE_MIN_SPATIAL &&
            adaptive.gradientScore >= FORCED_RESCUE_MIN_GRADIENT
        ) {
            const size = adaptive.region.size;
            const position = {
                x: adaptive.region.x,
                y: adaptive.region.y,
                width: size,
                height: size
            };
            if (isValidRegionPosition(position, canvasWidth, canvasHeight)) {
                const alphaMap = size === 96 ? alpha96 : interpolateAlphaMap(alpha96, 96, size);
                bestCandidate = {
                    position,
                    config: {
                        logoSize: size,
                        marginRight: canvasWidth - position.x - size,
                        marginBottom: canvasHeight - position.y - size
                    },
                    alphaMap,
                    spatialScore: adaptive.spatialScore,
                    gradientScore: adaptive.gradientScore,
                    combined: adaptive.confidence
                };
            }
        }
    }

    if (!bestCandidate) return null;

    let bestGain = 1;
    let bestAbsResidual = Infinity;
    for (const gain of [0.6, 0.75, 0.9, 1.0, 1.12, 1.25]) {
        const probe = {
            width: canvasWidth,
            height: canvasHeight,
            data: new Uint8ClampedArray(imageData.data)
        };
        removeWatermark(probe, bestCandidate.alphaMap, bestCandidate.position, { alphaGain: gain });
        const residual = Math.abs(computeRegionSpatialCorrelation({
            imageData: probe,
            alphaMap: bestCandidate.alphaMap,
            region: {
                x: bestCandidate.position.x,
                y: bestCandidate.position.y,
                size: bestCandidate.position.width
            }
        }));
        if (residual < bestAbsResidual) {
            bestAbsResidual = residual;
            bestGain = gain;
        }
    }

    if (bestAbsResidual >= Math.abs(bestCandidate.spatialScore) - 0.05) {
        return null;
    }

    removeWatermark(imageData, bestCandidate.alphaMap, bestCandidate.position, { alphaGain: bestGain });
    ctx.putImageData(imageData, 0, 0);

    return {
        position: bestCandidate.position,
        config: bestCandidate.config,
        qualityStatus: 'visible-residual',
        source: 'thorough-forced-rescue',
        alphaGain: bestGain
    };
}

export async function applyImageThoroughCleanup({
    ctx,
    canvasWidth,
    canvasHeight,
    engine,
    meta,
    options = {}
} = {}) {
    const thoroughMode = options.thoroughMode === true;
    const hasAiRuntime = Boolean(options.aiDenoiseRuntime || typeof options.resolveAiDenoiseRuntime === 'function');
    if (!thoroughMode && !hasAiRuntime) {
        return meta;
    }

    let nextMeta = meta ? { ...meta } : null;
    let regions = collectTargetRegions(nextMeta, canvasWidth, canvasHeight);

    if (regions.length === 0 && thoroughMode && nextMeta?.applied === false) {
        const rescued = await tryForcedThoroughRescue(ctx, engine, canvasWidth, canvasHeight);
        if (rescued) {
            regions = [rescued];
            nextMeta = {
                ...nextMeta,
                applied: true,
                decisionTier: 'rescued-thorough',
                position: rescued.position,
                size: rescued.position.width,
                config: rescued.config,
                source: rescued.source,
                qualityStatus: 'clean'
            };
        }
    }

    if (regions.length === 0) {
        return nextMeta;
    }

    let aiAppliedCount = 0;
    let canvasPolishedCount = 0;

    for (const region of regions) {
        const alphaMap = await resolveRegionAlphaMap(engine, region).catch(() => null);
        if (!alphaMap || alphaMap.length !== region.position.width * region.position.height) {
            continue;
        }

        const hasVisibleResidual =
            region.qualityStatus === 'visible-residual' ||
            region.qualityStatus === 'mixed' ||
            nextMeta?.qualityStatus === 'visible-residual' ||
            nextMeta?.qualityStatus === 'mixed';
        const profile = resolveAllenkFdncnnRuntimeProfile(region.position);
        let runtime = options.aiDenoiseRuntime || null;
        if (!runtime && typeof options.resolveAiDenoiseRuntime === 'function') {
            runtime = await options.resolveAiDenoiseRuntime(profile).catch(() => null);
        }

        if (runtime) {
            const sigma = Number.isFinite(options.aiSigma)
                ? options.aiSigma
                : (hasVisibleResidual ? RESIDUAL_IMAGE_AI_SIGMA : DEFAULT_IMAGE_AI_SIGMA);
            const edgeStrength = Number.isFinite(options.edgeDenoiseStrength)
                ? options.edgeDenoiseStrength
                : (hasVisibleResidual ? RESIDUAL_IMAGE_AI_EDGE_STRENGTH : DEFAULT_IMAGE_AI_EDGE_STRENGTH);
            const residualPolish = Number.isFinite(options.residualCleanupStrength)
                ? options.residualCleanupStrength
                : (hasVisibleResidual ? RESIDUAL_IMAGE_RESIDUAL_POLISH : DEFAULT_IMAGE_RESIDUAL_POLISH);

            const cleanupResult = await applyVideoResidualCleanupAsync(ctx, region.position, alphaMap, {
                denoiseBackend: VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE,
                allenkFdncnnRuntime: runtime,
                allenkFdncnnSigma: sigma,
                allenkFdncnnPadding: profile.padding,
                edgeDenoiseStrength: edgeStrength,
                residualCleanupStrength: residualPolish,
                highQualityCleanup: true
            });
            if (cleanupResult?.denoiseRuntimeStatus === 'applied') {
                aiAppliedCount++;
                continue;
            }
        }

        if (thoroughMode || hasVisibleResidual) {
            applyVideoResidualCleanup(ctx, region.position, alphaMap, {
                denoiseBackend: VIDEO_DENOISE_BACKENDS.CANVAS_FOOTPRINT_POLISH,
                edgeDenoiseStrength: hasVisibleResidual ? 0.95 : 0.65,
                residualCleanupStrength: hasVisibleResidual ? 1.15 : 0.65,
                highQualityCleanup: true
            });
            canvasPolishedCount++;
        }
    }

    if (aiAppliedCount > 0 || canvasPolishedCount > 0) {
        const prevQuality = nextMeta?.qualityStatus;
        const upgradedQuality = (prevQuality === 'visible-residual' || prevQuality === 'mixed')
            ? 'clean'
            : (prevQuality ?? 'clean');
        nextMeta = {
            ...nextMeta,
            qualityStatus: upgradedQuality,
            source: aiAppliedCount > 0
                ? `${nextMeta.source || 'processed'}+ai-fdncnn`
                : `${nextMeta.source || 'processed'}+thorough-polish`,
            aiCleanupApplied: aiAppliedCount > 0,
            aiCleanedRegionCount: aiAppliedCount,
            thoroughCleanupApplied: true
        };
        if (Array.isArray(nextMeta.watermarkRegions)) {
            nextMeta.watermarkRegions = nextMeta.watermarkRegions.map((item) => ({
                ...item,
                qualityStatus: item.qualityStatus === 'visible-residual' || item.qualityStatus === 'mixed'
                    ? 'clean'
                    : item.qualityStatus
            }));
        }
    }

    return nextMeta;
}
