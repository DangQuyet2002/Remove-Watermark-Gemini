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

const FORCED_RESCUE_MIN_SPATIAL = 0.18;
const FORCED_RESCUE_MIN_GRADIENT = 0.11;
const FORCED_RESCUE_MIN_PROMINENCE = 0.075;
const FORCED_RESCUE_ADAPTIVE_MIN_SPATIAL = 0.22;
const FORCED_RESCUE_ADAPTIVE_MIN_GRADIENT = 0.12;
const FORCED_RESCUE_ADAPTIVE_MIN_PROMINENCE = 0.09;

const EXTRA_THOROUGH_CANDIDATE_CONFIGS = Object.freeze([
    Object.freeze({ logoSize: 48, marginRight: 96, marginBottom: 96 }),
    Object.freeze({ logoSize: 48, marginRight: 32, marginBottom: 32 }),
    Object.freeze({ logoSize: 48, marginRight: 72, marginBottom: 72 }),
    Object.freeze({ logoSize: 48, marginRight: 73, marginBottom: 73 }),
    Object.freeze({ logoSize: 48, marginRight: 73, marginBottom: 73, alphaVariant: 'v2' }),
    Object.freeze({ logoSize: 48, marginRight: 96, marginBottom: 96, alphaVariant: 'v2' }),
    Object.freeze({ logoSize: 52, marginRight: 102, marginBottom: 102, alphaVariant: 'v2' }),
    Object.freeze({ logoSize: 36, marginRight: 96, marginBottom: 96, alphaVariant: 'v2' }),
    Object.freeze({ logoSize: 72, marginRight: 96, marginBottom: 96 }),
    Object.freeze({ logoSize: 72, marginRight: 108, marginBottom: 108 }),
    Object.freeze({ logoSize: 72, marginRight: 144, marginBottom: 144 }),
    Object.freeze({ logoSize: 72, marginRight: 144, marginBottom: 144, alphaVariant: 'v2' }),
    Object.freeze({ logoSize: 96, marginRight: 64, marginBottom: 64 }),
    Object.freeze({ logoSize: 96, marginRight: 128, marginBottom: 128 }),
    Object.freeze({ logoSize: 96, marginRight: 192, marginBottom: 192 }),
    Object.freeze({ logoSize: 96, marginRight: 192, marginBottom: 192, alphaVariant: '20260520' }),
    Object.freeze({ logoSize: 96, marginRight: 192, marginBottom: 192, alphaVariant: 'v2' }),
    Object.freeze({ logoSize: 44, marginRight: 29, marginBottom: 40 })
]);

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

    if (variant === 'v2') {
        const base36 = await engine.getAlphaMap('36-v2');
        if (!base36 || base36.length !== 36 * 36) return null;
        return size === 36 ? base36 : interpolateAlphaMap(base36, 36, size);
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

function measureLocalSpatialProminence(imageData, alphaMap, position, spatialScore) {
    const { width, height } = imageData;
    const size = position.width;
    let maxControl = -Infinity;
    for (const [dx, dy] of [[-1, 0], [0, -1], [-1, -1], [-2, 0], [0, -2]]) {
        const cx = position.x + dx * size;
        const cy = position.y + dy * size;
        if (cx < 0 || cy < 0 || cx + size > width || cy + size > height) continue;
        const controlScore = computeRegionSpatialCorrelation({
            imageData,
            alphaMap,
            region: { x: cx, y: cy, size }
        });
        if (Number.isFinite(controlScore) && controlScore > maxControl) {
            maxControl = controlScore;
        }
    }
    return Number.isFinite(maxControl) ? spatialScore - maxControl : Infinity;
}

function overlapsExistingRegion(position, existingRegions) {
    for (const existing of existingRegions) {
        const ep = existing.position;
        const ix = Math.max(0, Math.min(position.x + position.width, ep.x + ep.width) - Math.max(position.x, ep.x));
        const iy = Math.max(0, Math.min(position.y + position.height, ep.y + ep.height) - Math.max(position.y, ep.y));
        const inter = ix * iy;
        const minArea = Math.min(position.width * position.height, ep.width * ep.height);
        if (minArea > 0 && inter / minArea > 0.25) return true;
    }
    return false;
}

function extractPatchScratch(imageData, position) {
    const out = {
        width: position.width,
        height: position.height,
        data: new Uint8ClampedArray(position.width * position.height * 4)
    };
    for (let row = 0; row < position.height; row++) {
        const srcStart = ((position.y + row) * imageData.width + position.x) * 4;
        out.data.set(
            imageData.data.subarray(srcStart, srcStart + position.width * 4),
            row * position.width * 4
        );
    }
    return out;
}

function buildThoroughRescueCatalogConfigs(canvasWidth, canvasHeight, defaultConfig) {
    const rawConfigs = [
        ...resolveGeminiWatermarkSearchCatalogEntries(canvasWidth, canvasHeight, defaultConfig)
            .map((entry) => entry?.config)
            .filter(Boolean),
        ...EXTRA_THOROUGH_CANDIDATE_CONFIGS
    ];

    const longSide = Math.max(canvasWidth, canvasHeight);
    const shortSide = Math.min(canvasWidth, canvasHeight);
    if (longSide >= 768) {
        const scaledV2Margin = Math.max(24, Math.round(192 * (longSide / (shortSide >= 566 ? 2752 : 2816))));
        rawConfigs.push({
            logoSize: 36,
            marginRight: scaledV2Margin,
            marginBottom: scaledV2Margin,
            alphaVariant: 'v2'
        });
    }

    const seen = new Set();
    const deduped = [];
    for (const cfg of rawConfigs) {
        const key = `${cfg.logoSize}:${cfg.marginRight}:${cfg.marginBottom}:${cfg.alphaVariant ?? 'default'}`;
        if (seen.has(key)) continue;
        seen.add(key);
        deduped.push(cfg);
    }
    return deduped;
}

async function findSingleForcedRescueCandidate(imageData, engine, canvasWidth, canvasHeight, existingRegions = []) {
    const defaultConfig = detectWatermarkConfig(canvasWidth, canvasHeight);
    const alpha96 = await engine.getAlphaMap(96);
    const configs = buildThoroughRescueCatalogConfigs(canvasWidth, canvasHeight, defaultConfig);

    let bestCandidate = null;
    for (const cfg of configs) {
        const size = cfg.logoSize;
        const basePos = {
            x: canvasWidth - cfg.marginRight - size,
            y: canvasHeight - cfg.marginBottom - size,
            width: size,
            height: size
        };
        if (!isValidRegionPosition(basePos, canvasWidth, canvasHeight)) continue;
        if (overlapsExistingRegion(basePos, existingRegions)) continue;

        const alphaMap = await resolveRegionAlphaMap(engine, { config: cfg, position: basePos }).catch(() => null);
        if (!alphaMap || alphaMap.length !== size * size) continue;

        let localBest = null;
        for (const [dx, dy] of [[0, 0], [-2, 0], [2, 0], [0, -2], [0, 2]]) {
            const position = {
                x: basePos.x + dx,
                y: basePos.y + dy,
                width: size,
                height: size
            };
            if (!isValidRegionPosition(position, canvasWidth, canvasHeight)) continue;
            if (overlapsExistingRegion(position, existingRegions)) continue;

            const spatialScore = computeRegionSpatialCorrelation({
                imageData,
                alphaMap,
                region: { x: position.x, y: position.y, size }
            });
            if (!Number.isFinite(spatialScore) || spatialScore < FORCED_RESCUE_MIN_SPATIAL) continue;

            const gradientScore = computeRegionGradientCorrelation({
                imageData,
                alphaMap,
                region: { x: position.x, y: position.y, size }
            });
            if (!Number.isFinite(gradientScore) || gradientScore < FORCED_RESCUE_MIN_GRADIENT) continue;

            const prominence = measureLocalSpatialProminence(imageData, alphaMap, position, spatialScore);
            if (prominence < FORCED_RESCUE_MIN_PROMINENCE) continue;

            const combined = spatialScore * 0.6 + gradientScore * 0.4;
            if (!localBest || combined > localBest.combined) {
                localBest = {
                    position,
                    config: {
                        ...cfg,
                        marginRight: canvasWidth - position.x - size,
                        marginBottom: canvasHeight - position.y - size
                    },
                    alphaMap,
                    spatialScore,
                    gradientScore,
                    combined
                };
            }
            if (dx === 0 && dy === 0 && spatialScore < 0.3) break;
        }

        if (localBest && (!bestCandidate || localBest.combined > bestCandidate.combined)) {
            bestCandidate = localBest;
        }
    }

    if (!bestCandidate) {
        const adaptiveSeeds = [defaultConfig];
        if (defaultConfig.logoSize > 56) {
            adaptiveSeeds.push(
                { logoSize: 96, marginRight: 160, marginBottom: 160 },
                { logoSize: 48, marginRight: 80, marginBottom: 80 }
            );
        }

        for (const seedConfig of adaptiveSeeds) {
            const adaptive = detectAdaptiveWatermarkRegion({
                imageData,
                alpha96,
                defaultConfig: seedConfig,
                threshold: 0.35
            });
            if (
                !adaptive?.found ||
                !adaptive.region ||
                adaptive.spatialScore < FORCED_RESCUE_ADAPTIVE_MIN_SPATIAL ||
                adaptive.gradientScore < FORCED_RESCUE_ADAPTIVE_MIN_GRADIENT
            ) {
                continue;
            }
            const size = adaptive.region.size;
            const position = {
                x: adaptive.region.x,
                y: adaptive.region.y,
                width: size,
                height: size
            };
            if (!isValidRegionPosition(position, canvasWidth, canvasHeight)) continue;
            if (overlapsExistingRegion(position, existingRegions)) continue;

            const alphaMap = size === 96 ? alpha96 : interpolateAlphaMap(alpha96, 96, size);
            const prominence = measureLocalSpatialProminence(imageData, alphaMap, position, adaptive.spatialScore);
            if (prominence < FORCED_RESCUE_ADAPTIVE_MIN_PROMINENCE) continue;

            const combined = adaptive.spatialScore * 0.6 + adaptive.gradientScore * 0.4;
            if (!bestCandidate || combined > bestCandidate.combined) {
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
                    combined
                };
            }
        }
    }

    if (!bestCandidate) return null;

    const basePatch = extractPatchScratch(imageData, bestCandidate.position);
    const localPos = {
        x: 0,
        y: 0,
        width: bestCandidate.position.width,
        height: bestCandidate.position.height
    };
    const localRegion = { x: 0, y: 0, size: bestCandidate.position.width };
    let bestGain = 1;
    let bestAbsResidual = Infinity;
    for (const gain of [0.45, 0.6, 0.75, 0.9, 1.0, 1.12, 1.25]) {
        const probe = {
            width: basePatch.width,
            height: basePatch.height,
            data: new Uint8ClampedArray(basePatch.data)
        };
        removeWatermark(probe, bestCandidate.alphaMap, localPos, { alphaGain: gain });
        const residual = Math.abs(computeRegionSpatialCorrelation({
            imageData: probe,
            alphaMap: bestCandidate.alphaMap,
            region: localRegion
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

    return {
        position: bestCandidate.position,
        config: bestCandidate.config,
        qualityStatus: 'visible-residual',
        source: 'thorough-forced-rescue',
        alphaGain: bestGain
    };
}

async function tryForcedThoroughRescue(ctx, engine, canvasWidth, canvasHeight) {
    const imageData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    const rescuedRegions = [];

    for (let pass = 0; pass < 2; pass++) {
        const rescued = await findSingleForcedRescueCandidate(
            imageData,
            engine,
            canvasWidth,
            canvasHeight,
            rescuedRegions
        );
        if (!rescued) break;
        rescuedRegions.push(rescued);
    }

    if (rescuedRegions.length === 0) return [];
    ctx.putImageData(imageData, 0, 0);
    return rescuedRegions;
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
        const rescuedList = await tryForcedThoroughRescue(ctx, engine, canvasWidth, canvasHeight);
        if (rescuedList.length > 0) {
            regions = rescuedList;
            const primary = rescuedList[0];
            nextMeta = {
                ...nextMeta,
                applied: true,
                decisionTier: 'rescued-thorough',
                position: primary.position,
                size: primary.position.width,
                config: primary.config,
                source: rescuedList.length > 1
                    ? `${primary.source}+multi-region`
                    : primary.source,
                qualityStatus: 'clean',
                ...(rescuedList.length > 1
                    ? {
                        processedRegionCount: rescuedList.length,
                        watermarkRegions: rescuedList.map((r) => ({
                            position: r.position,
                            config: r.config,
                            source: r.source,
                            qualityStatus: 'clean'
                        }))
                    }
                    : {})
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
            nextMeta?.qualityStatus === 'mixed' ||
            Number(nextMeta?.detection?.processedGradientScore) >= 0.22;
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
                if (thoroughMode && hasVisibleResidual) {
                    applyVideoResidualCleanup(ctx, region.position, alphaMap, {
                        denoiseBackend: VIDEO_DENOISE_BACKENDS.CANVAS_FOOTPRINT_POLISH,
                        edgeDenoiseStrength: 0.55,
                        residualCleanupStrength: 0,
                        highQualityCleanup: true
                    });
                }
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
