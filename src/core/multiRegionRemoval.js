import {
    computeRegionSpatialCorrelation,
    detectAdaptiveWatermarkRegion
} from './adaptiveDetector.js';
import { cloneImageData, scoreRegion } from './restorationMetrics.js';
import { detectWatermarkConfig } from './watermarkConfig.js';
import {
    matchOfficialGeminiImageSize,
    resolveGeminiWatermarkSearchCatalogEntries
} from './geminiSizeCatalog.js';

const SECONDARY_MIN_SPATIAL_SCORE = 0.28;
const SECONDARY_MIN_GRADIENT_SCORE = 0.11;
const SECONDARY_MIN_LOCAL_PROMINENCE = 0.075;
const SECONDARY_STRONG_SPATIAL_SCORE = 0.78;
const SECONDARY_STRONG_LOCAL_PROMINENCE = 0.18;
const SECONDARY_MAX_OVERLAP_FRACTION = 0.25;
const SECONDARY_DIFF_SIZE_RATIO_THRESHOLD = 0.8;
const SECONDARY_DIFF_SIZE_MAX_IOU = 0.45;
const SECONDARY_DIFF_SIZE_MIN_CENTER_DISTANCE = 6;
const DEFAULT_MAX_WATERMARK_REGIONS = 4;

const SAME_ANCHOR_SECOND_LAYER_MIN_SPATIAL = 0.45;
const SAME_ANCHOR_SECOND_LAYER_MIN_GRADIENT = 0.22;
const SAME_ANCHOR_SECOND_LAYER_MIN_PROMINENCE = 0.15;
const SAME_ANCHOR_MIN_SPATIAL_IMPROVEMENT = 0.12;

const ADAPTIVE_SECONDARY_THRESHOLD = 0.4;
const ADAPTIVE_SECONDARY_MIN_SPATIAL = 0.42;
const ADAPTIVE_SECONDARY_MIN_GRADIENT = 0.11;
const ADAPTIVE_SECONDARY_MIN_PROMINENCE = 0.12;

const EXTRA_MULTI_LOGO_CANDIDATE_CONFIGS = Object.freeze([
    Object.freeze({ logoSize: 48, marginRight: 32, marginBottom: 32 }),
    Object.freeze({ logoSize: 48, marginRight: 72, marginBottom: 72 }),
    Object.freeze({ logoSize: 48, marginRight: 73, marginBottom: 73 }),
    Object.freeze({ logoSize: 48, marginRight: 96, marginBottom: 96 }),
    Object.freeze({ logoSize: 72, marginRight: 108, marginBottom: 108 }),
    Object.freeze({ logoSize: 72, marginRight: 144, marginBottom: 144 }),
    Object.freeze({ logoSize: 96, marginRight: 64, marginBottom: 64 }),
    Object.freeze({ logoSize: 96, marginRight: 128, marginBottom: 128 }),
    Object.freeze({ logoSize: 96, marginRight: 192, marginBottom: 192 }),
    Object.freeze({ logoSize: 96, marginRight: 192, marginBottom: 192, alphaVariant: '20260520' }),
    Object.freeze({ logoSize: 36, marginRight: 96, marginBottom: 96, alphaVariant: 'v2' }),
    Object.freeze({ logoSize: 44, marginRight: 29, marginBottom: 40 })
]);

function isValidPosition(position, imageData) {
    return position &&
        [position.x, position.y, position.width, position.height]
            .every((value) => Number.isFinite(value)) &&
        position.width > 0 && position.height > 0 &&
        position.x >= 0 && position.y >= 0 &&
        position.x + position.width <= imageData.width &&
        position.y + position.height <= imageData.height;
}

function intersectionArea(left, right) {
    const width = Math.max(0, Math.min(left.x + left.width, right.x + right.width) -
        Math.max(left.x, right.x));
    const height = Math.max(0, Math.min(left.y + left.height, right.y + right.height) -
        Math.max(left.y, right.y));
    return width * height;
}

function overlapFraction(left, right) {
    const area = intersectionArea(left, right);
    const minArea = Math.min(left.width * left.height, right.width * right.height);
    return minArea > 0 ? area / minArea : 0;
}

function intersectionOverUnion(left, right) {
    const inter = intersectionArea(left, right);
    if (inter <= 0) return 0;
    const union = left.width * left.height + right.width * right.height - inter;
    return union > 0 ? inter / union : 0;
}

function centerDistance(left, right) {
    const dx = (left.x + left.width / 2) - (right.x + right.width / 2);
    const dy = (left.y + left.height / 2) - (right.y + right.height / 2);
    return Math.hypot(dx, dy);
}

function isDistinctFromSingleRegion(position, existing) {
    const minSide = Math.min(position.width, existing.width);
    const maxSide = Math.max(position.width, existing.width);
    const sizeRatio = maxSide > 0 ? minSide / maxSide : 1;

    if (sizeRatio < SECONDARY_DIFF_SIZE_RATIO_THRESHOLD) {
        return intersectionOverUnion(position, existing) <= SECONDARY_DIFF_SIZE_MAX_IOU &&
            centerDistance(position, existing) >= SECONDARY_DIFF_SIZE_MIN_CENTER_DISTANCE;
    }

    return overlapFraction(position, existing) <= SECONDARY_MAX_OVERLAP_FRACTION;
}

function isDistinctFromAllRegions(position, existingPositions) {
    return existingPositions.every((existing) => isDistinctFromSingleRegion(position, existing));
}

function resolveCatalogAlphaMap(config, options) {
    if (config.alphaVariant === 'v2' && config.logoSize === 36) {
        return options.getAlphaMap?.('36-v2') ?? null;
    }
    if (config.alphaVariant && config.logoSize === 96) {
        return options.alpha96Variants?.[config.alphaVariant] ?? null;
    }
    if (config.logoSize === 48) return options.alpha48;
    if (config.logoSize === 96) return options.alpha96;
    return options.getAlphaMap?.(config.logoSize) ?? null;
}

function localProminence(imageData, alphaMap, position, spatialScore) {
    let maximumControl = -Infinity;
    for (const [dx, dy] of [[-1, 0], [0, -1], [-1, -1], [-2, 0], [0, -2]]) {
        const control = {
            ...position,
            x: position.x + dx * position.width,
            y: position.y + dy * position.height
        };
        if (!isValidPosition(control, imageData)) continue;
        maximumControl = Math.max(
            maximumControl,
            computeRegionSpatialCorrelation({
                imageData,
                alphaMap,
                region: { x: control.x, y: control.y, size: control.width }
            })
        );
    }
    return Number.isFinite(maximumControl)
        ? spatialScore - maximumControl
        : Infinity;
}

function maskRegionsWithBorderMean(imageData, regions) {
    const masked = cloneImageData(imageData);
    const { width, height, data } = masked;

    for (const region of regions) {
        if (!isValidPosition(region, masked)) continue;
        let rSum = 0;
        let gSum = 0;
        let bSum = 0;
        let count = 0;
        const x0 = Math.max(0, region.x - 4);
        const y0 = Math.max(0, region.y - 4);
        const x1 = Math.min(width - 1, region.x + region.width + 3);
        const y1 = Math.min(height - 1, region.y + region.height + 3);

        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) {
                const inside = x >= region.x && x < region.x + region.width &&
                    y >= region.y && y < region.y + region.height;
                if (inside) continue;
                const idx = (y * width + x) * 4;
                rSum += data[idx];
                gSum += data[idx + 1];
                bSum += data[idx + 2];
                count++;
            }
        }

        const rMean = count > 0 ? Math.round(rSum / count) : 128;
        const gMean = count > 0 ? Math.round(gSum / count) : 128;
        const bMean = count > 0 ? Math.round(bSum / count) : 128;

        for (let y = region.y; y < region.y + region.height; y++) {
            for (let x = region.x; x < region.x + region.width; x++) {
                const idx = (y * width + x) * 4;
                data[idx] = rMean;
                data[idx + 1] = gMean;
                data[idx + 2] = bMean;
            }
        }
    }

    return masked;
}

function buildCandidateEntries(imageData, defaultConfig) {
    const entries = resolveGeminiWatermarkSearchCatalogEntries(
        imageData.width,
        imageData.height,
        defaultConfig
    ).map((entry) => ({ config: entry.config }));

    const officialSize = matchOfficialGeminiImageSize(
        imageData.width,
        imageData.height
    );
    if (['1k', '2k', '4k'].includes(officialSize?.resolutionTier)) {
        entries.push({
            config: {
                logoSize: 96,
                marginRight: 192,
                marginBottom: 192
            }
        });
        entries.push({
            config: {
                logoSize: 96,
                marginRight: 192,
                marginBottom: 192,
                alphaVariant: '20260520'
            }
        });
    }

    for (const extra of EXTRA_MULTI_LOGO_CANDIDATE_CONFIGS) {
        entries.push({ config: { ...extra } });
    }

    const seen = new Set();
    const deduped = [];
    for (const entry of entries) {
        const cfg = entry?.config;
        if (!cfg) continue;
        const key = `${cfg.logoSize}:${cfg.marginRight}:${cfg.marginBottom}:${cfg.alphaVariant ?? 'default'}`;
        if (seen.has(key)) continue;
        seen.add(key);
        deduped.push(entry);
    }
    return deduped;
}

function findAdaptiveSecondaryMatch(imageData, existingPositions, defaultConfig, options) {
    const adaptiveMode = options?.adaptiveMode || 'auto';
    if (adaptiveMode === 'never' || adaptiveMode === 'off' || !options?.alpha96) {
        return null;
    }

    const masked = maskRegionsWithBorderMean(imageData, existingPositions);
    const searchConfigs = [defaultConfig];
    if (defaultConfig.logoSize > 56) {
        searchConfigs.push({
            logoSize: 48,
            marginRight: 48,
            marginBottom: 48
        });
    }

    let bestMatch = null;
    for (const searchConfig of searchConfigs) {
        const adaptive = detectAdaptiveWatermarkRegion({
            imageData: masked,
            alpha96: options.alpha96,
            defaultConfig: searchConfig,
            threshold: ADAPTIVE_SECONDARY_THRESHOLD
        });
        if (!adaptive?.found || !adaptive.region) continue;

        const baseSize = adaptive.region.size;
        const centerX = adaptive.region.x + baseSize / 2;
        const centerY = adaptive.region.y + baseSize / 2;
        const candidateSizes = new Set([
            baseSize - 4,
            baseSize - 2,
            baseSize - 1,
            baseSize,
            baseSize + 1,
            baseSize + 2,
            baseSize + 4
        ]);
        for (const canonicalSize of [36, 48, 72, 96]) {
            if (Math.abs(canonicalSize - baseSize) <= 6) {
                candidateSizes.add(canonicalSize);
            }
        }

        let bestSize = baseSize;
        let bestX = adaptive.region.x;
        let bestY = adaptive.region.y;
        let bestFineScore = -Infinity;
        for (const probeSize of candidateSizes) {
            if (probeSize < 24) continue;
            let probeAlpha = null;
            try {
                probeAlpha = resolveCatalogAlphaMap({ logoSize: probeSize }, options);
            } catch {
                continue;
            }
            if (!probeAlpha || probeAlpha.length !== probeSize * probeSize) continue;
            const anchorX = Math.round(centerX - probeSize / 2);
            const anchorY = Math.round(centerY - probeSize / 2);
            const canonicalBonus = [36, 48, 72, 96].includes(probeSize) ? 0.025 : 0;
            for (const dy of [-2, -1, 0, 1, 2]) {
                for (const dx of [-2, -1, 0, 1, 2]) {
                    const probePos = {
                        x: anchorX + dx,
                        y: anchorY + dy,
                        width: probeSize,
                        height: probeSize
                    };
                    if (!isValidPosition(probePos, imageData)) continue;
                    const probeScored = scoreRegion(imageData, probeAlpha, probePos);
                    const total =
                        (probeScored.spatialScore || 0) * 0.7 +
                        (probeScored.gradientScore || 0) * 0.3 +
                        canonicalBonus;
                    if (total > bestFineScore) {
                        bestFineScore = total;
                        bestSize = probeSize;
                        bestX = probePos.x;
                        bestY = probePos.y;
                    }
                }
            }
        }

        const size = bestSize;
        const position = {
            x: bestX,
            y: bestY,
            width: size,
            height: size
        };
        if (!isValidPosition(position, imageData) ||
            !isDistinctFromAllRegions(position, existingPositions)) {
            continue;
        }

        const config = {
            logoSize: size,
            marginRight: imageData.width - position.x - size,
            marginBottom: imageData.height - position.y - size,
            fixedVariant: true
        };
        let alphaMap = null;
        try {
            alphaMap = resolveCatalogAlphaMap(config, options);
        } catch {
            continue;
        }
        if (!alphaMap || alphaMap.length !== size * size) continue;

        const { spatialScore, gradientScore } = scoreRegion(imageData, alphaMap, position);
        if (!Number.isFinite(spatialScore) || !Number.isFinite(gradientScore)) continue;
        const prominence = localProminence(imageData, alphaMap, position, spatialScore);
        if (
            spatialScore < ADAPTIVE_SECONDARY_MIN_SPATIAL ||
            gradientScore < ADAPTIVE_SECONDARY_MIN_GRADIENT ||
            prominence < ADAPTIVE_SECONDARY_MIN_PROMINENCE
        ) {
            continue;
        }

        const candidate = {
            position,
            config,
            spatialScore,
            gradientScore,
            adaptiveFound: true
        };
        if (
            !bestMatch ||
            candidate.spatialScore + candidate.gradientScore >
                bestMatch.spatialScore + bestMatch.gradientScore
        ) {
            bestMatch = candidate;
        }
    }

    return bestMatch;
}

function findSecondaryCatalogMatches(imageData, existingPositions, options) {
    const defaultConfig = detectWatermarkConfig(imageData.width, imageData.height);
    const entries = buildCandidateEntries(imageData, defaultConfig);
    const matches = [];

    for (const { config } of entries) {
        const size = config.logoSize;
        const position = {
            x: imageData.width - config.marginRight - size,
            y: imageData.height - config.marginBottom - size,
            width: size,
            height: size
        };
        if (!isValidPosition(position, imageData) ||
            !isDistinctFromAllRegions(position, existingPositions)) continue;

        let alphaMap;
        try {
            alphaMap = resolveCatalogAlphaMap(config, options);
        } catch {
            continue;
        }
        if (!alphaMap || alphaMap.length !== size * size) continue;
        const { spatialScore, gradientScore } = scoreRegion(
            imageData,
            alphaMap,
            position
        );
        const prominence = spatialScore >= SECONDARY_MIN_SPATIAL_SCORE
            ? localProminence(imageData, alphaMap, position, spatialScore)
            : -Infinity;
        const directWitness = spatialScore >= SECONDARY_MIN_SPATIAL_SCORE &&
            gradientScore >= SECONDARY_MIN_GRADIENT_SCORE &&
            prominence >= SECONDARY_MIN_LOCAL_PROMINENCE;
        const strongSpatialWitness = spatialScore >= SECONDARY_STRONG_SPATIAL_SCORE &&
            prominence >= SECONDARY_STRONG_LOCAL_PROMINENCE;
        if (!Number.isFinite(spatialScore) ||
            !Number.isFinite(gradientScore) ||
            !(directWitness || strongSpatialWitness)) continue;

        matches.push({ position, config, spatialScore, gradientScore });
    }

    const adaptiveMatch = findAdaptiveSecondaryMatch(
        imageData,
        existingPositions,
        defaultConfig,
        options
    );
    if (adaptiveMatch && !matchesCatalogRegion(adaptiveMatch.position, matches)) {
        matches.push(adaptiveMatch);
    }

    return matches.sort((left, right) =>
        right.spatialScore + right.gradientScore -
        left.spatialScore - left.gradientScore
    );
}

function matchesCatalogRegion(position, matches) {
    return matches.some((match) => {
        const expected = match.position;
        const sizeTolerance = Math.max(2, expected.width * 0.15);
        const centerTolerance = Math.max(4, expected.width * 0.25);
        return Math.abs(position.width - expected.width) <= sizeTolerance &&
            Math.abs(position.height - expected.height) <= sizeTolerance &&
            Math.abs(position.x + position.width / 2 -
                expected.x - expected.width / 2) <= centerTolerance &&
            Math.abs(position.y + position.height / 2 -
                expected.y - expected.height / 2) <= centerTolerance;
    });
}

function combineQualityStatus(first, second) {
    if (first === second) return first;
    if (first === 'clean') return second;
    if (second === 'clean') return first;
    return 'mixed';
}

function summarizeRegion(meta) {
    return {
        position: meta.position,
        config: meta.config,
        source: meta.source,
        qualityStatus: meta.qualityStatus,
        detection: meta.detection
    };
}

function maybeRemoveStackedSameAnchorLayer(currentResult, options, processOne) {
    const position = currentResult?.meta?.position;
    const config = currentResult?.meta?.config ?? (position ? {
        logoSize: position.width,
        marginRight: currentResult.imageData.width - position.x - position.width,
        marginBottom: currentResult.imageData.height - position.y - position.height
    } : null);
    if (!config || !isValidPosition(position, currentResult?.imageData)) {
        return { result: currentResult, extraPasses: 0 };
    }

    let alphaMap = null;
    try {
        alphaMap = resolveCatalogAlphaMap(config, options);
    } catch {
        return { result: currentResult, extraPasses: 0 };
    }
    if (!alphaMap || alphaMap.length !== position.width * position.height) {
        return { result: currentResult, extraPasses: 0 };
    }

    const before = scoreRegion(currentResult.imageData, alphaMap, position);
    if (
        !Number.isFinite(before.spatialScore) ||
        !Number.isFinite(before.gradientScore) ||
        before.spatialScore < SAME_ANCHOR_SECOND_LAYER_MIN_SPATIAL ||
        before.gradientScore < SAME_ANCHOR_SECOND_LAYER_MIN_GRADIENT
    ) {
        return { result: currentResult, extraPasses: 0 };
    }

    const prominence = localProminence(
        currentResult.imageData,
        alphaMap,
        position,
        before.spatialScore
    );
    if (prominence < SAME_ANCHOR_SECOND_LAYER_MIN_PROMINENCE) {
        return { result: currentResult, extraPasses: 0 };
    }

    let secondLayerResult = null;
    try {
        secondLayerResult = processOne(currentResult.imageData, {
            secondaryCatalogConfig: config
        });
    } catch {
        return { result: currentResult, extraPasses: 0 };
    }

    const secondPosition = secondLayerResult?.meta?.position;
    if (
        secondLayerResult?.meta?.applied !== true ||
        !isValidPosition(secondPosition, currentResult.imageData) ||
        centerDistance(position, secondPosition) > Math.max(4, position.width * 0.2)
    ) {
        return { result: currentResult, extraPasses: 0 };
    }

    const after = scoreRegion(secondLayerResult.imageData, alphaMap, position);
    if (
        !Number.isFinite(after.spatialScore) ||
        Math.abs(before.spatialScore) - Math.abs(after.spatialScore) < SAME_ANCHOR_MIN_SPATIAL_IMPROVEMENT
    ) {
        return { result: currentResult, extraPasses: 0 };
    }

    return {
        result: {
            ...currentResult,
            imageData: secondLayerResult.imageData,
            meta: {
                ...currentResult.meta,
                source: `${currentResult.meta.source}+stacked-layer`,
                qualityStatus: secondLayerResult.meta.qualityStatus ?? currentResult.meta.qualityStatus,
                detection: secondLayerResult.meta.detection ?? currentResult.meta.detection,
                stackedLayerCount: 2
            }
        },
        extraPasses: 1
    };
}

function isValidSecondaryRestoration(candidateResult, match) {
    const detection = candidateResult?.meta?.detection;
    if (!detection) return true;

    const origGrad = Number(detection.originalGradientScore);
    const procGrad = Number(detection.processedGradientScore);
    const procSpatial = Number(detection.processedSpatialScore);
    if (
        Number.isFinite(origGrad) &&
        Number.isFinite(procGrad) &&
        procGrad > Math.max(0.22, origGrad + 0.04)
    ) {
        return false;
    }
    if (
        match?.adaptiveFound === true &&
        Number.isFinite(origGrad) &&
        origGrad < 0.2 &&
        Number.isFinite(procSpatial) &&
        procSpatial < -0.2
    ) {
        return false;
    }
    return true;
}

function tryProcessSecondaryCandidate(imageData, match, existingPositions, matches, options, processOne) {
    let candidateResult = null;
    try {
        candidateResult = processOne(imageData, {
            secondaryCatalogConfig: match.config,
            ...(match.adaptiveFound ? { adaptiveMode: 'never' } : {})
        });
    } catch {
        return { applied: false, reason: 'secondary-processing-failed', result: null };
    }

    const canRetryAdaptive = (
        candidateResult?.meta?.applied !== true &&
        options?.adaptiveMode !== 'never' &&
        options?.adaptiveMode !== 'off' &&
        match.spatialScore >= 0.5 &&
        match.gradientScore >= 0.22
    );
    if (canRetryAdaptive) {
        try {
            candidateResult = processOne(imageData, {
                secondaryCatalogConfig: match.config,
                adaptiveMode: 'always'
            });
        } catch {
            return { applied: false, reason: 'secondary-processing-failed', result: null };
        }
    }

    const candidatePosition = candidateResult?.meta?.position;
    if (
        candidateResult?.meta?.applied !== true ||
        !isValidPosition(candidatePosition, imageData) ||
        !isDistinctFromAllRegions(candidatePosition, existingPositions) ||
        !matchesCatalogRegion(candidatePosition, matches) ||
        !isValidSecondaryRestoration(candidateResult, match)
    ) {
        return {
            applied: false,
            reason: candidateResult?.meta?.skipReason ?? 'secondary-candidate-not-selected',
            result: candidateResult
        };
    }

    return {
        applied: true,
        reason: null,
        result: candidateResult
    };
}

export function processSecondaryCatalogWatermark({ firstResult, options, processOne }) {
    if (options.multiRegion === false ||
        firstResult?.meta?.applied !== true ||
        !isValidPosition(firstResult.meta.position, firstResult.imageData)) {
        return firstResult;
    }

    const maxRegions = Math.max(2, Math.round(options.maxRegions ?? DEFAULT_MAX_WATERMARK_REGIONS));
    const stackedFirst = maybeRemoveStackedSameAnchorLayer(firstResult, options, processOne);
    let workingResult = stackedFirst.result;
    const processedMetaList = [workingResult.meta];
    const processedPositions = [workingResult.meta.position];
    let secondaryProcessingMs = 0;
    let firstFailureMeta = null;

    while (processedPositions.length < maxRegions) {
        const matches = findSecondaryCatalogMatches(
            workingResult.imageData,
            processedPositions,
            options
        );
        if (matches.length === 0) break;

        let acceptedStep = null;
        for (const match of matches) {
            const attempt = tryProcessSecondaryCandidate(
                workingResult.imageData,
                match,
                processedPositions,
                matches,
                options,
                processOne
            );
            if (attempt.applied) {
                acceptedStep = attempt.result;
                break;
            }
            if (!firstFailureMeta) {
                firstFailureMeta = {
                    detected: true,
                    applied: false,
                    position: match.position,
                    reason: attempt.reason
                };
            }
        }

        if (!acceptedStep) break;

        const stackedNext = maybeRemoveStackedSameAnchorLayer(acceptedStep, options, processOne);
        const completedStep = stackedNext.result;
        workingResult = {
            ...workingResult,
            imageData: completedStep.imageData
        };
        processedMetaList.push(completedStep.meta);
        processedPositions.push(completedStep.meta.position);
        if (completedStep.debugTimings?.totalMs != null) {
            secondaryProcessingMs += completedStep.debugTimings.totalMs;
        }
    }

    if (processedMetaList.length === 1) {
        if (firstFailureMeta && stackedFirst.extraPasses === 0) {
            return {
                ...workingResult,
                meta: {
                    ...workingResult.meta,
                    secondaryWatermark: firstFailureMeta
                }
            };
        }
        return workingResult;
    }

    let combinedQuality = processedMetaList[0].qualityStatus;
    let anyBestEffort = processedMetaList[0].bestEffort === true;
    for (let i = 1; i < processedMetaList.length; i++) {
        combinedQuality = combineQualityStatus(combinedQuality, processedMetaList[i].qualityStatus);
        anyBestEffort = anyBestEffort || processedMetaList[i].bestEffort === true;
    }

    return {
        ...firstResult,
        imageData: workingResult.imageData,
        meta: {
            ...workingResult.meta,
            position: processedMetaList[0].position,
            config: processedMetaList[0].config,
            detection: processedMetaList[0].detection,
            source: `${firstResult.meta.source}+multi-region`,
            qualityStatus: combinedQuality,
            bestEffort: anyBestEffort,
            processedRegionCount: processedMetaList.length,
            watermarkRegions: processedMetaList.map(summarizeRegion)
        },
        debugTimings: firstResult.debugTimings
            ? {
                ...firstResult.debugTimings,
                secondaryProcessingMs,
                totalMs: (firstResult.debugTimings.totalMs ?? 0) + secondaryProcessingMs
            }
            : firstResult.debugTimings
    };
}
