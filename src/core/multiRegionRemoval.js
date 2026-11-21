import { computeRegionSpatialCorrelation } from './adaptiveDetector.js';
import { scoreRegion } from './restorationMetrics.js';
import { detectWatermarkConfig } from './watermarkConfig.js';
import {
    matchOfficialGeminiImageSize,
    resolveGeminiWatermarkSearchCatalogEntries
} from './geminiSizeCatalog.js';

const SECONDARY_MIN_SPATIAL_SCORE = 0.3;
const SECONDARY_MIN_GRADIENT_SCORE = 0.12;
const SECONDARY_MIN_LOCAL_PROMINENCE = 0.08;
const SECONDARY_STRONG_SPATIAL_SCORE = 0.8;
const SECONDARY_STRONG_LOCAL_PROMINENCE = 0.2;
const SECONDARY_MAX_OVERLAP_FRACTION = 0.2;

function isValidPosition(position, imageData) {
    return position &&
        [position.x, position.y, position.width, position.height]
            .every((value) => Number.isFinite(value)) &&
        position.width > 0 && position.height > 0 &&
        position.x >= 0 && position.y >= 0 &&
        position.x + position.width <= imageData.width &&
        position.y + position.height <= imageData.height;
}

function overlapFraction(left, right) {
    const width = Math.max(0, Math.min(left.x + left.width, right.x + right.width) -
        Math.max(left.x, right.x));
    const height = Math.max(0, Math.min(left.y + left.height, right.y + right.height) -
        Math.max(left.y, right.y));
    return width * height / Math.min(
        left.width * left.height,
        right.width * right.height
    );
}

function isDistinctRegion(position, firstPosition) {
    return overlapFraction(position, firstPosition) <= SECONDARY_MAX_OVERLAP_FRACTION;
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

function findSecondaryCatalogMatches(imageData, firstPosition, options) {
    const defaultConfig = detectWatermarkConfig(imageData.width, imageData.height);
    const entries = resolveGeminiWatermarkSearchCatalogEntries(
        imageData.width,
        imageData.height,
        defaultConfig
    );
    const officialSize = matchOfficialGeminiImageSize(
        imageData.width,
        imageData.height
    );
    if (['1k', '2k', '4k'].includes(officialSize?.resolutionTier)) {
        // An older edit can retain a 96px logo at this anchor even when the
        // current output catalog prefers a different size or alpha profile.
        entries.push({ config: {
            logoSize: 96,
            marginRight: 192,
            marginBottom: 192
        } });
        entries.push({ config: {
            logoSize: 96,
            marginRight: 192,
            marginBottom: 192,
            alphaVariant: '20260520'
        } });
    }
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
            !isDistinctRegion(position, firstPosition)) continue;

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

export function processSecondaryCatalogWatermark({ firstResult, options, processOne }) {
    if (options.multiRegion === false ||
        firstResult?.meta?.applied !== true ||
        !isValidPosition(firstResult.meta.position, firstResult.imageData)) {
        return firstResult;
    }

    const matches = findSecondaryCatalogMatches(
        firstResult.imageData,
        firstResult.meta.position,
        options
    );
    if (matches.length === 0) return firstResult;

    let secondResult;
    try {
        secondResult = processOne(firstResult.imageData, {
            secondaryCatalogConfig: matches[0].config
        });
    } catch {
        return {
            ...firstResult,
            meta: {
                ...firstResult.meta,
                secondaryWatermark: {
                    detected: true,
                    applied: false,
                    position: matches[0].position,
                    reason: 'secondary-processing-failed'
                }
            }
        };
    }
    const secondPosition = secondResult?.meta?.position;
    if (secondResult?.meta?.applied !== true ||
        !isValidPosition(secondPosition, firstResult.imageData) ||
        !isDistinctRegion(secondPosition, firstResult.meta.position) ||
        !matchesCatalogRegion(secondPosition, matches)) {
        return {
            ...firstResult,
            meta: {
                ...firstResult.meta,
                secondaryWatermark: {
                    detected: true,
                    applied: false,
                    position: matches[0].position,
                    reason: secondResult?.meta?.skipReason ?? 'secondary-candidate-not-selected'
                }
            }
        };
    }

    return {
        ...firstResult,
        imageData: secondResult.imageData,
        meta: {
            ...firstResult.meta,
            source: `${firstResult.meta.source}+multi-region`,
            qualityStatus: combineQualityStatus(
                firstResult.meta.qualityStatus,
                secondResult.meta.qualityStatus
            ),
            bestEffort: firstResult.meta.bestEffort === true ||
                secondResult.meta.bestEffort === true,
            processedRegionCount: 2,
            watermarkRegions: [
                summarizeRegion(firstResult.meta),
                summarizeRegion(secondResult.meta)
            ]
        },
        debugTimings: firstResult.debugTimings && secondResult.debugTimings
            ? {
                ...firstResult.debugTimings,
                secondaryProcessingMs: secondResult.debugTimings.totalMs ?? null,
                totalMs: (firstResult.debugTimings.totalMs ?? 0) +
                    (secondResult.debugTimings.totalMs ?? 0)
            }
            : firstResult.debugTimings
    };
}
