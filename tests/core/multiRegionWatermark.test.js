import test from 'node:test';
import assert from 'node:assert/strict';

import { computeRegionSpatialCorrelation, interpolateAlphaMap } from '../../src/core/adaptiveDetector.js';
import { getEmbeddedAlphaMap } from '../../src/core/embeddedAlphaMaps.js';
import { processWatermarkImageData } from '../../src/core/watermarkProcessor.js';
import { decodeImageDataInNode } from '../../scripts/sample-benchmark.js';
import {
    applySyntheticWatermark,
    cloneTestImageData,
    createPatternImageData,
    createSyntheticAlphaMap
} from './syntheticWatermarkTestUtils.js';

const canonicalPosition = { x: 944, y: 944, width: 48, height: 48 };
const largeMarginPosition = { x: 880, y: 880, width: 48, height: 48 };

function spatialScore(imageData, alphaMap, position) {
    return computeRegionSpatialCorrelation({
        imageData,
        alphaMap,
        region: { x: position.x, y: position.y, size: position.width }
    });
}

test('default image processing clears two independent Gemini catalog positions', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const imageData = createPatternImageData(1024, 1024);
    applySyntheticWatermark(imageData, alpha48, canonicalPosition);
    applySyntheticWatermark(imageData, alpha48, largeMarginPosition);

    const options = {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => interpolateAlphaMap(alpha96, 96, size)
    };
    const singleRegion = processWatermarkImageData(imageData, {
        ...options,
        multiRegion: false
    });
    const result = processWatermarkImageData(imageData, options);

    assert.equal(singleRegion.meta.applied, true);
    assert.equal(result.meta.processedRegionCount, 2);
    assert.deepEqual(
        result.meta.watermarkRegions.map((region) => region.position),
        [largeMarginPosition, canonicalPosition]
    );
    assert.ok(Math.abs(spatialScore(result.imageData, alpha48, canonicalPosition)) < 0.1);
    assert.ok(Math.abs(spatialScore(result.imageData, alpha48, largeMarginPosition)) < 0.1);
    assert.ok(
        Math.abs(spatialScore(singleRegion.imageData, alpha48, canonicalPosition)) > 0.7,
        'the one-region baseline should retain the second logo'
    );
});

test('one Gemini watermark keeps the one-region output', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const imageData = createPatternImageData(1024, 1024);
    applySyntheticWatermark(imageData, alpha48, canonicalPosition);
    const options = {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => interpolateAlphaMap(alpha96, 96, size)
    };

    const result = processWatermarkImageData(imageData, options);
    const oneRegion = processWatermarkImageData(imageData, {
        ...options,
        multiRegion: false
    });

    assert.equal(result.meta.applied, true);
    assert.equal(result.meta.processedRegionCount, undefined);
    assert.deepEqual(result.imageData.data, oneRegion.imageData.data);
});

test('a retained 96px 192px-margin logo is processed after the current logo', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const currentPosition = { x: 864, y: 864, width: 96, height: 96 };
    const retainedPosition = { x: 736, y: 736, width: 96, height: 96 };
    const imageData = createPatternImageData(1024, 1024);
    applySyntheticWatermark(imageData, alpha96, currentPosition);
    applySyntheticWatermark(imageData, alpha96, retainedPosition);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.processedRegionCount, 2);
    assert.deepEqual(
        result.meta.watermarkRegions.map((region) => region.position),
        [currentPosition, retainedPosition]
    );
    assert.ok(Math.abs(spatialScore(result.imageData, alpha96, currentPosition)) < 0.1);
    assert.ok(Math.abs(spatialScore(result.imageData, alpha96, retainedPosition)) < 0.1);
});

test('a new 48px logo can be removed over the corner of a retained 96px logo', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const retainedPosition = { x: 864, y: 864, width: 96, height: 96 };
    const imageData = createPatternImageData(1024, 1024);
    applySyntheticWatermark(imageData, alpha96, retainedPosition);
    applySyntheticWatermark(imageData, alpha48, canonicalPosition);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.processedRegionCount, 2);
    assert.deepEqual(
        result.meta.watermarkRegions.map((region) => region.position),
        [canonicalPosition, retainedPosition]
    );
    assert.ok(Math.abs(spatialScore(result.imageData, alpha48, canonicalPosition)) < 0.1);
    assert.ok(Math.abs(spatialScore(result.imageData, alpha96, retainedPosition)) < 0.1);
});

test('a second catalog watermark added to a real Gemini fixture is processed', async () => {
    const alpha48 = getEmbeddedAlphaMap(48);
    const alpha96 = getEmbeddedAlphaMap(96);
    const imageData = cloneTestImageData(await decodeImageDataInNode(
        'tests/fixtures/gemini-1024-large-margin.png'
    ));
    applySyntheticWatermark(imageData, alpha48, canonicalPosition);
    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) =>
            getEmbeddedAlphaMap(size) || interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.processedRegionCount, 2);
    assert.deepEqual(
        new Set(result.meta.watermarkRegions.map((region) =>
            `${region.position.x},${region.position.y}`
        )),
        new Set(['944,944', '880,880'])
    );
    assert.ok(spatialScore(imageData, alpha48, canonicalPosition) > 0.9);
    assert.ok(spatialScore(imageData, alpha48, largeMarginPosition) > 0.9);
    assert.ok(spatialScore(result.imageData, alpha48, canonicalPosition) < 0);
    assert.ok(spatialScore(result.imageData, alpha48, largeMarginPosition) < 0);
});

test('adaptive multi-region removal clears a shifted non-catalog second Gemini logo', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const shiftedSecondPosition = { x: 920, y: 900, width: 48, height: 48 };
    const imageData = createPatternImageData(1024, 1024);
    applySyntheticWatermark(imageData, alpha48, canonicalPosition);
    applySyntheticWatermark(imageData, alpha48, shiftedSecondPosition);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'always',
        getAlphaMap: (size) => interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.processedRegionCount, 2);
    assert.ok(Math.abs(spatialScore(result.imageData, alpha48, canonicalPosition)) < 0.15);
    assert.ok(Math.abs(spatialScore(result.imageData, alpha48, shiftedSecondPosition)) < 0.15);
});

test('multi-region removal clears three independent Gemini logos in one image', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const retained96 = { x: 736, y: 736, width: 96, height: 96 };
    const imageData = createPatternImageData(1024, 1024);
    applySyntheticWatermark(imageData, alpha48, canonicalPosition);
    applySyntheticWatermark(imageData, alpha48, largeMarginPosition);
    applySyntheticWatermark(imageData, alpha96, retained96);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.processedRegionCount, 3);
    assert.ok(Math.abs(spatialScore(result.imageData, alpha48, canonicalPosition)) < 0.12);
    assert.ok(Math.abs(spatialScore(result.imageData, alpha48, largeMarginPosition)) < 0.12);
    assert.ok(Math.abs(spatialScore(result.imageData, alpha96, retained96)) < 0.12);
});

test('stacked double-watermarked logo at the exact same anchor is cleared completely', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const imageData = createPatternImageData(1024, 1024);
    // Stamp two Gemini logos on top of each other at the exact same position
    applySyntheticWatermark(imageData, alpha48, canonicalPosition);
    applySyntheticWatermark(imageData, alpha48, canonicalPosition);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.applied, true);
    assert.ok(Math.abs(spatialScore(result.imageData, alpha48, canonicalPosition)) < 0.15);
});
