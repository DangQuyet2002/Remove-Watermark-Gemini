import test from 'node:test';
import assert from 'node:assert/strict';

import { WatermarkEngine } from '../../src/core/watermarkEngine.js';
import { applyImageThoroughCleanup } from '../../src/core/imageAiDenoiseCleanup.js';
import { computeRegionSpatialCorrelation, interpolateAlphaMap } from '../../src/core/adaptiveDetector.js';
import { createPatternImageData, applySyntheticWatermark } from './syntheticWatermarkTestUtils.js';

function createMockCanvas2DContext(imageData) {
    let current = {
        width: imageData.width,
        height: imageData.height,
        data: new Uint8ClampedArray(imageData.data)
    };
    return {
        canvas: { width: imageData.width, height: imageData.height },
        getImageData(x, y, w, h) {
            const out = new Uint8ClampedArray(w * h * 4);
            for (let row = 0; row < h; row++) {
                const srcStart = ((y + row) * current.width + x) * 4;
                out.set(current.data.subarray(srcStart, srcStart + w * 4), row * w * 4);
            }
            return { width: w, height: h, data: out };
        },
        putImageData(patch, x, y) {
            for (let row = 0; row < patch.height; row++) {
                const srcStart = row * patch.width * 4;
                const dstStart = ((y + row) * current.width + x) * 4;
                current.data.set(patch.data.subarray(srcStart, srcStart + patch.width * 4), dstStart);
            }
        },
        readFull() {
            return current;
        }
    };
}

test('applyImageThoroughCleanup rescues 2K portrait V2 48px watermark when initial pass skipped', async () => {
    const engine = await WatermarkEngine.create();
    const alpha36v2 = await engine.getAlphaMap('36-v2');
    const alpha48v2 = interpolateAlphaMap(alpha36v2, 36, 48);
    const width = 1536;
    const height = 2752;
    const imageData = createPatternImageData(width, height);
    for (let i = 0; i < imageData.data.length; i += 4) {
        imageData.data[i] = Math.round(imageData.data[i] * 0.25 + 80);
        imageData.data[i + 1] = Math.round(imageData.data[i + 1] * 0.25 + 85);
        imageData.data[i + 2] = Math.round(imageData.data[i + 2] * 0.25 + 90);
    }
    const position = {
        x: width - 73 - 48,
        y: height - 73 - 48,
        width: 48,
        height: 48
    };
    applySyntheticWatermark(imageData, alpha48v2, position);

    const beforeSpatial = computeRegionSpatialCorrelation({
        imageData,
        alphaMap: alpha48v2,
        region: { x: position.x, y: position.y, size: 48 }
    });
    assert.ok(beforeSpatial > 0.4, `expected positive watermark correlation, got ${beforeSpatial}`);

    const ctx = createMockCanvas2DContext(imageData);
    const updatedMeta = await applyImageThoroughCleanup({
        ctx,
        canvasWidth: width,
        canvasHeight: height,
        engine,
        meta: { applied: false, source: 'skipped', skipReason: 'no-watermark-detected' },
        options: { thoroughMode: true }
    });

    assert.equal(updatedMeta.applied, true);
    assert.equal(updatedMeta.decisionTier, 'rescued-thorough');
    const afterSpatial = computeRegionSpatialCorrelation({
        imageData: ctx.readFull(),
        alphaMap: alpha48v2,
        region: { x: position.x, y: position.y, size: 48 }
    });
    assert.ok(Math.abs(afterSpatial) < 0.2, `expected residual < 0.2, got ${afterSpatial}`);
});

test('applyImageThoroughCleanup leaves clean 2K portrait image untouched when skipped', async () => {
    const engine = await WatermarkEngine.create();
    const width = 1536;
    const height = 2752;
    const imageData = createPatternImageData(width, height);
    const ctx = createMockCanvas2DContext(imageData);

    const updatedMeta = await applyImageThoroughCleanup({
        ctx,
        canvasWidth: width,
        canvasHeight: height,
        engine,
        meta: { applied: false, source: 'skipped', skipReason: 'no-watermark-detected' },
        options: { thoroughMode: true }
    });

    assert.equal(updatedMeta.applied, false);
    assert.deepEqual(ctx.readFull().data, imageData.data);
});
