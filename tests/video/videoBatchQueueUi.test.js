import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('video-preview.html contains batch upload input and queue panel structure', () => {
    const html = readFileSync(new URL('../../public/video-preview.html', import.meta.url), 'utf8');

    // file input should support multiple files
    assert.match(html, /<input id="fileInput"[^>]*multiple/);

    // queue panel and header controls
    assert.match(html, /id="queuePanel"/);
    assert.match(html, /id="queueBadge"/);
    assert.match(html, /id="processAllBtn"/);
    assert.match(html, /id="cancelBatchBtn"/);
    assert.match(html, /id="downloadAllBtn"/);
    assert.match(html, /id="clearQueueBtn"/);

    // batch progress and queue container
    assert.match(html, /id="batchProgressWrapper"/);
    assert.match(html, /id="batchProgressBar"/);
    assert.match(html, /id="videoQueueList"/);
});

test('src/video-app.js exposes batch queue methods and listeners', () => {
    const source = readFileSync(new URL('../../src/video-app.js', import.meta.url), 'utf8');

    assert.match(source, /function addVideosToQueue\(/);
    assert.match(source, /function processAllQueue\(/);
    assert.match(source, /function cancelBatch\(/);
    assert.match(source, /function downloadAllProcessed\(/);
    assert.match(source, /function renderQueueList\(/);
    assert.match(source, /function updateBatchControls\(/);
});
