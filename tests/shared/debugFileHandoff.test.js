import test from 'node:test';
import assert from 'node:assert/strict';
import {
    getDebugFileKind,
    pickDebugUploadFile,
    pickDebugUploadFiles
} from '../../src/shared/debugFileHandoff.js';

test('getDebugFileKind should classify supported image and video files', () => {
    assert.equal(getDebugFileKind({ name: 'sample.png', type: 'image/png' }), 'image');
    assert.equal(getDebugFileKind({ name: 'sample.webp', type: '' }), 'image');
    assert.equal(getDebugFileKind({ name: 'clip.mp4', type: 'video/mp4' }), 'video');
    assert.equal(getDebugFileKind({ name: 'clip.mov', type: '' }), 'video');
    assert.equal(getDebugFileKind({ name: 'notes.txt', type: 'text/plain' }), null);
});

test('pickDebugUploadFile should prefer video when mixed media is dropped', () => {
    const image = { name: 'sample.png', type: 'image/png' };
    const video = { name: 'clip.mp4', type: 'video/mp4' };

    assert.equal(pickDebugUploadFile([image, video]), video);
    assert.equal(pickDebugUploadFile([image]), image);
    assert.equal(pickDebugUploadFile([{ name: 'notes.txt', type: 'text/plain' }]), null);
});

test('pickDebugUploadFiles should return all video files when multiple videos or mixed media is dropped', () => {
    const img1 = { name: 'img1.png', type: 'image/png' };
    const img2 = { name: 'img2.jpg', type: 'image/jpeg' };
    const vid1 = { name: 'vid1.mp4', type: 'video/mp4' };
    const vid2 = { name: 'vid2.webm', type: 'video/webm' };
    const doc = { name: 'doc.txt', type: 'text/plain' };

    assert.deepEqual(pickDebugUploadFiles([img1, vid1, img2, vid2, doc]), [vid1, vid2]);
    assert.deepEqual(pickDebugUploadFiles([img1, img2, doc]), [img1, img2]);
    assert.deepEqual(pickDebugUploadFiles([doc]), []);
});
