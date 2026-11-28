export function canUseWatermarkWorker(env = globalThis) {
    return typeof env.Worker !== 'undefined' && typeof env.Blob !== 'undefined';
}

function normalizeError(errorLike) {
    if (!errorLike) return 'Unknown worker error';
    if (typeof errorLike === 'string') return errorLike;
    if (typeof errorLike.message === 'string' && errorLike.message.length > 0) {
        return errorLike.message;
    }
    return 'Unknown worker error';
}

function toError(errorLike) {
    if (errorLike instanceof Error) return errorLike;
    return new Error(normalizeError(errorLike));
}

export class WatermarkWorkerClient {
    constructor({
        workerUrl = './workers/watermark-worker.js',
        WorkerClass = globalThis.Worker
    } = {}) {
        if (typeof WorkerClass === 'undefined') {
            throw new Error('Worker is not supported in this runtime');
        }

        this.worker = new WorkerClass(workerUrl, { type: 'module' });
        this.pending = new Map();
        this.queue = [];
        this.activeRequestId = null;
        this.requestId = 0;

        this.onMessage = this.onMessage.bind(this);
        this.onError = this.onError.bind(this);
        this.worker.addEventListener('message', this.onMessage);
        this.worker.addEventListener('error', this.onError);
    }

    get load() {
        return (this.activeRequestId !== null ? 1 : 0) + this.queue.length;
    }

    dispose() {
        this.worker.removeEventListener('message', this.onMessage);
        this.worker.removeEventListener('error', this.onError);
        this.worker.terminate();
        this.rejectAllPending(new Error('Worker disposed'));
    }

    onMessage(event) {
        const payload = event?.data;
        if (!payload || typeof payload.id === 'undefined') return;
        const pending = this.pending.get(payload.id);
        if (!pending) return;

        this.pending.delete(payload.id);
        clearTimeout(pending.timeoutId);
        if (this.activeRequestId === payload.id) {
            this.activeRequestId = null;
            this.flushQueue();
        }
        if (payload.ok) {
            pending.resolve(payload.result);
            return;
        }

        pending.reject(new Error(normalizeError(payload.error)));
    }

    onError(event) {
        const reason = event?.message || 'Worker execution failed';
        this.rejectAllPending(new Error(reason));
    }

    rejectAllPending(error) {
        for (const pending of this.pending.values()) {
            if (pending.timeoutId) clearTimeout(pending.timeoutId);
            pending.reject(error);
        }
        this.pending.clear();
        for (const queued of this.queue) {
            queued.reject(error);
        }
        this.queue = [];
        this.activeRequestId = null;
    }

    dispatchEntry(entry) {
        const { id, type, payload, transferList, timeoutMs, resolve, reject } = entry;
        const timeoutId = setTimeout(() => {
            this.pending.delete(id);
            if (this.activeRequestId === id) {
                this.activeRequestId = null;
                this.flushQueue();
            }
            reject(new Error(`Worker request timed out: ${type}`));
        }, timeoutMs);

        this.pending.set(id, { resolve, reject, timeoutId });
        if (type !== 'ping') {
            this.activeRequestId = id;
        }
        try {
            this.worker.postMessage({ id, type, ...payload }, transferList);
        } catch (error) {
            clearTimeout(timeoutId);
            this.pending.delete(id);
            if (this.activeRequestId === id) {
                this.activeRequestId = null;
                this.flushQueue();
            }
            reject(toError(error));
        }
    }

    flushQueue() {
        while (this.activeRequestId === null && this.queue.length > 0) {
            const next = this.queue.shift();
            this.dispatchEntry(next);
        }
    }

    request(type, payload, transferList = [], timeoutMs = 120000) {
        const id = ++this.requestId;
        return new Promise((resolve, reject) => {
            const entry = {
                id,
                type,
                payload,
                transferList,
                timeoutMs,
                resolve,
                reject
            };
            if (type === 'ping' || this.activeRequestId === null) {
                this.dispatchEntry(entry);
                return;
            }
            this.queue.push(entry);
        });
    }

    async ping(timeoutMs = 3000) {
        await this.request('ping', {}, [], timeoutMs);
    }

    async processBlob(blob, options = {}) {
        const inputBuffer = await blob.arrayBuffer();
        const result = await this.request(
            'process-image',
            {
                inputBuffer,
                mimeType: blob.type || 'image/png',
                options
            },
            [inputBuffer]
        );

        return {
            blob: new Blob([result.processedBuffer], { type: result.mimeType || 'image/png' }),
            meta: result.meta || null
        };
    }
}
