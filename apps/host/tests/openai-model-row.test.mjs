import test from 'node:test';
import assert from 'node:assert/strict';
import {
  nativeModelRow,
  runtimeModelRow,
  THINKING_EFFORTS,
  THINKING_WIRE,
  STRATA_CHAT_CAPS,
} from '../server/openai-model-row.mjs';
import { createLocalShare } from '../server/local-share.mjs';

const baseNative = {
  id: 'inst-1',
  apiModelId: 'qwen-iq3_s',
  gpus: [0],
  label: 'Qwen IQ3_S',
  aliases: ['qwen'],
};

function assertChatDiscovery(row, name) {
  assert.equal(row.name, name);
  assert.deepEqual(row.thinking, { supported: true, default: 'high', efforts: [...THINKING_EFFORTS] });
  assert.deepEqual(row.wire, THINKING_WIRE);
  for (const [key, value] of Object.entries(STRATA_CHAT_CAPS)) assert.equal(row[key], value);
  assert.equal(row.max_tokens, undefined);
  assert.equal(row.max_output_tokens, undefined);
  assert.equal(row.max_completion_tokens, undefined);
}

test('native /v1/models prefers metrics.engine.max_context and keeps existing meta keys', () => {
  const row = nativeModelRow({
    ...baseNative,
    context: 131072,
    metrics: { engine: { max_context: 524288 } },
    vision: false,
  });
  assert.equal(row.context_length, 524288);
  assert.equal(row.max_model_len, 524288);
  assert.equal(row.meta.n_ctx, 524288);
  assert.deepEqual(row.meta, {
    instanceId: 'inst-1',
    gpus: [0],
    label: 'Qwen IQ3_S',
    aliases: ['qwen'],
    n_ctx: 524288,
  });
  assertChatDiscovery(row, 'Qwen IQ3_S');
});

test('native /v1/models falls back to instance.context when metrics have no window', () => {
  const row = nativeModelRow({ ...baseNative, context: 262144, metrics: { engine: {} }, vision: true });
  assert.equal(row.context_length, 262144);
  assert.equal(row.max_model_len, 262144);
  assert.equal(row.meta.n_ctx, 262144);
});

test('native /v1/models omits context fields when neither value is a positive integer', () => {
  const row = nativeModelRow({ ...baseNative, context: 0, metrics: null });
  assert.equal(row.context_length, undefined);
  assert.equal(row.max_model_len, undefined);
  assert.equal(row.meta.n_ctx, undefined);
  assert.deepEqual(Object.keys(row.meta).sort(), ['aliases', 'gpus', 'instanceId', 'label']);
  assertChatDiscovery(row, 'Qwen IQ3_S');
});

test('native /v1/models sets input modalities from known vision, and omits architecture when unknown', () => {
  assert.deepEqual(
    nativeModelRow({ ...baseNative, context: 32768, vision: true }).architecture,
    { input_modalities: ['text', 'image'], output_modalities: ['text'] },
  );
  assert.deepEqual(
    nativeModelRow({ ...baseNative, context: 32768, vision: false }).architecture,
    { input_modalities: ['text'], output_modalities: ['text'] },
  );
  assert.equal(nativeModelRow({ ...baseNative, context: 32768 }).architecture, undefined);
});

test('native /v1/models uses a configured reasoning effort as thinking.default', () => {
  const row = nativeModelRow({ ...baseNative, context: 32768, reasoningEffort: 'low' });
  assert.equal(row.thinking.default, 'low');
});

test('runtime /v1/models advertises memory.context on standard keys and keeps meta.context', () => {
  const row = runtimeModelRow(
    {
      id: 'rt-1',
      apiModelId: 'qwen38-q4_k_m',
      family: 'qwen38',
      loadedAt: '2026-10-14T00:00:00.000Z',
      gpuIds: ['nvidia:0'],
      memory: { context: 32768 },
    },
    { mode: 'single' },
  );
  assert.equal(row.context_length, 32768);
  assert.equal(row.max_model_len, 32768);
  assert.equal(row.meta.n_ctx, 32768);
  assert.equal(row.meta.context, 32768);
  assert.equal(row.meta.label, 'qwen38');
  assert.equal(row.meta.mode, 'single');
  assertChatDiscovery(row, 'qwen38');
});

test('share allowlist still rejects host and engine discovery paths', async () => {
  const share = createLocalShare({
    host: '127.0.0.1',
    port: 0,
    hasKey: () => false,
    handle: (req, res) => { res.writeHead(200); res.end('ok'); },
  });
  try {
    const { url } = await share.start();
    const origin = new URL(url).origin;
    for (const path of ['/props', '/health', '/v1/status', '/slots', '/metrics', '/status', '/api/runtime', '/api/health']) {
      assert.equal((await fetch(origin + path)).status, 404, path);
    }
    assert.equal((await fetch(origin + '/v1/models')).status, 200);
  } finally {
    await share.stop();
  }
});
