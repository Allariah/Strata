/** Build a GET /v1/models row that standard clients can read for context, thinking, and tools. */

export const THINKING_EFFORTS = ['none', 'low', 'medium', 'high'];
export const THINKING_WIRE = { request: 'reasoning_effort', stream: 'reasoning_content' };

/** Same hints as the engine's chat_template_caps for Strata's bundled chat template. */
export const STRATA_CHAT_CAPS = {
  supports_tools: true,
  supports_tool_calls: true,
  supports_system_role: true,
  supports_parallel_tool_calls: true,
  supports_preserve_reasoning: true,
};

function positiveInt(value) {
  return Number.isInteger(value) && value > 0 ? value : null;
}

function withChatDiscovery(row, { name, effort = 'high', caps = STRATA_CHAT_CAPS } = {}) {
  if (name) row.name = name;
  row.thinking = {
    supported: true,
    default: THINKING_EFFORTS.includes(effort) ? effort : 'high',
    efforts: [...THINKING_EFFORTS],
  };
  row.wire = { ...THINKING_WIRE };
  Object.assign(row, caps);
  return row;
}

/** Context window from a native instance: engine metrics first, then the load-time value. */
export function nativeContext(instance) {
  return positiveInt(instance?.metrics?.engine?.max_context) ?? positiveInt(instance?.context);
}

/** A ready native instance as an OpenAI-style model object. */
export function nativeModelRow(instance) {
  const meta = {
    instanceId: instance.id,
    gpus: instance.gpus,
    label: instance.label,
    aliases: instance.aliases,
  };
  const row = { id: instance.apiModelId, object: 'model', owned_by: 'strata', meta };
  const ctx = nativeContext(instance);
  if (ctx != null) {
    row.context_length = ctx;
    row.max_model_len = ctx;
    meta.n_ctx = ctx;
  }
  if (instance.vision === true) {
    row.architecture = { input_modalities: ['text', 'image'], output_modalities: ['text'] };
  } else if (instance.vision === false) {
    row.architecture = { input_modalities: ['text'], output_modalities: ['text'] };
  }
  return withChatDiscovery(row, {
    name: instance.label,
    effort: instance.reasoningEffort,
    caps: instance.chatCaps ?? STRATA_CHAT_CAPS,
  });
}

/** A runtime (simulated) instance as an OpenAI-style model object. */
export function runtimeModelRow(instance, runtime) {
  const context = instance.memory.context;
  const meta = {
    label: instance.family,
    context,
    instanceId: instance.id,
    gpuIds: instance.gpuIds,
    mode: runtime.mode,
  };
  const row = {
    id: instance.apiModelId,
    object: 'model',
    created: Math.floor(new Date(instance.loadedAt).getTime() / 1000),
    owned_by: 'strata',
    meta,
  };
  const ctx = positiveInt(context);
  if (ctx != null) {
    row.context_length = ctx;
    row.max_model_len = ctx;
    meta.n_ctx = ctx;
  }
  return withChatDiscovery(row, { name: instance.family });
}
