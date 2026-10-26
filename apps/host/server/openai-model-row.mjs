/** Build a GET /v1/models row with a context window clients can read (#3931). */

function positiveInt(value) {
  return Number.isInteger(value) && value > 0 ? value : null;
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
  return row;
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
  return row;
}
