// Scene depth estimation, image-space helpers and the depth composite.

let depthPipe = null;
let depthPipeModel = null;

// Depth Anything V2 through transformers.js. Returns Float32Array (W*H), 1 = near, 0 = far.
export async function estimateDepth(imageCanvas, model, onProgress) {
  const { pipeline, env, RawImage } = await import('@huggingface/transformers');
  env.allowLocalModels = false;
  if (!depthPipe || depthPipeModel !== model) {
    const hasGPU = !!navigator.gpu && !!(await navigator.gpu.requestAdapter().catch(() => null));
    const opts = {
      device: hasGPU ? 'webgpu' : 'wasm',
      dtype: hasGPU ? 'fp32' : 'q8',
      progress_callback: (e) => {
        if (e.status === 'progress' && e.total) onProgress?.(`Baixando modelo de depth… ${Math.round((e.loaded / e.total) * 100)}%`);
      },
    };
    if (model.endsWith('-base')) opts.dtype = hasGPU ? 'fp16' : 'q8';
    try {
      depthPipe = await pipeline('depth-estimation', model, opts);
    } catch (err) {
      if (opts.device !== 'webgpu') throw err;
      opts.device = 'wasm'; opts.dtype = 'q8';
      depthPipe = await pipeline('depth-estimation', model, opts);
    }
    depthPipeModel = model;
  }
  onProgress?.('Calculando depth da cena…');
  const W = imageCanvas.width, H = imageCanvas.height;
  const image = RawImage.fromCanvas(imageCanvas);
  const { predicted_depth } = await depthPipe(image);
  const [h, w] = predicted_depth.dims.slice(-2);
  const out = resizeBilinear(predicted_depth.data, w, h, W, H);
  return normalizeRobust(out);
}

// Load a user-provided depth map image (grayscale, white = near).
export function depthFromImage(img, W, H) {
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, W, H);
  const d = ctx.getImageData(0, 0, W, H).data;
  const out = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) out[i] = (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 765;
  return out;
}

export function resizeBilinear(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh);
  const sx = sw / dw, sy = sh / dh;
  for (let y = 0; y < dh; y++) {
    const fy = Math.min(sh - 1, Math.max(0, (y + 0.5) * sy - 0.5));
    const y0 = Math.floor(fy), y1 = Math.min(sh - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < dw; x++) {
      const fx = Math.min(sw - 1, Math.max(0, (x + 0.5) * sx - 0.5));
      const x0 = Math.floor(fx), x1 = Math.min(sw - 1, x0 + 1), tx = fx - x0;
      const a = src[y0 * sw + x0], b = src[y0 * sw + x1], c = src[y1 * sw + x0], d = src[y1 * sw + x1];
      out[y * dw + x] = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
    }
  }
  return out;
}

function normalizeRobust(a) {
  const sorted = Float32Array.from(a).sort();
  const lo = sorted[Math.floor(sorted.length * 0.002)];
  const hi = sorted[Math.floor(sorted.length * 0.998)];
  const out = new Float32Array(a.length);
  const k = 1 / Math.max(1e-6, hi - lo);
  for (let i = 0; i < a.length; i++) out[i] = Math.min(1, Math.max(0, (a[i] - lo) * k));
  return out;
}

// ---------- Binary mask helpers (Uint8Array 0/1) ----------

function boxAny(src, W, H, r, horizontal) {
  const out = new Uint8Array(W * H);
  const n = horizontal ? W : H, m = horizontal ? H : W;
  for (let j = 0; j < m; j++) {
    let count = 0;
    const at = (i) => (horizontal ? j * W + i : i * W + j);
    for (let i = 0; i < Math.min(n, r); i++) count += src[at(i)];
    for (let i = 0; i < n; i++) {
      if (i + r < n) count += src[at(i + r)];
      if (i - r - 1 >= 0) count -= src[at(i - r - 1)];
      out[at(i)] = count > 0 ? 1 : 0;
    }
  }
  return out;
}

export function dilate(mask, W, H, r) {
  if (r <= 0) return mask.slice();
  return boxAny(boxAny(mask, W, H, r, true), W, H, r, false);
}

export function erode(mask, W, H, r) {
  if (r <= 0) return mask.slice();
  const inv = mask.map((v) => 1 - v);
  return dilate(inv, W, H, r).map((v) => 1 - v);
}

// Fill invalid pixels smoothly from valid neighbours (push-pull pyramid).
export function fillHoles(values, valid, W, H) {
  const levels = [{ v: Float32Array.from(values, (x, i) => (valid[i] ? x : 0)), w: Float32Array.from(valid), W, H }];
  while (levels[levels.length - 1].W > 1 || levels[levels.length - 1].H > 1) {
    const L = levels[levels.length - 1];
    const nW = Math.max(1, Math.ceil(L.W / 2)), nH = Math.max(1, Math.ceil(L.H / 2));
    const v = new Float32Array(nW * nH), w = new Float32Array(nW * nH);
    for (let y = 0; y < nH; y++) for (let x = 0; x < nW; x++) {
      let sv = 0, sw = 0;
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const X = 2 * x + dx, Y = 2 * y + dy;
        if (X >= L.W || Y >= L.H) continue;
        const i = Y * L.W + X;
        sv += L.v[i] * L.w[i]; sw += L.w[i];
      }
      v[y * nW + x] = sw > 0 ? sv / sw : 0;
      w[y * nW + x] = Math.min(1, sw);
    }
    levels.push({ v, w, W: nW, H: nH });
    if (nW === 1 && nH === 1) break;
  }
  for (let l = levels.length - 2; l >= 0; l--) {
    const L = levels[l], C = levels[l + 1];
    const up = resizeBilinear(C.v, C.W, C.H, L.W, L.H);
    for (let i = 0; i < L.v.length; i++) {
      const w = L.w[i];
      if (w < 1) {
        L.v[i] = w > 0 ? L.v[i] * w + up[i] * (1 - w) : up[i];
        L.w[i] = 1;
      }
    }
  }
  return levels[0].v;
}

// Fill holes left by the person with what is most likely behind them: for each hole pixel take the
// nearest valid samples in the 4 directions and average the two farthest, then smooth the patch.
export function fillBehind(values, valid, W, H) {
  const N = W * H;
  const left = new Float32Array(N).fill(NaN), right = new Float32Array(N).fill(NaN);
  const up = new Float32Array(N).fill(NaN), down = new Float32Array(N).fill(NaN);
  for (let y = 0; y < H; y++) {
    let last = NaN;
    for (let x = 0; x < W; x++) { const i = y * W + x; if (valid[i]) last = values[i]; else left[i] = last; }
    last = NaN;
    for (let x = W - 1; x >= 0; x--) { const i = y * W + x; if (valid[i]) last = values[i]; else right[i] = last; }
  }
  for (let x = 0; x < W; x++) {
    let last = NaN;
    for (let y = 0; y < H; y++) { const i = y * W + x; if (valid[i]) last = values[i]; else up[i] = last; }
    last = NaN;
    for (let y = H - 1; y >= 0; y--) { const i = y * W + x; if (valid[i]) last = values[i]; else down[i] = last; }
  }
  const out = Float32Array.from(values);
  const cand = [0, 0, 0, 0];
  for (let i = 0; i < N; i++) {
    if (valid[i]) continue;
    let n = 0;
    for (const a of [left, right, up, down]) if (!Number.isNaN(a[i])) cand[n++] = a[i];
    if (!n) { out[i] = 0; continue; }
    const c = cand.slice(0, n).sort((a, b) => a - b);
    out[i] = n === 1 ? c[0] : (c[0] + c[1]) / 2;
  }
  const smooth = blur(out, W, H, 8);
  for (let i = 0; i < N; i++) if (!valid[i]) out[i] = smooth[i];
  return out;
}

// Rough colour fill of `hole` from the pixels outside `keepOut` (a blurred clean-plate hint).
export function fillColor(ctx, hole, keepOut, W, H) {
  const img = ctx.getImageData(0, 0, W, H);
  const valid = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) valid[i] = keepOut[i] ? 0 : 1;
  for (let ch = 0; ch < 3; ch++) {
    const v = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) v[i] = img.data[i * 4 + ch];
    const f = fillHoles(v, valid, W, H);
    for (let i = 0; i < W * H; i++) if (hole[i]) img.data[i * 4 + ch] = f[i];
  }
  ctx.putImageData(img, 0, 0);
}

// ---------- Composite ----------

// Fit disparity = a / z + b from pixels where the original-pose mannequin overlaps the person.
export function fitDepthMapping(D, z0, person, W, H) {
  const inner = erode(person, W, H, 4);
  let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  const xs = [], ys = [];
  for (let i = 0; i < W * H; i += 3) {
    if (!inner[i] || z0[i] <= 0) continue;
    const x = 1 / z0[i], y = D[i];
    xs.push(x); ys.push(y);
    n++; sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  if (n < 50) return null;
  const mx = sx / n, my = sy / n;
  const varx = sxx / n - mx * mx, cov = sxy / n - mx * my;
  // Prior: pure inverse depth through the medians (b = 0). Ridge toward it when the fit is ill-conditioned.
  const a0 = my / mx;
  const lambda = 1e-3 * mx * mx;
  let a = (cov + lambda * a0) / (varx + lambda);
  if (!(a > 0)) a = a0;
  const b = my - a * mx;
  return { a, b };
}

/**
 * Build the new depth map.
 * D: original scene depth (1 near). person: segmentation (0/1).
 * r0/r1: id+depth renders of the original and new pose. changed: Set of segment ids.
 */
export function composeDepth({ D, person, r0, r1, changed, W, H, map, removePad, frontBias, offset }) {
  const personDil = dilate(person, W, H, 3);
  const bg = fillBehind(D, dilate(person, W, H, 6).map((v) => 1 - v), W, H);

  const old = new Uint8Array(W * H);
  const neu = new Uint8Array(W * H);
  const unchangedNew = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) {
    if (r0.id[i] && changed.has(r0.id[i])) old[i] = 1;
    if (r1.id[i]) {
      if (changed.has(r1.id[i])) neu[i] = 1; else unchangedNew[i] = 1;
    }
  }
  // Every person pixel belongs to the nearest mannequin part of the original pose, so fingers,
  // sleeves and hair attached to a moved limb are removed with it.
  const label = propagateLabels(r0.id, personDil, W, H);
  const seedRemove = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) seedRemove[i] = (label[i] && changed.has(label[i])) || old[i] ? 1 : 0;
  const grown = dilate(seedRemove, W, H, removePad);
  const remove = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) remove[i] = grown[i] && personDil[i] && !unchangedNew[i] ? 1 : 0;

  const val = (z) => map.a / z + map.b + offset;
  const out = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    let v = remove[i] ? bg[i] : D[i];
    if (r1.id[i]) {
      const m = val(r1.z[i]);
      if (neu[i]) {
        // Moved parts: strict z-test against the scene, lenient against the person's own body.
        const lenient = person[i] && !remove[i];
        if (m > v) v = m;
        else if (lenient && m > v - frontBias) v += 0.012;
      } else if (remove[i]) {
        v = Math.max(v, m);
      }
    }
    out[i] = Math.min(1, Math.max(0, v));
  }
  return { depth: out, removed: remove, newMask: neu, oldMask: old };
}

// Multi-source BFS: give each pixel inside `region` the id of the nearest seeded pixel.
export function propagateLabels(ids, region, W, H) {
  const label = new Uint8Array(W * H);
  const queue = new Int32Array(W * H);
  let head = 0, tail = 0;
  for (let i = 0; i < W * H; i++) if (ids[i] && region[i]) { label[i] = ids[i]; queue[tail++] = i; }
  while (head < tail) {
    const i = queue[head++];
    const x = i % W;
    const l = label[i];
    if (x > 0 && region[i - 1] && !label[i - 1]) { label[i - 1] = l; queue[tail++] = i - 1; }
    if (x < W - 1 && region[i + 1] && !label[i + 1]) { label[i + 1] = l; queue[tail++] = i + 1; }
    if (i >= W && region[i - W] && !label[i - W]) { label[i - W] = l; queue[tail++] = i - W; }
    if (i < W * (H - 1) && region[i + W] && !label[i + W]) { label[i + W] = l; queue[tail++] = i + W; }
  }
  return label;
}

// Change mask for inpainting: original + new location of the moved parts, padded and feathered.
export function changeMask(removed, newMask, W, H, pad, feather) {
  const u = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) u[i] = removed[i] || newMask[i] ? 1 : 0;
  const d = dilate(u, W, H, pad);
  if (feather <= 0) return Float32Array.from(d);
  return blur(Float32Array.from(d), W, H, feather);
}

export function blur(src, W, H, r) {
  const pass = (a, horizontal) => {
    const out = new Float32Array(W * H);
    const n = horizontal ? W : H, m = horizontal ? H : W;
    const at = (j, i) => (horizontal ? j * W + i : i * W + j);
    for (let j = 0; j < m; j++) {
      let s = 0, c = 0;
      for (let i = -r; i <= r; i++) if (i >= 0 && i < n) { s += a[at(j, i)]; c++; }
      for (let i = 0; i < n; i++) {
        out[at(j, i)] = s / c;
        const add = i + r + 1, rem = i - r;
        if (add < n) { s += a[at(j, add)]; c++; }
        if (rem >= 0) { s -= a[at(j, rem)]; c--; }
      }
    }
    return out;
  };
  return pass(pass(src, true), false);
}

export function grayToCanvas(values, W, H) {
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const g = Math.round(Math.min(1, Math.max(0, values[i])) * 255);
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = g;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}
