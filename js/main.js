import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  JOINTS, JOINT_NAMES, Pose, poseFromLandmarks, defaultPose, dragJoint, nudgeDepth, focalPx,
  changedSegments, SEGMENTS,
} from './skeleton.js';
import { Mannequin, Offscreen, SEGMENT_ID, makeClayMaterial, addStudioLights } from './mannequin.js';
import {
  estimateDepth, depthFromImage, resizeBilinear, dilate, composeDepth, changeMask, fitDepthMapping, grayToCanvas, fillColor, propagateLabels,
} from './depth.js';
import { renderOpenPose, canvasToBlob, downloadBlob, downloadZip } from './exports.js';

const MAX_SIDE = 2048;
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task';
const MP_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';

const $ = (s) => document.querySelector(s);
const ui = {
  stage: $('#stage'), wrap: $('#viewWrap'), photo: $('#photo'), view: $('#view3d'), side: $('#side'),
  sidePanel: $('#sidePanel'), empty: $('#empty'), status: $('#status'), file: $('#file'), depthFile: $('#depthFile'),
  fov: $('#fov'), fovVal: $('#fovVal'), opacity: $('#opacity'), ik: $('#ik'), fullBody: $('#fullBody'),
  removePad: $('#removePad'), maskPad: $('#maskPad'), feather: $('#feather'), offset: $('#offset'),
  prompt: $('#prompt'), cleanGuide: $('#cleanGuide'), depthModel: $('#depthModel'), selName: $('#selName'),
};

const state = {
  W: 0, H: 0, fov: 40, photoCanvas: null, fileName: 'imagem',
  det: null, person: null, D: null, depthBusy: false,
  pose0: null, pose: null, poseEdited: false, mode: 'align', tab: 'edit',
  selected: null, hovered: null, dragging: null, undo: [], redo: [],
  outputs: null, outputsDirty: true,
};

// ---------- Three.js setup ----------
const renderer = new THREE.WebGLRenderer({ canvas: ui.view, alpha: true, antialias: true, preserveDrawingBuffer: false });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
const sideRenderer = new THREE.WebGLRenderer({ canvas: ui.side, antialias: true });
sideRenderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
sideRenderer.setClearColor(0x1b1c1f);

const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 200);
const sideCam = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
sideCam.layers.enable(1);
const controls = new OrbitControls(sideCam, ui.side);
controls.enableDamping = false;

const scene = new THREE.Scene();
addStudioLights(scene);
const clay = makeClayMaterial({ transparent: true, opacity: 0.6, depthWrite: true });
const mannequin = new Mannequin(clay);
scene.add(mannequin.group);

const offMannequin = new Mannequin(makeClayMaterial());
const offscreen = new Offscreen(renderer);

// Skeleton lines (current pose) and ghost lines (original pose, shown while editing the new pose).
const BONES = JOINT_NAMES.filter((n) => JOINTS[n].parent && !['eyeL', 'eyeR', 'earL', 'earR'].includes(n))
  .map((n) => [JOINTS[n].parent, n]);
function makeLines(color, opacity) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(BONES.length * 6), 3));
  const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthTest: false }));
  l.renderOrder = 998;
  l.frustumCulled = false;
  scene.add(l);
  return l;
}
const lines = makeLines(0xffffff, 0.7);
const ghost = makeLines(0xff4fa3, 0.75);

const SIDE_COLORS = { L: 0x4aa8ff, R: 0xff8a3d, C: 0xf2f2f2 };
const handles = {};
const handleGeo = new THREE.SphereGeometry(1, 16, 12);
for (const n of JOINT_NAMES) {
  if (!JOINTS[n].handle) continue;
  const m = new THREE.Mesh(handleGeo, new THREE.MeshBasicMaterial({
    color: SIDE_COLORS[JOINTS[n].side], depthTest: false, transparent: true, opacity: 0.95,
  }));
  m.renderOrder = 999;
  m.userData.joint = n;
  scene.add(m);
  handles[n] = m;
}
const camHelper = new THREE.CameraHelper(camera);
camHelper.layers.set(1);
scene.add(camHelper);

function activePose() { return state.mode === 'align' ? state.pose0 : state.pose; }

function setLines(lineObj, pose) {
  const arr = lineObj.geometry.attributes.position.array;
  BONES.forEach(([a, b], i) => { pose.p[a].toArray(arr, i * 6); pose.p[b].toArray(arr, i * 6 + 3); });
  lineObj.geometry.attributes.position.needsUpdate = true;
}

function syncScene() {
  const pose = activePose();
  if (!pose) return;
  mannequin.update(pose);
  setLines(lines, pose);
  ghost.visible = state.mode === 'pose' && state.tab === 'edit';
  if (ghost.visible) setLines(ghost, state.pose0);
  for (const [n, h] of Object.entries(handles)) {
    h.position.copy(pose.p[n]);
    const c = SIDE_COLORS[JOINTS[n].side];
    h.material.color.setHex(n === state.selected || n === state.dragging ? 0xffe14d : n === state.hovered ? 0xffffff : c);
  }
  ui.selName.textContent = state.selected ? JOINT_LABELS[state.selected] || state.selected : '—';
}

function scaleHandles(cam, viewportH) {
  const fpx = viewportH / 2 / Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2);
  const tmp = new THREE.Vector3();
  for (const [n, h] of Object.entries(handles)) {
    const d = tmp.copy(h.position).applyMatrix4(cam.matrixWorldInverse).z;
    const px = n === state.selected || n === state.hovered ? 8 : 6;
    h.scale.setScalar((px * Math.abs(d)) / fpx);
  }
}

let raf = 0;
function requestRender() {
  if (raf) return;
  raf = requestAnimationFrame(() => {
    raf = 0;
    syncScene();
    const showOverlay = state.tab === 'edit' && !!state.pose0;
    mannequin.group.visible = showOverlay;
    lines.visible = showOverlay;
    for (const h of Object.values(handles)) h.visible = showOverlay;
    camera.updateMatrixWorld();
    scaleHandles(camera, ui.view.clientHeight || 1);
    renderer.render(scene, camera);
    if (!ui.sidePanel.hidden && state.pose0) {
      mannequin.group.visible = true; lines.visible = true;
      for (const h of Object.values(handles)) h.visible = true;
      const prevOpacity = clay.opacity;
      clay.opacity = 1;
      sideCam.updateMatrixWorld();
      scaleHandles(sideCam, ui.side.clientHeight || 1);
      sideRenderer.render(scene, sideCam);
      clay.opacity = prevOpacity;
    }
  });
}
controls.addEventListener('change', requestRender);

// ---------- Layout ----------
function layout() {
  if (!state.W) return;
  const r = ui.stage.getBoundingClientRect();
  const pad = 24;
  const s = Math.min((r.width - pad * 2) / state.W, (r.height - pad * 2) / state.H);
  const w = Math.max(50, Math.floor(state.W * s)), h = Math.max(50, Math.floor(state.H * s));
  ui.wrap.style.width = w + 'px';
  ui.wrap.style.height = h + 'px';
  renderer.setSize(w, h, true);
  camera.aspect = state.W / state.H;
  camera.updateProjectionMatrix();
  const sw = ui.side.clientWidth, sh = ui.side.clientHeight;
  if (sw && sh) { sideRenderer.setSize(sw, sh, false); sideCam.aspect = sw / sh; sideCam.updateProjectionMatrix(); }
  requestRender();
}
new ResizeObserver(layout).observe(ui.stage);
new ResizeObserver(layout).observe(ui.side);

function setStatus(msg, kind = '') {
  ui.status.textContent = msg;
  ui.status.dataset.kind = kind;
}

// ---------- Loading an image ----------
async function loadImage(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await img.decode();
  state.fileName = file.name.replace(/\.[^.]+$/, '') || 'imagem';
  const s = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  state.W = Math.round(img.naturalWidth * s);
  state.H = Math.round(img.naturalHeight * s);
  const c = document.createElement('canvas');
  c.width = state.W; c.height = state.H;
  c.getContext('2d').drawImage(img, 0, 0, state.W, state.H);
  state.photoCanvas = c;
  state.D = null; state.det = null; state.person = null;
  state.undo = []; state.redo = []; state.selected = null;
  ui.photo.width = state.W; ui.photo.height = state.H;
  ui.empty.hidden = true;
  ui.wrap.hidden = false;
  document.body.classList.add('has-image');
  setMode('align');
  setTab('edit');
  layout();
  drawPhoto();

  await detectPose();
  runDepth();
}

function drawPhoto(source = state.photoCanvas) {
  const ctx = ui.photo.getContext('2d');
  ctx.clearRect(0, 0, state.W, state.H);
  ctx.drawImage(source, 0, 0, state.W, state.H);
}

let landmarker = null;
async function getLandmarker() {
  if (landmarker) return landmarker;
  const { FilesetResolver, PoseLandmarker } = await import('@mediapipe/tasks-vision');
  const vision = await FilesetResolver.forVisionTasks(MP_WASM);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: POSE_MODEL, delegate },
    runningMode: 'IMAGE', numPoses: 4, outputSegmentationMasks: true,
    minPoseDetectionConfidence: 0.4, minPosePresenceConfidence: 0.4,
  });
  try {
    landmarker = await PoseLandmarker.createFromOptions(vision, opts('GPU'));
  } catch (e) {
    console.warn('GPU delegate failed, using CPU', e);
    landmarker = await PoseLandmarker.createFromOptions(vision, opts('CPU'));
  }
  return landmarker;
}

async function detectPose() {
  setStatus('Detectando a pose…', 'busy');
  try {
    const lm = await getLandmarker();
    const res = lm.detect(state.photoCanvas);
    // Several people can be in frame: keep the largest one.
    let best = -1, bestArea = 0;
    res.landmarks.forEach((pts, i) => {
      const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
      const area = (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
      if (area > bestArea) { bestArea = area; best = i; }
    });
    if (best >= 0) {
      state.det = { lm2d: res.landmarks[best], world: res.worldLandmarks[best] };
      const mask = res.segmentationMasks?.[best];
      if (mask) {
        const m = mask.getAsFloat32Array();
        const full = mask.width === state.W && mask.height === state.H
          ? m : resizeBilinear(m, mask.width, mask.height, state.W, state.H);
        state.person = Uint8Array.from(full, (v) => (v > 0.5 ? 1 : 0));
      }
      res.segmentationMasks?.forEach((mk) => mk.close());
      setStatus('Pose detectada. Ajuste os pontos que estiverem fora do lugar.', 'ok');
    } else {
      setStatus('Nenhuma pessoa detectada. Posicione o boneco manualmente.', 'warn');
    }
  } catch (e) {
    console.error(e);
    setStatus('Falha ao detectar pose: ' + e.message, 'error');
  }
  fitPose();
}

function fitPose() {
  state.pose0 = state.det
    ? poseFromLandmarks(state.det.lm2d, state.det.world, state.W, state.H, state.fov)
    : defaultPose(state.W, state.H, state.fov);
  state.pose = state.pose0.clone();
  state.poseEdited = false;
  camera.fov = state.fov;
  camera.updateProjectionMatrix();
  resetSideCam();
  markDirty();
}

function resetSideCam() {
  const p = state.pose0.p.pelvis.clone().lerp(state.pose0.p.neck, 0.3);
  const s = state.pose0.scale();
  controls.target.copy(p);
  sideCam.position.copy(p).add(new THREE.Vector3(2.6 * s, 0.4 * s, 0));
  controls.update();
}

async function runDepth() {
  if (state.depthBusy) return;
  state.depthBusy = true;
  const canvas = state.photoCanvas;
  try {
    const D = await estimateDepth(canvas, ui.depthModel.value, (m) => setStatus(m, 'busy'));
    if (canvas !== state.photoCanvas) return;
    state.D = D;
    setStatus('Depth pronto. Mude a pose e confira as abas de saída.', 'ok');
    markDirty();
  } catch (e) {
    console.error(e);
    setStatus('Não foi possível calcular o depth automaticamente (' + e.message + '). Você pode carregar um depth map próprio.', 'error');
  } finally {
    state.depthBusy = false;
  }
}

// ---------- Interaction ----------
const raycaster = new THREE.Raycaster();
function rayFrom(ev, canvas, cam) {
  const r = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, cam);
  return raycaster.ray.clone();
}

function pickHandle(ev, canvas, cam, viewportH) {
  // Pick by screen distance (handles are small); nearest within 14 px.
  const r = canvas.getBoundingClientRect();
  let best = null, bestD = 14;
  for (const [n, h] of Object.entries(handles)) {
    const p = h.position.clone().project(cam);
    if (p.z > 1) continue;
    const x = (p.x * 0.5 + 0.5) * r.width, y = (-p.y * 0.5 + 0.5) * r.height;
    const d = Math.hypot(ev.clientX - r.left - x, ev.clientY - r.top - y);
    if (d < bestD) { bestD = d; best = n; }
  }
  return best;
}

function pushUndo() {
  state.undo.push({ mode: state.mode, data: activePose().toArray(), edited: state.poseEdited });
  if (state.undo.length > 100) state.undo.shift();
  state.redo = [];
}
function undo(redo = false) {
  const from = redo ? state.redo : state.undo, to = redo ? state.undo : state.redo;
  const snap = from.pop();
  if (!snap) return;
  const pose = snap.mode === 'align' ? state.pose0 : state.pose;
  to.push({ mode: snap.mode, data: pose.toArray(), edited: state.poseEdited });
  pose.fromArray(snap.data);
  if (snap.mode === 'align' && !state.poseEdited) state.pose.copy(state.pose0);
  markDirty();
}

function bindPointer(canvas, getCam, isSide) {
  canvas.addEventListener('pointerdown', (ev) => {
    if (!state.pose0 || (!isSide && state.tab !== 'edit')) return;
    const cam = getCam();
    const hit = pickHandle(ev, canvas, cam);
    if (!hit) { if (!isSide) { state.selected = null; requestRender(); } return; }
    ev.preventDefault();
    if (isSide) controls.enabled = false;
    canvas.setPointerCapture(ev.pointerId);
    pushUndo();
    state.dragging = hit;
    state.selected = hit;
    requestRender();
  });
  canvas.addEventListener('pointermove', (ev) => {
    if (!state.pose0) return;
    const cam = getCam();
    if (state.dragging) {
      const ray = rayFrom(ev, canvas, cam);
      const viewDir = cam.getWorldDirection(new THREE.Vector3());
      dragJoint(activePose(), state.dragging, ray, viewDir, { mode: state.mode, ik: ui.ik.checked });
      afterEdit();
      requestRender();
    } else if (isSide || state.tab === 'edit') {
      const h = pickHandle(ev, canvas, cam);
      if (h !== state.hovered) { state.hovered = h; canvas.style.cursor = h ? 'grab' : ''; requestRender(); }
    }
  });
  const end = (ev) => {
    if (!state.dragging) return;
    state.dragging = null;
    controls.enabled = true;
    if (canvas.hasPointerCapture(ev.pointerId)) canvas.releasePointerCapture(ev.pointerId);
    markDirty();
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('wheel', (ev) => {
    if (isSide || !state.pose0 || state.tab !== 'edit') return;
    const name = state.hovered || state.selected;
    if (!name) return;
    ev.preventDefault();
    pushUndo();
    const amount = -ev.deltaY * 0.0006 * state.pose0.scale();
    nudgeDepth(activePose(), name, amount, new THREE.Vector3(0, 0, -1), { mode: state.mode, ik: ui.ik.checked });
    afterEdit();
    markDirty();
  }, { passive: false });
}
bindPointer(ui.view, () => camera, false);
bindPointer(ui.side, () => sideCam, true);

function afterEdit() {
  if (state.mode === 'align' && !state.poseEdited) state.pose.copy(state.pose0);
  if (state.mode === 'pose') state.poseEdited = true;
  state.outputsDirty = true;
}

let dirtyTimer = 0;
function markDirty() {
  state.outputsDirty = true;
  requestRender();
  if (state.tab !== 'edit') {
    clearTimeout(dirtyTimer);
    dirtyTimer = setTimeout(showTab, 60);
  }
}

window.addEventListener('keydown', (ev) => {
  if (ev.target.matches('input, textarea, select')) return;
  const mod = ev.ctrlKey || ev.metaKey;
  if (mod && ev.key.toLowerCase() === 'z') { ev.preventDefault(); undo(ev.shiftKey); }
  else if (mod && ev.key.toLowerCase() === 'y') { ev.preventDefault(); undo(true); }
  else if (ev.key === 'Escape') { state.selected = null; requestRender(); }
  else if (state.selected && (ev.key === 'PageUp' || ev.key === 'PageDown' || ev.key === 'q' || ev.key === 'e')) {
    pushUndo();
    const dir = ev.key === 'PageUp' || ev.key === 'e' ? 1 : -1;
    nudgeDepth(activePose(), state.selected, dir * 0.03 * state.pose0.scale(), new THREE.Vector3(0, 0, -1), { mode: state.mode, ik: ui.ik.checked });
    afterEdit();
    markDirty();
  }
});

// ---------- Modes & tabs ----------
function setMode(mode) {
  if (mode === 'pose' && state.pose0 && !state.poseEdited) state.pose.copy(state.pose0);
  if (mode === 'align' && state.poseEdited) {
    if (!confirm('Voltar ao alinhamento descarta a pose nova. Continuar?')) { syncModeButtons(); return; }
    state.pose.copy(state.pose0);
    state.poseEdited = false;
  }
  state.mode = mode;
  clay.opacity = Number(ui.opacity.value);
  syncModeButtons();
  markDirty();
}
function syncModeButtons() {
  document.querySelectorAll('[data-mode]').forEach((b) => b.classList.toggle('active', b.dataset.mode === state.mode));
  document.body.dataset.mode = state.mode;
}
document.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));

function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll('[data-tab]').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
  showTab();
}
document.querySelectorAll('[data-tab]').forEach((b) => b.addEventListener('click', () => setTab(b.dataset.tab)));

// ---------- Outputs ----------
const CHANGED_EPS = 0.004;

function computeOutputs() {
  if (!state.outputsDirty && state.outputs) return state.outputs;
  const { W, H } = state;
  const f = focalPx(H, state.fov);
  camera.updateMatrixWorld();
  const r0 = offscreen.renderIdDepth(offMannequin, state.pose0, camera, W, H);
  const r1 = offscreen.renderIdDepth(offMannequin, state.pose, camera, W, H);
  const fullBody = ui.fullBody.checked;
  const changedNames = fullBody ? new Set(SEGMENTS) : changedSegments(state.pose0, state.pose, CHANGED_EPS * state.pose0.scale());
  const changed = new Set([...changedNames].map((s) => SEGMENT_ID[s]));

  let person = state.person;
  if (!person) person = dilate(Uint8Array.from(r0.id, (v) => (v ? 1 : 0)), W, H, 6);

  const out = { changedNames, f };
  let removed, newMask;
  if (state.D) {
    let map = fitDepthMapping(state.D, r0.z, person, W, H);
    if (!map) map = { a: 0.6 * -state.pose0.p.pelvis.z, b: 0 };
    const comp = composeDepth({
      D: state.D, person, r0, r1, changed, W, H, map,
      removePad: Number(ui.removePad.value), frontBias: 0.04, offset: Number(ui.offset.value),
    });
    out.depth = grayToCanvas(comp.depth, W, H);
    removed = comp.removed; newMask = comp.newMask;
  } else {
    // Without scene depth we can still produce the change mask from the mannequin alone.
    const personDil = dilate(person, W, H, 3);
    const label = propagateLabels(r0.id, personDil, W, H);
    const seed = new Uint8Array(W * H);
    newMask = new Uint8Array(W * H);
    for (let i = 0; i < W * H; i++) {
      if (label[i] && changed.has(label[i])) seed[i] = 1;
      if (r1.id[i] && changed.has(r1.id[i])) newMask[i] = 1;
    }
    removed = dilate(seed, W, H, Number(ui.removePad.value)).map((v, i) => v & personDil[i]);
  }
  out.mask = grayToCanvas(changeMask(removed, newMask, W, H, Number(ui.maskPad.value), Number(ui.feather.value)), W, H);
  out.poseRender = offscreen.renderShaded(offMannequin, state.pose, camera, W, H, { bg: 0xd9d9d9 });

  const guide = document.createElement('canvas');
  guide.width = W; guide.height = H;
  const g = guide.getContext('2d');
  g.drawImage(state.photoCanvas, 0, 0, W, H);
  if (changedNames.size && ui.cleanGuide.checked && !fullBody) {
    // Erase the old limb so the guide doesn't show two arms.
    const hole = removed.map((v, i) => (v && !newMask[i] ? 1 : 0));
    fillColor(g, dilate(hole, W, H, 4), dilate(person, W, H, 8), W, H);
  }
  if (changedNames.size) {
    g.drawImage(offscreen.renderShaded(offMannequin, state.pose, camera, W, H, { segments: changedNames }), 0, 0);
  } else {
    g.globalAlpha = 0.6;
    g.drawImage(offscreen.renderShaded(offMannequin, state.pose, camera, W, H), 0, 0);
    g.globalAlpha = 1;
  }
  out.guide = guide;
  out.openpose = renderOpenPose(state.pose, W, H, f);
  state.outputs = out;
  state.outputsDirty = false;
  updatePrompt(changedNames);
  return out;
}

function showTab() {
  if (!state.photoCanvas) return;
  ui.view.style.visibility = state.tab === 'edit' ? 'visible' : 'hidden';
  if (state.tab === 'edit') { drawPhoto(); requestRender(); return; }
  if (!state.pose0) return;
  const out = computeOutputs();
  const map = { depth: out.depth, mask: out.mask, pose: out.poseRender, guide: out.guide, openpose: out.openpose };
  const c = map[state.tab];
  if (c) drawPhoto(c);
  else {
    drawPhoto();
    const ctx = ui.photo.getContext('2d');
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(0, 0, state.W, state.H);
    ctx.fillStyle = '#fff';
    ctx.font = `${Math.round(state.H / 24)}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText(state.depthBusy ? 'Calculando depth…' : 'Depth indisponível — carregue um depth map', state.W / 2, state.H / 2);
  }
  requestRender();
}

// ---------- Prompt suggestion ----------
const PART_NAMES = {
  head: 'head', torso: 'torso',
  upperArmL: 'left arm', forearmL: 'left arm', handL: 'left hand',
  upperArmR: 'right arm', forearmR: 'right arm', handR: 'right hand',
  thighL: 'left leg', shinL: 'left leg', footL: 'left foot',
  thighR: 'right leg', shinR: 'right leg', footR: 'right foot',
};
const JOINT_LABELS = {
  pelvis: 'Quadril (centro)', hipL: 'Quadril esq.', hipR: 'Quadril dir.', kneeL: 'Joelho esq.', kneeR: 'Joelho dir.',
  ankleL: 'Tornozelo esq.', ankleR: 'Tornozelo dir.', toeL: 'Pé esq.', toeR: 'Pé dir.', neck: 'Pescoço',
  shL: 'Ombro esq.', shR: 'Ombro dir.', elL: 'Cotovelo esq.', elR: 'Cotovelo dir.', wrL: 'Pulso esq.',
  wrR: 'Pulso dir.', handL: 'Mão esq.', handR: 'Mão dir.', head: 'Cabeça', nose: 'Rosto (direção)',
};

function updatePrompt(changedNames) {
  if (ui.prompt.dataset.userEdited === '1') return;
  let parts = [...new Set([...changedNames].map((s) => PART_NAMES[s]))];
  if (parts.includes('left arm')) parts = parts.filter((p) => p !== 'left hand');
  if (parts.includes('right arm')) parts = parts.filter((p) => p !== 'right hand');
  if (parts.includes('left leg')) parts = parts.filter((p) => p !== 'left foot');
  if (parts.includes('right leg')) parts = parts.filter((p) => p !== 'right foot');
  const what = ui.fullBody.checked || parts.length > 4 || !parts.length
    ? 'the body pose'
    : 'only the position of the person\'s ' + parts.join(', ').replace(/, ([^,]*)$/, ' and $1') + ' (their own left/right)';
  ui.prompt.value =
`Edit image 1. Change ${what} to match the gray mannequin in image 2. Same camera, same framing, same lens.
Image 2 shows the new pose only; do not copy its gray color or material.
Keep the face, expression, hair, clothing, hands' objects, lighting, film look and the entire background exactly as in image 1.`;
}
ui.prompt.addEventListener('input', () => { ui.prompt.dataset.userEdited = '1'; });
$('#resetPrompt').addEventListener('click', () => { ui.prompt.dataset.userEdited = ''; updatePrompt(computeOutputs().changedNames); });
$('#copyPrompt').addEventListener('click', async () => {
  await navigator.clipboard.writeText(ui.prompt.value).catch(() => {});
  setStatus('Prompt copiado.', 'ok');
});

// ---------- Controls ----------
ui.file.addEventListener('change', () => ui.file.files[0] && loadImage(ui.file.files[0]));
$('#pick').addEventListener('click', () => ui.file.click());
$('#emptyPick').addEventListener('click', () => ui.file.click());
for (const ev of ['dragover', 'dragenter']) ui.stage.addEventListener(ev, (e) => { e.preventDefault(); ui.stage.classList.add('drop'); });
for (const ev of ['dragleave', 'drop']) ui.stage.addEventListener(ev, () => ui.stage.classList.remove('drop'));
ui.stage.addEventListener('drop', (e) => {
  e.preventDefault();
  const f = [...e.dataTransfer.files].find((x) => x.type.startsWith('image/'));
  if (f) loadImage(f);
});

ui.depthFile.addEventListener('change', async () => {
  const file = ui.depthFile.files[0];
  if (!file || !state.W) return;
  const img = new Image();
  img.src = URL.createObjectURL(file);
  await img.decode();
  state.D = depthFromImage(img, state.W, state.H);
  setStatus('Depth map carregado.', 'ok');
  markDirty();
});
$('#depthUpload').addEventListener('click', () => ui.depthFile.click());
$('#depthRerun').addEventListener('click', () => { if (state.photoCanvas) runDepth(); });

ui.fov.addEventListener('input', () => { ui.fovVal.textContent = ui.fov.value + '°'; });
ui.fov.addEventListener('change', () => {
  state.fov = Number(ui.fov.value);
  if (state.photoCanvas) fitPose();
});
ui.opacity.addEventListener('input', () => { clay.opacity = Number(ui.opacity.value); requestRender(); });
for (const el of [ui.fullBody, ui.cleanGuide, ui.removePad, ui.maskPad, ui.feather, ui.offset]) el.addEventListener('input', markDirty);

$('#refit').addEventListener('click', () => { if (state.photoCanvas) { pushUndo(); fitPose(); } });
$('#resetPose').addEventListener('click', () => {
  if (!state.pose0) return;
  pushUndo();
  state.pose.copy(state.pose0);
  state.poseEdited = false;
  markDirty();
});
$('#undo').addEventListener('click', () => undo(false));
$('#redo').addEventListener('click', () => undo(true));
$('#toggleSide').addEventListener('click', () => {
  ui.sidePanel.hidden = !ui.sidePanel.hidden;
  layout();
});
$('#sideReset').addEventListener('click', () => { if (state.pose0) { resetSideCam(); requestRender(); } });

async function exportFiles() {
  const out = computeOutputs();
  const files = [];
  const add = async (name, canvas) => canvas && files.push([`${state.fileName}_${name}.png`, await canvasToBlob(canvas)]);
  await add('depth', out.depth);
  await add('mask', out.mask);
  await add('mannequin', out.poseRender);
  await add('guide', out.guide);
  await add('openpose', out.openpose);
  const json = {
    image: { width: state.W, height: state.H }, fovDeg: state.fov,
    joints: Object.fromEntries(JOINT_NAMES.map((n) => [n, state.pose.p[n].toArray()])),
    originalJoints: Object.fromEntries(JOINT_NAMES.map((n) => [n, state.pose0.p[n].toArray()])),
    changed: [...out.changedNames], prompt: ui.prompt.value,
  };
  files.push([`${state.fileName}_pose.json`, new Blob([JSON.stringify(json, null, 1)], { type: 'application/json' })]);
  return files;
}
$('#downloadZip').addEventListener('click', async () => {
  if (!state.pose0) return;
  await downloadZip(await exportFiles(), `${state.fileName}_pose-studio.zip`);
});
document.querySelectorAll('[data-download]').forEach((b) => b.addEventListener('click', async () => {
  if (!state.pose0) return;
  const out = computeOutputs();
  const key = b.dataset.download;
  const c = { depth: out.depth, mask: out.mask, mannequin: out.poseRender, guide: out.guide, openpose: out.openpose }[key];
  if (!c) { setStatus('Ainda não há depth para exportar.', 'warn'); return; }
  downloadBlob(await canvasToBlob(c), `${state.fileName}_${key}.png`);
}));

// Expose for debugging / automated checks.
window.poseStudio = {
  state, computeOutputs, setTab, setMode, loadImage,
  handleScreen(name) {
    const r = ui.view.getBoundingClientRect();
    const p = activePose().p[name].clone().project(camera);
    return { x: r.left + (p.x * 0.5 + 0.5) * r.width, y: r.top + (-p.y * 0.5 + 0.5) * r.height };
  },
};
setStatus('Carregue uma imagem para começar.');
