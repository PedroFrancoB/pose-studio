// Skeleton definition, fitting from MediaPipe landmarks, and pose editing (FK / IK).
import * as THREE from 'three';

// Joint list. `parent` defines the hierarchy used for FK rotations.
// side: 'L' / 'R' / 'C' (person's own left/right), handle: shown as a draggable handle.
export const JOINTS = {
  pelvis: { parent: null, side: 'C', handle: true },
  hipL: { parent: 'pelvis', side: 'L', handle: true },
  hipR: { parent: 'pelvis', side: 'R', handle: true },
  kneeL: { parent: 'hipL', side: 'L', handle: true },
  kneeR: { parent: 'hipR', side: 'R', handle: true },
  ankleL: { parent: 'kneeL', side: 'L', handle: true },
  ankleR: { parent: 'kneeR', side: 'R', handle: true },
  heelL: { parent: 'ankleL', side: 'L', handle: false },
  heelR: { parent: 'ankleR', side: 'R', handle: false },
  toeL: { parent: 'ankleL', side: 'L', handle: true },
  toeR: { parent: 'ankleR', side: 'R', handle: true },
  neck: { parent: 'pelvis', side: 'C', handle: true },
  shL: { parent: 'neck', side: 'L', handle: true },
  shR: { parent: 'neck', side: 'R', handle: true },
  elL: { parent: 'shL', side: 'L', handle: true },
  elR: { parent: 'shR', side: 'R', handle: true },
  wrL: { parent: 'elL', side: 'L', handle: true },
  wrR: { parent: 'elR', side: 'R', handle: true },
  handL: { parent: 'wrL', side: 'L', handle: true },
  handR: { parent: 'wrR', side: 'R', handle: true },
  head: { parent: 'neck', side: 'C', handle: true },
  nose: { parent: 'head', side: 'C', handle: true },
  eyeL: { parent: 'head', side: 'L', handle: false },
  eyeR: { parent: 'head', side: 'R', handle: false },
  earL: { parent: 'head', side: 'L', handle: false },
  earR: { parent: 'head', side: 'R', handle: false },
};

export const JOINT_NAMES = Object.keys(JOINTS);

const CHILDREN = {};
for (const n of JOINT_NAMES) CHILDREN[n] = [];
for (const n of JOINT_NAMES) if (JOINTS[n].parent) CHILDREN[JOINTS[n].parent].push(n);

export function subtree(name) {
  const out = [];
  const stack = [name];
  while (stack.length) {
    const n = stack.pop();
    out.push(n);
    stack.push(...CHILDREN[n]);
  }
  return out;
}

// Two-bone IK chains: dragging the end joint solves the middle one.
export const IK_CHAINS = {
  wrL: ['shL', 'elL', 'wrL'],
  wrR: ['shR', 'elR', 'wrR'],
  ankleL: ['hipL', 'kneeL', 'ankleL'],
  ankleR: ['hipR', 'kneeR', 'ankleR'],
};

// Dragging the nose turns the whole face around the head center.
const PIVOT_OVERRIDE = {
  nose: { pivot: 'head', moved: () => subtree('head').filter((n) => n !== 'head') },
};

// MediaPipe pose landmark indices.
const MP = {
  nose: 0, eyeL: 2, eyeR: 5, earL: 7, earR: 8, shL: 11, shR: 12, elL: 13, elR: 14,
  wrL: 15, wrR: 16, pinkyL: 17, pinkyR: 18, indexL: 19, indexR: 20,
  hipL: 23, hipR: 24, kneeL: 25, kneeR: 26, ankleL: 27, ankleR: 28,
  heelL: 29, heelR: 30, toeL: 31, toeR: 32,
};

export class Pose {
  constructor() {
    this.p = {};
    for (const n of JOINT_NAMES) this.p[n] = new THREE.Vector3();
  }
  clone() {
    const c = new Pose();
    for (const n of JOINT_NAMES) c.p[n].copy(this.p[n]);
    return c;
  }
  copy(o) {
    for (const n of JOINT_NAMES) this.p[n].copy(o.p[n]);
    return this;
  }
  toArray() {
    return JOINT_NAMES.map((n) => this.p[n].toArray());
  }
  fromArray(a) {
    JOINT_NAMES.forEach((n, i) => this.p[n].fromArray(a[i]));
    return this;
  }
  // Body scale relative to an average adult (shoulder width ~0.36 m).
  scale() {
    return Math.max(0.2, this.p.shL.distanceTo(this.p.shR) / 0.36);
  }
}

// Camera model shared by the editor and the offscreen renders.
export function focalPx(height, fovDeg) {
  return height / 2 / Math.tan(THREE.MathUtils.degToRad(fovDeg) / 2);
}

export function project(v, W, H, f) {
  const z = -v.z;
  return { x: W / 2 + (f * v.x) / z, y: H / 2 - (f * v.y) / z };
}

// Point on the camera ray through pixel (x, y) at view depth z (negative).
function unprojectAtDepth(x, y, z, W, H, f) {
  const d = -z;
  return new THREE.Vector3(((x - W / 2) * d) / f, (-(y - H / 2) * d) / f, z);
}

const avg = (a, b) => a.clone().add(b).multiplyScalar(0.5);

// Build a Pose from MediaPipe results, placed in camera space so that it projects onto the photo.
export function poseFromLandmarks(lm2d, lmWorld, W, H, fovDeg) {
  const f = focalPx(H, fovDeg);
  const rel = (i) => new THREE.Vector3(lmWorld[i].x, -lmWorld[i].y, -lmWorld[i].z);
  const obs = (i) => ({ x: lm2d[i].x * W, y: lm2d[i].y * H, w: lm2d[i].visibility ?? 1 });

  // Solve the translation T so that projected world landmarks match the 2D detections.
  const idx = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
  const R = idx.map(rel);
  const O = idx.map(obs);
  let spread3 = 0, spread2 = 0;
  for (let k = 0; k < idx.length; k++) {
    spread3 += Math.hypot(R[k].x, R[k].y);
    spread2 += Math.hypot(O[k].x - O[3].x, O[k].y - O[3].y);
  }
  const cx = O.reduce((s, o) => s + o.x, 0) / O.length;
  const cy = O.reduce((s, o) => s + o.y, 0) / O.length;
  const T = new THREE.Vector3(0, 0, -Math.max(0.5, (f * spread3) / Math.max(1, spread2)));
  T.x = ((cx - W / 2) * -T.z) / f;
  T.y = (-(cy - H / 2) * -T.z) / f;
  const resid = (t) => {
    let e = 0;
    for (let k = 0; k < idx.length; k++) {
      const p = project(R[k].clone().add(t), W, H, f);
      const w = Math.max(0.05, O[k].w);
      e += w * ((p.x - O[k].x) ** 2 + (p.y - O[k].y) ** 2);
    }
    return e;
  };
  // Simple coordinate descent with shrinking steps (3 unknowns, cheap and robust).
  let step = 0.25 * Math.abs(T.z);
  for (let it = 0; it < 200 && step > 1e-4; it++) {
    let improved = false;
    for (const axis of ['x', 'y', 'z']) {
      for (const s of [step, -step]) {
        const t = T.clone();
        t[axis] += s;
        if (t.z > -0.3) continue;
        if (resid(t) < resid(T)) { T.copy(t); improved = true; }
      }
    }
    if (!improved) step *= 0.5;
  }

  // Place each landmark at its fitted depth, snapping visible ones exactly onto the 2D detection.
  const place = (i) => {
    const v = rel(i).add(T);
    const o = obs(i);
    if (o.w > 0.3) return unprojectAtDepth(o.x, o.y, v.z, W, H, f);
    return v;
  };
  const pose = new Pose();
  const P = pose.p;
  for (const n of ['nose', 'eyeL', 'eyeR', 'earL', 'earR', 'shL', 'shR', 'elL', 'elR', 'wrL', 'wrR',
    'hipL', 'hipR', 'kneeL', 'kneeR', 'ankleL', 'ankleR', 'heelL', 'heelR', 'toeL', 'toeR']) {
    P[n].copy(place(MP[n]));
  }
  P.handL.copy(avg(place(MP.pinkyL), place(MP.indexL)));
  P.handR.copy(avg(place(MP.pinkyR), place(MP.indexR)));
  finishDerived(pose);
  return pose;
}

function finishDerived(pose) {
  const P = pose.p;
  P.pelvis.copy(avg(P.hipL, P.hipR));
  P.neck.copy(avg(P.shL, P.shR));
  // Head center: between the ears, nudged back from the face.
  const earMid = avg(P.earL, P.earR);
  P.head.copy(earMid);
  const s = pose.scale();
  // Ensure the hand point is not collapsed onto the wrist.
  for (const side of ['L', 'R']) {
    const wr = P['wr' + side], hd = P['hand' + side], el = P['el' + side];
    if (wr.distanceTo(hd) < 0.04 * s) {
      const dir = wr.clone().sub(el).normalize();
      hd.copy(wr).addScaledVector(dir, 0.08 * s);
    }
  }
}

// A neutral standing figure for when no person is detected.
export function defaultPose(W, H, fovDeg) {
  const f = focalPx(H, fovDeg);
  const pose = new Pose();
  const P = pose.p;
  const set = (n, x, y, z = 0) => P[n].set(x, y, z);
  set('nose', 0, 0.72, 0.1); set('eyeL', 0.035, 0.75, 0.08); set('eyeR', -0.035, 0.75, 0.08);
  set('earL', 0.075, 0.72, 0); set('earR', -0.075, 0.72, 0);
  set('shL', 0.18, 0.5); set('shR', -0.18, 0.5);
  set('elL', 0.22, 0.22); set('elR', -0.22, 0.22);
  set('wrL', 0.24, -0.02); set('wrR', -0.24, -0.02);
  set('handL', 0.25, -0.1); set('handR', -0.25, -0.1);
  set('hipL', 0.1, 0); set('hipR', -0.1, 0);
  set('kneeL', 0.11, -0.44); set('kneeR', -0.11, -0.44);
  set('ankleL', 0.11, -0.86); set('ankleR', -0.11, -0.86);
  set('heelL', 0.11, -0.9, -0.04); set('heelR', -0.11, -0.9, -0.04);
  set('toeL', 0.12, -0.92, 0.14); set('toeR', -0.12, -0.92, 0.14);
  // Person's left (+x) is on the image right when facing the camera.
  const dist = (1.9 * f) / (0.8 * H);
  for (const n of JOINT_NAMES) P[n].z -= dist;
  finishDerived(pose);
  return pose;
}

// ---------- Editing operations ----------

const _q = new THREE.Quaternion();

function rotateAround(pose, names, pivot, q) {
  for (const n of names) {
    const v = pose.p[n].sub(pivot);
    v.applyQuaternion(q);
    v.add(pivot);
  }
}

// Closest point to `ray` lying on the sphere (c, r); prefers the hemisphere containing `prefer`.
function raySphere(ray, c, r, prefer) {
  const oc = ray.origin.clone().sub(c);
  const b = oc.dot(ray.direction);
  const disc = b * b - (oc.lengthSq() - r * r);
  if (disc >= 0) {
    const sq = Math.sqrt(disc);
    const p1 = ray.at(-b - sq, new THREE.Vector3());
    const p2 = ray.at(-b + sq, new THREE.Vector3());
    return p1.distanceToSquared(prefer) <= p2.distanceToSquared(prefer) ? p1 : p2;
  }
  const closest = ray.at(-b, new THREE.Vector3());
  return c.clone().add(closest.sub(c).setLength(r));
}

// Move `name` toward the cursor ray. mode: 'align' (free move) or 'pose' (keep bone lengths).
export function dragJoint(pose, name, ray, viewDir, { mode, ik }) {
  const P = pose.p;
  const cur = P[name].clone();
  const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(viewDir, cur);
  const onPlane = ray.intersectPlane(plane, new THREE.Vector3());

  if (mode === 'align') {
    if (!onPlane) return;
    const delta = onPlane.clone().sub(cur);
    const moved = name === 'pelvis' ? JOINT_NAMES : name === 'head' ? subtree('head') : [name];
    for (const n of moved) P[n].add(delta);
    if (['hipL', 'hipR'].includes(name)) P.pelvis.copy(avg(P.hipL, P.hipR));
    if (['shL', 'shR'].includes(name)) P.neck.copy(avg(P.shL, P.shR));
    return;
  }

  if (name === 'pelvis') {
    if (!onPlane) return;
    const delta = onPlane.clone().sub(cur);
    for (const n of JOINT_NAMES) P[n].add(delta);
    return;
  }

  if (ik && IK_CHAINS[name] && onPlane) {
    solveIK(pose, IK_CHAINS[name], onPlane);
    return;
  }

  const ov = PIVOT_OVERRIDE[name];
  const pivotName = ov ? ov.pivot : JOINTS[name].parent;
  const moved = ov ? ov.moved() : subtree(name);
  const pivot = P[pivotName].clone();
  const r = cur.distanceTo(pivot);
  if (r < 1e-6) return;
  const target = raySphere(ray, pivot, r, cur);
  _q.setFromUnitVectors(cur.clone().sub(pivot).normalize(), target.sub(pivot).normalize());
  rotateAround(pose, moved, pivot, _q);
}

// Push a joint toward (+) or away from (-) the camera.
export function nudgeDepth(pose, name, amount, viewDir, { mode, ik }) {
  const P = pose.p;
  const target = P[name].clone().addScaledVector(viewDir, -amount);
  if (mode === 'align') {
    const moved = name === 'pelvis' ? JOINT_NAMES : name === 'head' ? subtree('head') : [name];
    const d = target.sub(P[name]);
    for (const n of moved) P[n].add(d);
    return;
  }
  if (name === 'pelvis') {
    const d = target.sub(P[name]);
    for (const n of JOINT_NAMES) P[n].add(d);
    return;
  }
  if (ik && IK_CHAINS[name]) { solveIK(pose, IK_CHAINS[name], target); return; }
  const ov = PIVOT_OVERRIDE[name];
  const pivot = P[ov ? ov.pivot : JOINTS[name].parent].clone();
  const moved = ov ? ov.moved() : subtree(name);
  const r = P[name].distanceTo(pivot);
  const to = target.sub(pivot).setLength(r);
  _q.setFromUnitVectors(P[name].clone().sub(pivot).normalize(), to.normalize());
  rotateAround(pose, moved, pivot, _q);
}

function solveIK(pose, [aN, bN, cN], target) {
  const P = pose.p;
  const A = P[aN].clone(), B = P[bN].clone(), C = P[cN].clone();
  const l1 = A.distanceTo(B), l2 = B.distanceTo(C);
  const toT = target.clone().sub(A);
  let d = toT.length();
  d = THREE.MathUtils.clamp(d, Math.abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4);
  const dir = toT.normalize();
  // Bend direction: keep the current elbow/knee direction.
  let pole = B.clone().sub(A);
  pole.addScaledVector(dir, -pole.dot(dir));
  if (pole.lengthSq() < 1e-10) pole = new THREE.Vector3(0, 0, 1).cross(dir);
  pole.normalize();
  const a = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(0, l1 * l1 - a * a));
  const newB = A.clone().addScaledVector(dir, a).addScaledVector(pole, h);
  const newC = A.clone().addScaledVector(dir, d);

  // Rotate the upper segment (and everything under it) to the new middle joint...
  _q.setFromUnitVectors(B.clone().sub(A).normalize(), newB.clone().sub(A).normalize());
  rotateAround(pose, subtree(bN), A, _q);
  // ...then the lower segment around the new middle joint.
  const curC = P[cN].clone();
  _q.setFromUnitVectors(curC.sub(P[bN]).normalize(), newC.clone().sub(P[bN]).normalize());
  rotateAround(pose, subtree(cN), P[bN].clone(), _q);
}

// Mesh segments of the mannequin and the joints each one depends on (for change detection).
export const SEGMENT_JOINTS = {
  head: ['head', 'nose', 'neck'],
  torso: ['neck', 'shL', 'shR', 'pelvis', 'hipL', 'hipR'],
  upperArmL: ['shL', 'elL'], upperArmR: ['shR', 'elR'],
  forearmL: ['elL', 'wrL'], forearmR: ['elR', 'wrR'],
  handL: ['wrL', 'handL'], handR: ['wrR', 'handR'],
  thighL: ['hipL', 'kneeL'], thighR: ['hipR', 'kneeR'],
  shinL: ['kneeL', 'ankleL'], shinR: ['kneeR', 'ankleR'],
  footL: ['ankleL', 'heelL', 'toeL'], footR: ['ankleR', 'heelR', 'toeR'],
};
export const SEGMENTS = Object.keys(SEGMENT_JOINTS);

export function changedSegments(pose0, pose, eps) {
  const moved = new Set(JOINT_NAMES.filter((n) => pose0.p[n].distanceTo(pose.p[n]) > eps));
  return new Set(SEGMENTS.filter((s) => SEGMENT_JOINTS[s].some((j) => moved.has(j))));
}
