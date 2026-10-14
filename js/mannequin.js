// Procedural mannequin built from a Pose: lofted torso + tapered limbs + spheres.
import * as THREE from 'three';
import { SEGMENTS } from './skeleton.js';

const UP = new THREE.Vector3(0, 1, 0);
const RING = 20;
// Torso cross-sections from shoulders (t=0) to hips (t=1): [t, halfWidthFactor, halfDepth (m)].
const TORSO_PROFILE = [
  [-0.06, 0.62, 0.07],
  [0.0, 0.92, 0.095],
  [0.18, 0.86, 0.11],
  [0.45, 0.66, 0.095],
  [0.7, 0.74, 0.1],
  [1.0, 1.0, 0.105],
  [1.1, 0.9, 0.085],
];

// Limb definitions: segment, joint a, joint b, radius a, radius b (meters at scale 1), extend past b.
const LIMBS = [
  ['head', 'neck', 'head', 0.05, 0.05, 0],
  ['upperArmL', 'shL', 'elL', 0.05, 0.041, 0], ['upperArmR', 'shR', 'elR', 0.05, 0.041, 0],
  ['forearmL', 'elL', 'wrL', 0.04, 0.03, 0], ['forearmR', 'elR', 'wrR', 0.04, 0.03, 0],
  ['handL', 'wrL', 'handL', 0.03, 0.034, 0.45], ['handR', 'wrR', 'handR', 0.03, 0.034, 0.45],
  ['thighL', 'hipL', 'kneeL', 0.085, 0.058, 0], ['thighR', 'hipR', 'kneeR', 0.085, 0.058, 0],
  ['shinL', 'kneeL', 'ankleL', 0.055, 0.04, 0], ['shinR', 'kneeR', 'ankleR', 0.055, 0.04, 0],
  ['footL', 'heelL', 'toeL', 0.035, 0.03, 0.1], ['footR', 'heelR', 'toeR', 0.035, 0.03, 0.1],
  ['footL', 'ankleL', 'heelL', 0.04, 0.035, 0], ['footR', 'ankleR', 'heelR', 0.04, 0.035, 0],
];
// Spheres: segment, joint, radius.
const BALLS = [
  ['head', 'head', 0.088], ['head', 'nose', 0.022],
  ['upperArmL', 'shL', 0.058], ['upperArmR', 'shR', 0.058],
  ['upperArmL', 'elL', 0.041], ['upperArmR', 'elR', 0.041],
  ['forearmL', 'wrL', 0.03], ['forearmR', 'wrR', 0.03],
  ['thighL', 'hipL', 0.085], ['thighR', 'hipR', 0.085],
  ['thighL', 'kneeL', 0.058], ['thighR', 'kneeR', 0.058],
  ['shinL', 'ankleL', 0.04], ['shinR', 'ankleR', 0.04],
  ['footL', 'heelL', 0.035], ['footR', 'heelR', 0.035],
];

export const SEGMENT_ID = Object.fromEntries(SEGMENTS.map((s, i) => [s, i + 1]));

const sphereGeo = new THREE.SphereGeometry(1, 24, 16);
const cylGeo = new THREE.CylinderGeometry(1, 1, 1, 20, 1, false); // radii scaled per-vertex below

export class Mannequin {
  constructor(material) {
    this.group = new THREE.Group();
    this.material = material;
    this.parts = []; // { mesh, segment, update(pose, s) }

    // Torso: lofted mesh rebuilt every update.
    const nRows = TORSO_PROFILE.length;
    const torsoGeo = new THREE.BufferGeometry();
    const pos = new Float32Array((nRows * RING + 2) * 3);
    torsoGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const index = [];
    for (let r = 0; r < nRows - 1; r++) {
      for (let k = 0; k < RING; k++) {
        const a = r * RING + k, b = r * RING + ((k + 1) % RING);
        const c = a + RING, d = b + RING;
        index.push(a, b, c, b, d, c);
      }
    }
    const top = nRows * RING, bot = top + 1;
    for (let k = 0; k < RING; k++) {
      index.push(top, (k + 1) % RING, k);
      const o = (nRows - 1) * RING;
      index.push(bot, o + k, o + ((k + 1) % RING));
    }
    torsoGeo.setIndex(index);
    const torso = new THREE.Mesh(torsoGeo, material);
    this._addPart(torso, 'torso', (P, s) => this._updateTorso(torsoGeo, P, s));

    for (const [seg, a, b, ra, rb, ext] of LIMBS) {
      const geo = cylGeo.clone();
      const mesh = new THREE.Mesh(geo, material);
      const base = geo.attributes.position.array.slice();
      this._addPart(mesh, seg, (P, s) => {
        const A = P[a], B = P[b].clone();
        if (ext) B.add(P[b].clone().sub(A).multiplyScalar(ext));
        const len = Math.max(1e-4, A.distanceTo(B));
        // Taper: cylinder y in [-0.5, 0.5] maps radius from ra (bottom, at A) to rb (top, at B).
        const arr = geo.attributes.position.array;
        for (let i = 0; i < arr.length; i += 3) {
          const t = base[i + 1] + 0.5;
          const r = (ra + (rb - ra) * t) * s;
          arr[i] = base[i] * r;
          arr[i + 1] = base[i + 1] * len;
          arr[i + 2] = base[i + 2] * r;
        }
        geo.attributes.position.needsUpdate = true;
        geo.computeVertexNormals();
        geo.computeBoundingSphere();
        mesh.position.copy(A).add(B).multiplyScalar(0.5);
        mesh.quaternion.setFromUnitVectors(UP, B.clone().sub(A).normalize());
      });
    }
    for (const [seg, j, r] of BALLS) {
      const mesh = new THREE.Mesh(sphereGeo, material);
      this._addPart(mesh, seg, (P, s) => {
        mesh.position.copy(P[j]);
        mesh.scale.setScalar(r * s);
        if (j === 'head') {
          // Slightly egg-shaped head, oriented along neck->head.
          mesh.scale.set(r * s * 0.92, r * s * 1.12, r * s * 0.98);
          mesh.quaternion.setFromUnitVectors(UP, P.head.clone().sub(P.neck).normalize());
        }
      });
    }
  }

  _addPart(mesh, segment, update) {
    mesh.userData.segment = segment;
    mesh.frustumCulled = false;
    this.group.add(mesh);
    this.parts.push({ mesh, segment, update });
  }

  _updateTorso(geo, P, s) {
    const arr = geo.attributes.position.array;
    const shW = P.shR.clone().sub(P.shL);
    const hipW = P.hipR.clone().sub(P.hipL);
    const shHalf = shW.length() / 2, hipHalf = hipW.length() / 2 + 0.045 * s;
    const top = P.neck, bottom = P.pelvis;
    const spine = top.clone().sub(bottom);
    const n = TORSO_PROFILE.length;
    for (let r = 0; r < n; r++) {
      const [t, wf, hd] = TORSO_PROFILE[r];
      const center = top.clone().lerp(bottom, t);
      const tt = THREE.MathUtils.clamp(t, 0, 1);
      const lateral = shW.clone().normalize().lerp(hipW.clone().normalize(), tt).normalize();
      const fwd = new THREE.Vector3().crossVectors(lateral, spine).normalize();
      // Keep lateral orthogonal to the spine so the cross-section is a clean ellipse.
      const lat = new THREE.Vector3().crossVectors(spine, fwd).normalize();
      if (lat.dot(lateral) < 0) lat.negate();
      const half = (shHalf + (hipHalf - shHalf) * tt) * wf;
      const depth = hd * s;
      for (let k = 0; k < RING; k++) {
        const ang = (k / RING) * Math.PI * 2;
        const v = center.clone()
          .addScaledVector(lat, Math.cos(ang) * half)
          .addScaledVector(fwd, Math.sin(ang) * depth);
        v.toArray(arr, (r * RING + k) * 3);
      }
    }
    const topCap = top.clone().lerp(bottom, TORSO_PROFILE[0][0]);
    const botCap = top.clone().lerp(bottom, TORSO_PROFILE[n - 1][0]);
    topCap.toArray(arr, n * RING * 3);
    botCap.toArray(arr, (n * RING + 1) * 3);
    geo.attributes.position.needsUpdate = true;
    geo.computeVertexNormals();
    geo.computeBoundingSphere();
  }

  update(pose) {
    const s = pose.scale();
    for (const part of this.parts) part.update(pose.p, s);
  }

  // Show only some segments (null = all).
  setVisibleSegments(set) {
    for (const { mesh, segment } of this.parts) mesh.visible = !set || set.has(segment);
  }

  setMaterial(mat) {
    for (const { mesh } of this.parts) mesh.material = mat;
  }
}

// Material that writes view-space distance (R) and segment id (G) into a float target.
export function makeIdDepthMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: { segId: { value: 0 } },
    vertexShader: /* glsl */`
      varying float vViewZ;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vViewZ = -mv.z;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */`
      uniform float segId;
      varying float vViewZ;
      void main() { gl_FragColor = vec4(vViewZ, segId, 0.0, 1.0); }`,
  });
}

export function makeClayMaterial(opts = {}) {
  return new THREE.MeshStandardMaterial({ color: 0xbdbab4, roughness: 0.75, metalness: 0, ...opts });
}

export function addStudioLights(scene) {
  scene.add(new THREE.HemisphereLight(0xffffff, 0x55504a, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(-1.5, 2.5, 2);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xffffff, 0.5);
  rim.position.set(2, 1, -1.5);
  scene.add(rim);
}

// Offscreen renderer helpers. The camera matches the photo exactly.
export class Offscreen {
  constructor(renderer) {
    this.renderer = renderer;
    this.scene = new THREE.Scene();
    this.mannequin = null;
    this.idMats = new Map();
  }

  // Returns { z: Float32Array (0 = empty), id: Uint8Array } in image row order (top row first).
  renderIdDepth(mannequin, pose, camera, W, H) {
    const r = this.renderer;
    const rt = new THREE.WebGLRenderTarget(W, H, { type: THREE.FloatType, depthBuffer: true });
    const scene = new THREE.Scene();
    mannequin.update(pose);
    const saved = mannequin.parts.map((p) => p.mesh.material);
    for (const part of mannequin.parts) {
      let m = this.idMats.get(part.segment);
      if (!m) { m = makeIdDepthMaterial(); m.uniforms.segId.value = SEGMENT_ID[part.segment]; this.idMats.set(part.segment, m); }
      part.mesh.material = m;
    }
    const parent = mannequin.group.parent;
    scene.add(mannequin.group);
    const prevTarget = r.getRenderTarget();
    const prevClear = r.getClearColor(new THREE.Color()).clone();
    const prevAlpha = r.getClearAlpha();
    r.setRenderTarget(rt);
    r.setClearColor(0x000000, 0);
    r.clear();
    r.render(scene, camera);
    const buf = new Float32Array(W * H * 4);
    r.readRenderTargetPixels(rt, 0, 0, W, H, buf);
    r.setRenderTarget(prevTarget);
    r.setClearColor(prevClear, prevAlpha);
    rt.dispose();
    mannequin.parts.forEach((p, i) => { p.mesh.material = saved[i]; });
    if (parent) parent.add(mannequin.group);

    const z = new Float32Array(W * H);
    const id = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) {
      const src = (H - 1 - y) * W; // GL origin is bottom-left
      for (let x = 0; x < W; x++) {
        const i = (src + x) * 4;
        z[y * W + x] = buf[i];
        id[y * W + x] = Math.round(buf[i + 1]);
      }
    }
    return { z, id };
  }

  // Shaded render to a 2D canvas. `segments` limits what is drawn; bg null = transparent.
  renderShaded(mannequin, pose, camera, W, H, { segments = null, bg = null } = {}) {
    const r = this.renderer;
    const rt = new THREE.WebGLRenderTarget(W, H, { samples: 4 });
    rt.texture.colorSpace = THREE.SRGBColorSpace;
    const scene = new THREE.Scene();
    addStudioLights(scene);
    mannequin.update(pose);
    mannequin.setVisibleSegments(segments);
    const parent = mannequin.group.parent;
    scene.add(mannequin.group);
    const prevTarget = r.getRenderTarget();
    const prevClear = r.getClearColor(new THREE.Color()).clone();
    const prevAlpha = r.getClearAlpha();
    r.setRenderTarget(rt);
    if (bg) r.setClearColor(bg, 1); else r.setClearColor(0x000000, 0);
    r.clear();
    r.render(scene, camera);
    const buf = new Uint8Array(W * H * 4);
    r.readRenderTargetPixels(rt, 0, 0, W, H, buf);
    r.setRenderTarget(prevTarget);
    r.setClearColor(prevClear, prevAlpha);
    rt.dispose();
    mannequin.setVisibleSegments(null);
    if (parent) parent.add(mannequin.group);

    const canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(W, H);
    for (let y = 0; y < H; y++) {
      img.data.set(buf.subarray((H - 1 - y) * W * 4, (H - y) * W * 4), y * W * 4);
    }
    ctx.putImageData(img, 0, 0);
    return canvas;
  }
}
