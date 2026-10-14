// OpenPose rendering and file downloads.
import { project } from './skeleton.js';

// OpenPose COCO-18 keypoint order and colours (ControlNet convention).
const OP_ORDER = ['nose', 'neck', 'shR', 'elR', 'wrR', 'shL', 'elL', 'wrL', 'hipR', 'kneeR', 'ankleR',
  'hipL', 'kneeL', 'ankleL', 'eyeR', 'eyeL', 'earR', 'earL'];
const OP_LIMBS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
  [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]];
const OP_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
  [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
  [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]];

export function renderOpenPose(pose, W, H, f) {
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);
  const pts = OP_ORDER.map((n) => project(pose.p[n], W, H, f));
  const stick = Math.max(3, Math.round(Math.max(W, H) / 256));
  ctx.globalAlpha = 0.6;
  OP_LIMBS.forEach(([a, b], i) => {
    const A = pts[a], B = pts[b];
    const len = Math.hypot(B.x - A.x, B.y - A.y);
    const ang = Math.atan2(B.y - A.y, B.x - A.x);
    ctx.fillStyle = `rgb(${OP_COLORS[i].join(',')})`;
    ctx.beginPath();
    ctx.ellipse((A.x + B.x) / 2, (A.y + B.y) / 2, len / 2, stick, ang, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
  pts.forEach((p, i) => {
    ctx.fillStyle = `rgb(${OP_COLORS[i].join(',')})`;
    ctx.beginPath();
    ctx.arc(p.x, p.y, stick, 0, Math.PI * 2);
    ctx.fill();
  });
  return c;
}

export function canvasToBlob(canvas) {
  return new Promise((res) => canvas.toBlob(res, 'image/png'));
}

export function downloadBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

export async function downloadZip(files, name) {
  const zip = new window.JSZip();
  for (const [fname, blob] of files) zip.file(fname, blob);
  downloadBlob(await zip.generateAsync({ type: 'blob' }), name);
}
