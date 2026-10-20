// Before / after / change reference sheet: one image that spells the pose change out for the AI.
import { project, JOINTS, JOINT_NAMES } from './skeleton.js';

const BG = '#111214';
const RED = [255, 70, 70];
const GREEN = [60, 230, 120];
const ARROW = '#ffe14d';

function scaled(src, w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c.getContext('2d').drawImage(src, 0, 0, w, h);
  return c;
}

// Base image dimmed, with red over `removed` and green over `added` (both W*H masks).
export function diffOverlay(base, removed, added, W, H, dim = 0.55) {
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  ctx.drawImage(base, 0, 0, W, H);
  const img = ctx.getImageData(0, 0, W, H);
  const d = img.data;
  for (let i = 0; i < W * H; i++) {
    const k = i * 4;
    let r = d[k] * dim, g = d[k + 1] * dim, b = d[k + 2] * dim;
    const col = added[i] ? GREEN : removed[i] ? RED : null;
    if (col) { r = r * 0.35 + col[0] * 0.65; g = g * 0.35 + col[1] * 0.65; b = b * 0.35 + col[2] * 0.65; }
    d[k] = r; d[k + 1] = g; d[k + 2] = b;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

// Paint a transparent render with a solid tint (keeps its alpha).
export function tint(src, color, alpha) {
  const c = document.createElement('canvas');
  c.width = src.width; c.height = src.height;
  const ctx = c.getContext('2d');
  ctx.drawImage(src, 0, 0);
  ctx.globalCompositeOperation = 'source-atop';
  ctx.fillStyle = color;
  ctx.globalAlpha = alpha;
  ctx.fillRect(0, 0, c.width, c.height);
  return c;
}

export function overlayOf(base, over, alpha = 1) {
  const c = document.createElement('canvas');
  c.width = base.width; c.height = base.height;
  const ctx = c.getContext('2d');
  ctx.drawImage(base, 0, 0);
  ctx.globalAlpha = alpha;
  ctx.drawImage(over, 0, 0, c.width, c.height);
  return c;
}

// Joint moves in image pixels, for arrows. Only handle joints that moved visibly.
export function jointMoves(pose0, pose, W, H, f, minPx) {
  const moves = [];
  for (const n of JOINT_NAMES) {
    if (!JOINTS[n].handle) continue;
    const a = project(pose0.p[n], W, H, f), b = project(pose.p[n], W, H, f);
    if (Math.hypot(b.x - a.x, b.y - a.y) >= minPx) moves.push({ a, b });
  }
  return moves;
}

function drawArrows(ctx, moves, s, lw) {
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const { a, b } of moves) {
    const ax = a.x * s, ay = a.y * s, bx = b.x * s, by = b.y * s;
    const ang = Math.atan2(by - ay, bx - ax);
    const head = Math.max(8, lw * 3.2);
    const path = () => {
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.moveTo(bx, by);
      ctx.lineTo(bx - head * Math.cos(ang - 0.45), by - head * Math.sin(ang - 0.45));
      ctx.moveTo(bx, by);
      ctx.lineTo(bx - head * Math.cos(ang + 0.45), by - head * Math.sin(ang + 0.45));
    };
    ctx.strokeStyle = 'rgba(0,0,0,0.75)';
    ctx.lineWidth = lw + 3;
    path(); ctx.stroke();
    ctx.strokeStyle = ARROW;
    ctx.lineWidth = lw;
    path(); ctx.stroke();
    ctx.fillStyle = ARROW;
    ctx.beginPath();
    ctx.arc(ax, ay, lw * 0.9, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function label(ctx, text, x, y, size) {
  ctx.save();
  ctx.font = `600 ${size}px ui-sans-serif, system-ui, sans-serif`;
  const pad = size * 0.45;
  const w = ctx.measureText(text).width + pad * 2, h = size + pad * 1.4;
  ctx.fillStyle = 'rgba(0,0,0,0.72)';
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, size * 0.3);
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x + pad, y + h / 2 + 1);
  ctx.restore();
}

/**
 * rows: [{ name, before, after, change, arrowsOn: ['after','change'] }]
 * Columns are BEFORE / AFTER / CHANGE (CHANGE optional).
 */
export function buildSheet({ rows, W, H, moves, showChange, showArrows, maxPanel = 900 }) {
  const cols = showChange ? 3 : 2;
  const pw = Math.min(W, maxPanel);
  const s = pw / W;
  const ph = Math.round(H * s);
  const gap = Math.round(pw * 0.015) + 4;
  const titleH = Math.round(pw * 0.055) + 12;
  const sheet = document.createElement('canvas');
  sheet.width = cols * pw + (cols + 1) * gap;
  sheet.height = titleH + rows.length * ph + (rows.length + 1) * gap;
  const ctx = sheet.getContext('2d');
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, sheet.width, sheet.height);

  const fs = Math.max(13, Math.round(pw * 0.026));
  ctx.fillStyle = '#fff';
  ctx.font = `700 ${Math.round(fs * 1.15)}px ui-sans-serif, system-ui, sans-serif`;
  ctx.textBaseline = 'middle';
  const heads = ['BEFORE (current pose)', 'AFTER (target pose)', 'CHANGE (red = remove · green = new · arrows = movement)'];
  for (let c = 0; c < cols; c++) {
    const x = gap + c * (pw + gap);
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, 0, pw, titleH + gap);
    ctx.clip();
    ctx.font = `700 ${c === 2 ? Math.round(fs * 0.85) : Math.round(fs * 1.15)}px ui-sans-serif, system-ui, sans-serif`;
    ctx.fillText(heads[c], x, gap + titleH / 2);
    ctx.restore();
  }

  const lw = Math.max(2, Math.round(pw / 220));
  rows.forEach((row, r) => {
    const y = titleH + gap + r * (ph + gap);
    const cells = [row.before, row.after, row.change].slice(0, cols);
    cells.forEach((img, c) => {
      const x = gap + c * (pw + gap);
      if (img) ctx.drawImage(scaled(img, pw, ph), x, y);
      else { ctx.fillStyle = '#222'; ctx.fillRect(x, y, pw, ph); }
      const col = ['before', 'after', 'change'][c];
      if (showArrows && row.arrowsOn?.includes(col)) {
        ctx.save();
        ctx.beginPath(); ctx.rect(x, y, pw, ph); ctx.clip();
        ctx.translate(x, y);
        drawArrows(ctx, moves, s, lw);
        ctx.restore();
      }
      label(ctx, row.name, x + gap * 0.6, y + gap * 0.6, Math.round(fs * 0.8));
    });
  });
  return sheet;
}
