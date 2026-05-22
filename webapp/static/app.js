'use strict';

// ── Elements ──────────────────────────────────────────────────────────────────
const dropZone     = document.getElementById('drop-zone');
const fileInput    = document.getElementById('file-input');
const thumbWrap    = document.getElementById('thumb-wrap');
const thumbImg     = document.getElementById('thumb-img');
const thumbSize    = document.getElementById('thumb-size');
const changeBtn    = document.getElementById('change-btn');
const upscaleBtn   = document.getElementById('upscale-btn');
const emptyState   = document.getElementById('empty-state');
const gridEmpty    = document.getElementById('grid-empty');
const spinner      = document.getElementById('spinner');
const errorBox     = document.getElementById('error-box');
const viewer       = document.getElementById('viewer');
const grid         = document.getElementById('grid');
const healthText   = document.getElementById('health-text');
const resultMeta   = document.getElementById('result-meta');
const actionBtns   = document.getElementById('action-btns');
const mSize        = document.getElementById('m-size');
const mTime        = document.getElementById('m-time');
const mDevice      = document.getElementById('m-device');
const downloadBtn  = document.getElementById('download-btn');
const resetBtn     = document.getElementById('reset-btn');

// Slider
const sliderWrap  = document.getElementById('slider-wrap');
const imgRight    = document.getElementById('img-right');
const imgLeft     = document.getElementById('img-left');
const handle      = document.getElementById('handle');
const cmpRange    = document.getElementById('cmp-range');
const selLeft     = document.getElementById('sel-left');
const selRight    = document.getElementById('sel-right');

// Grid images
const gLR        = document.getElementById('g-lr');
const gBicubic   = document.getElementById('g-bicubic');
const gModelV1   = document.getElementById('g-model-v1');
const gModelV2   = document.getElementById('g-model-v2');
const gModelV3   = document.getElementById('g-model-v3');
const gModelEdsr = document.getElementById('g-model-edsr');
const gModelDrln = document.getElementById('g-model-drln');
const gModelRealesrgan = document.getElementById('g-model-realesrgan');

// ── State ─────────────────────────────────────────────────────────────────────
let currentFile = null;
let imgData     = null;

// Source key -> label + payload field on the API response.
const SOURCES = [
  { key: 'lr',         label: 'LR nearest ×4',       field: 'lr_nearest_png' },
  { key: 'bicubic',    label: 'Bicubic ×4',          field: 'bicubic_png'    },
  { key: 'model_v1',   label: 'Modelo v1 ×4',        field: 'model_v1_png'   },
  { key: 'model_v2',   label: 'Modelo v2 ×4',        field: 'model_v2_png'   },
  { key: 'model_v3',   label: 'Modelo v3 ×4',        field: 'model_v3_png'   },
  { key: 'model_edsr', label: 'EDSR-base ×4 (ref.)', field: 'model_edsr_png' },
  { key: 'model_drln', label: 'DRLN ×4 (ref.)',      field: 'model_drln_png' },
  { key: 'model_realesrgan', label: 'Real-ESRGAN ×4 (ref.)', field: 'model_realesrgan_png' },
];

function sourceByKey(key) { return SOURCES.find(s => s.key === key); }

function imgUrl(key) {
  if (!imgData) return '';
  const src = sourceByKey(key);
  if (!src) return '';
  const b64 = imgData[src.field];
  return b64 ? `data:image/png;base64,${b64}` : '';
}

function availableKeys() {
  return SOURCES.filter(s => imgData && imgData[s.field]).map(s => s.key);
}

// ── Health ────────────────────────────────────────────────────────────────────
(async () => {
  try {
    const d = await (await fetch('/api/health')).json();
    const versions = (d.available || []).join('+') || '—';
    const params   = Object.values(d.params_M || {})[0] ?? '?';
    healthText.textContent = `${d.device} · ${versions} · ${params}M`;
  } catch { healthText.textContent = 'sin conexión'; }
})();

// ── Sidebar toggle ────────────────────────────────────────────────────────────
const sidebar          = document.getElementById('sidebar');
const sidebarEdgeBtn   = document.getElementById('sidebar-toggle-edge');
const sidebarTabBtn    = document.getElementById('sidebar-toggle-tab');

function toggleSidebar() {
  sidebar.classList.toggle('collapsed');
  setTimeout(() => { if (imgData) refreshSlider(); }, 230);
}
sidebarEdgeBtn.addEventListener('click', toggleSidebar);
sidebarTabBtn.addEventListener('click', toggleSidebar);

// ── Tabs ──────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));
    tab.classList.add('active');
    document.getElementById(`tab-${tab.dataset.tab}`).classList.remove('hidden');
    if (tab.dataset.tab === 'compare' && imgData)
      requestAnimationFrame(refreshSlider);
  });
});

// ── Drag & drop ───────────────────────────────────────────────────────────────
['dragenter','dragover'].forEach(e =>
  dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('dragging'); }));
['dragleave','drop'].forEach(e =>
  dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop', ev => { const f = ev.dataTransfer?.files?.[0]; if (f) loadFile(f); });
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });
changeBtn.addEventListener('click', () => fileInput.click());

// ── Load file ─────────────────────────────────────────────────────────────────
function loadFile(file) {
  if (!['image/png','image/jpeg','image/jpg','image/webp'].includes(file.type)) {
    showError(`Tipo no soportado: ${file.type}`); return;
  }
  if (file.size > 10 * 1024 * 1024) { showError('Archivo demasiado grande. Máximo 10 MB.'); return; }

  currentFile = file;
  imgData = null;

  const url = URL.createObjectURL(file);
  thumbImg.src = url;
  thumbImg.onload = () => {
    thumbSize.textContent =
      `${thumbImg.naturalWidth}×${thumbImg.naturalHeight}px · ${(file.size/1024).toFixed(0)}KB`;
  };

  dropZone.classList.add('hidden');
  thumbWrap.classList.remove('hidden');
  upscaleBtn.disabled = false;
  hideError();
  viewer.classList.add('hidden');
  grid.classList.add('hidden');
  emptyState.classList.remove('hidden');
  gridEmpty.classList.remove('hidden');
  resultMeta.classList.add('hidden');
  actionBtns.classList.add('hidden');
}

// ── Upscale ───────────────────────────────────────────────────────────────────
upscaleBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  setLoading(true);
  hideError();

  const form = new FormData();
  form.append('file', currentFile);
  try {
    const r = await fetch('/api/upscale', { method: 'POST', body: form });
    const data = await r.json();
    if (!r.ok) { showError(data.detail || `Error ${r.status}`); return; }
    imgData = data;
    renderResults(data);
  } catch (err) {
    showError(`Error de red: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

// ── Render ────────────────────────────────────────────────────────────────────
function renderResults(data) {
  // Grid images (only set src for sources that came back).
  gLR.src       = imgUrl('lr');
  gBicubic.src  = imgUrl('bicubic');
  if (imgUrl('model_v1'))   gModelV1.src   = imgUrl('model_v1');
  if (imgUrl('model_v2'))   gModelV2.src   = imgUrl('model_v2');
  if (imgUrl('model_v3'))   gModelV3.src   = imgUrl('model_v3');
  if (imgUrl('model_edsr')) gModelEdsr.src = imgUrl('model_edsr');
  if (imgUrl('model_drln')) gModelDrln.src = imgUrl('model_drln');
  if (imgUrl('model_realesrgan')) gModelRealesrgan.src = imgUrl('model_realesrgan');

  // Hide grid columns whose model is missing.
  gModelV1.parentElement.parentElement.classList.toggle('hidden',   !imgUrl('model_v1'));
  gModelV2.parentElement.parentElement.classList.toggle('hidden',   !imgUrl('model_v2'));
  gModelV3.parentElement.parentElement.classList.toggle('hidden',   !imgUrl('model_v3'));
  gModelEdsr.parentElement.parentElement.classList.toggle('hidden', !imgUrl('model_edsr'));
  gModelDrln.parentElement.parentElement.classList.toggle('hidden', !imgUrl('model_drln'));
  gModelRealesrgan.parentElement.parentElement.classList.toggle('hidden', !imgUrl('model_realesrgan'));

  // Meta. inference_ms is a dict {v1: ms, v2: ms}; show both if present.
  mSize.textContent   = `${data.original_size.join('×')} → ${data.output_size.join('×')}`;
  const timings = Object.entries(data.inference_ms || {})
    .map(([v, ms]) => `${v}: ${ms} ms`).join(' · ');
  mTime.textContent   = timings || '—';
  mDevice.textContent = data.device;
  resultMeta.classList.remove('hidden');
  actionBtns.classList.remove('hidden');

  // Build dropdowns (only with sources actually present in the payload).
  populateSelects();

  emptyState.classList.add('hidden');
  gridEmpty.classList.add('hidden');
  viewer.classList.remove('hidden');
  grid.classList.remove('hidden');
  // run after browser lays out the now-visible container
  requestAnimationFrame(refreshSlider);
}

// ── Dropdown selectors ────────────────────────────────────────────────────────
function populateSelects() {
  const keys = availableKeys();
  if (keys.length === 0) return;

  // Sensible defaults: bicubic vs latest model on the right.
  const preferredRight = keys.includes('model_v3') ? 'model_v3'
                        : keys.includes('model_v2') ? 'model_v2'
                        : keys.includes('model_v1') ? 'model_v1'
                        : keys[keys.length - 1];
  const preferredLeft  = keys.includes('bicubic') ? 'bicubic' : keys[0];

  for (const sel of [selLeft, selRight]) {
    sel.innerHTML = '';
    for (const key of keys) {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = sourceByKey(key).label;
      sel.appendChild(opt);
    }
  }
  selLeft.value  = preferredLeft;
  selRight.value = preferredRight;
  applyComparison();
}

function applyComparison() {
  const leftKey  = selLeft.value;
  const rightKey = selRight.value;
  imgLeft.src  = imgUrl(leftKey);
  imgRight.src = imgUrl(rightKey);
  imgRight.onload = refreshSlider;
  if (imgRight.complete && imgRight.naturalWidth) refreshSlider();
}

selLeft.addEventListener('change',  () => { if (imgData) applyComparison(); });
selRight.addEventListener('change', () => { if (imgData) applyComparison(); });

// Prevent slider drag from starting when interacting with the dropdowns.
[selLeft, selRight].forEach(s => {
  s.addEventListener('pointerdown', e => e.stopPropagation());
  s.addEventListener('click',       e => e.stopPropagation());
});

// ── Slider ────────────────────────────────────────────────────────────────────
function fitSlider() {
  if (!imgRight.naturalWidth || !imgRight.naturalHeight) return;
  const cs = getComputedStyle(viewer);
  const aw = viewer.clientWidth  - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  const ah = viewer.clientHeight - parseFloat(cs.paddingTop)  - parseFloat(cs.paddingBottom);
  if (aw <= 0 || ah <= 0) return;
  const r = imgRight.naturalWidth / imgRight.naturalHeight;
  let w, h;
  if (aw / ah > r) { h = ah; w = h * r; }
  else             { w = aw; h = w / r; }
  sliderWrap.style.width  = `${w}px`;
  sliderWrap.style.height = `${h}px`;
}

function syncSlider(val) {
  const p = Number(val) / 100;
  const cw = sliderWrap.clientWidth;
  const splitX = cw * p;
  imgLeft.style.clipPath = `inset(0 ${cw - splitX}px 0 0)`;
  handle.style.left      = `${splitX}px`;
  handle.style.transform = 'translateX(-50%)';
}

function pointerToRange(clientX) {
  const rect = sliderWrap.getBoundingClientRect();
  const p = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  return p * 100;
}

function refreshSlider() {
  fitSlider();
  syncSlider(cmpRange.value);
}

let dragging = false;
sliderWrap.addEventListener('pointerdown', e => {
  if (!imgData) return;
  // Ignore drags that start on the dropdowns.
  if (e.target.closest('.viewer-sel')) return;
  e.preventDefault();
  dragging = true;
  sliderWrap.setPointerCapture(e.pointerId);
  const v = pointerToRange(e.clientX);
  cmpRange.value = v;
  syncSlider(v);
});
sliderWrap.addEventListener('dragstart', e => e.preventDefault());
sliderWrap.addEventListener('pointermove', e => {
  if (!dragging) return;
  const v = pointerToRange(e.clientX);
  cmpRange.value = v;
  syncSlider(v);
});
['pointerup','pointercancel','pointerleave'].forEach(ev =>
  sliderWrap.addEventListener(ev, () => { dragging = false; }));

cmpRange.addEventListener('input', () => syncSlider(cmpRange.value));
window.addEventListener('resize',  refreshSlider);

// ── Download ──────────────────────────────────────────────────────────────────
// Downloads the latest model output preferentially (v2 > v1).
downloadBtn.addEventListener('click', () => {
  if (!imgData) return;
  const preferred = ['model_v3', 'model_v2', 'model_v1', 'bicubic', 'lr']
    .find(k => imgUrl(k));
  if (!preferred) return;
  const a = Object.assign(document.createElement('a'), {
    href:     imgUrl(preferred),
    download: `sr_x4_${preferred}_${currentFile?.name ?? 'output'}.png`,
  });
  a.click();
});

// ── Reset ─────────────────────────────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  currentFile = null;
  imgData = null;
  fileInput.value = '';
  thumbWrap.classList.add('hidden');
  dropZone.classList.remove('hidden');
  viewer.classList.add('hidden');
  grid.classList.add('hidden');
  emptyState.classList.remove('hidden');
  gridEmpty.classList.remove('hidden');
  resultMeta.classList.add('hidden');
  actionBtns.classList.add('hidden');
  upscaleBtn.disabled = true;
  cmpRange.value = 50;
  hideError();
});

// ── Utils ─────────────────────────────────────────────────────────────────────
function setLoading(on) {
  spinner.classList.toggle('hidden', !on);
  if (on) emptyState.classList.add('hidden');
  upscaleBtn.disabled = on;
  upscaleBtn.textContent = on ? 'Procesando…' : 'Super-resolver';
}
function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove('hidden');
  if (!imgData) emptyState.classList.remove('hidden');
}
function hideError()    { errorBox.classList.add('hidden'); }
