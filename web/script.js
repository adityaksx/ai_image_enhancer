// ─── STATE ────────────────────────────────────────────────────────────────────
let selectedFiles = [];   // single images
let folderFiles   = [];   // folder images

// ─── FILE INPUTS ──────────────────────────────────────────────────────────────
document.getElementById('single-upload').addEventListener('change', function (e) {
  selectedFiles = Array.from(e.target.files).filter(f => f.type.startsWith('image/'));
  folderFiles   = [];
  renderFileChips(selectedFiles, 'image');
  e.target.value = '';
});

document.getElementById('folder-upload').addEventListener('change', function (e) {
  folderFiles   = Array.from(e.target.files).filter(f => f.type.startsWith('image/'));
  selectedFiles = [];
  renderFileChips(folderFiles, 'folder');
  e.target.value = '';
});

function renderFileChips(files, mode) {
  const container = document.getElementById('file-chips');
  container.innerHTML = '';
  if (!files.length) return;

  const summary = document.createElement('div');
  summary.className = 'chip-image';
  summary.innerHTML = `
    <span class="chip-icon">${mode === 'folder' ? '📁' : '🖼'}</span>
    <span class="chip-label">${files.length} file${files.length > 1 ? 's' : ''} selected</span>
    <button class="chip-remove" onclick="clearFiles()">✕</button>
  `;
  container.appendChild(summary);
}

function clearFiles() {
  selectedFiles = [];
  folderFiles   = [];
  document.getElementById('file-chips').innerHTML = '';
}

// ─── TOGGLE CHIPS — show/hide sliders ────────────────────────────────────────
document.querySelectorAll('.toggle-chip').forEach(label => {
  label.addEventListener('click', function () {
    const cb  = this.querySelector('input[type=checkbox]');
    cb.checked = !cb.checked;
    this.classList.toggle('active', cb.checked);
    const val = cb.value;

    // show slider only for these options
    const sliderMap = ['lighting','color','denoise','sharpen','upscale'];
    if (sliderMap.includes(val)) {
      const slider = document.getElementById(`slider-${val}`);
      if (slider) slider.style.display = cb.checked ? 'flex' : 'none';
    }

    // If auto turned OFF, all sliders remain user-controlled
  });
});

// ─── GET SELECTED OPTIONS ─────────────────────────────────────────────────────
function getOptions() {
  return Array.from(document.querySelectorAll('.toggle-chip input:checked'))
    .map(cb => cb.value);
}

// ─── GET SLIDER PARAMS ────────────────────────────────────────────────────────
function getParams() {
  return {
    lighting_clip:   parseFloat(document.getElementById('range-lighting').value),
    color_sat:       parseFloat(document.getElementById('range-color').value),
    denoise_level:   parseInt(document.getElementById('range-denoise').value),
    sharpen_strength:parseFloat(document.getElementById('range-sharpen').value),
    upscale_factor:  parseInt(document.getElementById('range-upscale').value),
  };
}

// ─── PROCESS ──────────────────────────────────────────────────────────────────
async function processFiles() {
  const files   = selectedFiles.length ? selectedFiles : folderFiles;
  const options = getOptions();

  if (!files.length) {
    setStatus('⚠️ Select an image or folder first.', 'warn');
    return;
  }
  if (!options.length) {
    setStatus('⚠️ Select at least one enhancement.', 'warn');
    return;
  }

  const btn = document.getElementById('process-btn');
  btn.disabled = true;
  setStatus('');
  showProgress(0);

  // Hide welcome screen
  const welcome = document.getElementById('welcome-screen');
  if (welcome) welcome.style.display = 'none';

  const params  = getParams();
  const total   = files.length;
  let   success = 0;

  for (let i = 0; i < total; i++) {
    const file = files[i];
    setStatus(`Processing ${i + 1}/${total}: ${file.name}`);
    showProgress(Math.round(((i) / total) * 100));

    const form = new FormData();
    form.append('file', file, file.name);
    form.append('options', options.join(','));
    // Append slider params as JSON string
    form.append('params', JSON.stringify(params));

    try {
      const res  = await fetch('/api/process-image', { method: 'POST', body: form });
      const data = await res.json();

      if (data.output) {
        success++;
        addResultCard(file.name, data.output);
        addToRecent(file.name);
      }
    } catch (err) {
      console.error(`Error on ${file.name}:`, err);
    }
  }

  showProgress(100);
  setTimeout(() => hideProgress(), 800);
  setStatus(`✅ Done — ${success}/${total} processed`, 'success');
  btn.disabled = false;
}

// ─── RESULTS ──────────────────────────────────────────────────────────────────
function addResultCard(filename, outputPath) {
  const grid = document.getElementById('results-grid');

  const card = document.createElement('div');
  card.className = 'result-card';
  const outFile = outputPath.split(/[\\/]/).pop();

  card.innerHTML = `
    <div class="result-img-wrap">
      <img src="/output-img/${outFile}" loading="lazy" alt="${filename}"
           onclick="openFullscreen(this.src)">
    </div>
    <div class="result-info">
      <span class="result-name">${outFile}</span>
      <a class="result-dl" href="/output-img/${outFile}" download="${outFile}">⬇ Download</a>
    </div>
  `;
  grid.prepend(card);
}

function openFullscreen(src) {
  const overlay = document.createElement('div');
  overlay.className = 'fullscreen-overlay';
  overlay.innerHTML = `<img src="${src}"><button onclick="this.parentElement.remove()">✕</button>`;
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
  document.body.appendChild(overlay);
}

// ─── RECENT LIST ──────────────────────────────────────────────────────────────
function addToRecent(name) {
  const list = document.getElementById('recent-list');
  const li   = document.createElement('li');
  li.textContent = name.length > 28 ? name.slice(0, 26) + '…' : name;
  list.prepend(li);
}

// ─── CLEAR ────────────────────────────────────────────────────────────────────
function clearResults() {
  clearFiles();
  setStatus('');
}

async function clearOutput() {
  await fetch('/api/results/clear', { method: 'DELETE' });
  document.getElementById('results-grid').innerHTML = '';
  document.getElementById('recent-list').innerHTML  = '';
  setStatus('Output cleared.', 'warn');
}

// ─── LOAD EXISTING RESULTS ON PAGE LOAD ───────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  const res  = await fetch('/api/results');
  const data = await res.json();
  if (data.files && data.files.length > 0) {
    document.getElementById('welcome-screen').style.display = 'none';
    data.files.forEach(f => addResultCard(f, `output/${f}`));
  }
});

// ─── PROGRESS ─────────────────────────────────────────────────────────────────
function showProgress(pct) {
  document.getElementById('progress-wrap').classList.remove('hidden');
  document.getElementById('progress-bar').style.width  = pct + '%';
  document.getElementById('progress-label').textContent = pct < 100 ? `${pct}%` : 'Done!';
}

function hideProgress() {
  document.getElementById('progress-wrap').classList.add('hidden');
  document.getElementById('progress-bar').style.width = '0%';
}

function setStatus(msg, type = '') {
  const el = document.getElementById('status-text');
  el.textContent  = msg;
  el.className    = `status-text ${type}`;
}
