const input = document.getElementById('file-input');
const preview = document.getElementById('preview');
const statusEl = document.getElementById('status');
const qrContentEl = document.getElementById('qr-content');
const codeContentEl = document.getElementById('code-content');
const metaQrEl = document.getElementById('meta-qr');
const metaCodeEl = document.getElementById('meta-code');

const openCameraBtn = document.getElementById('open-camera');
const cameraPanel = document.getElementById('camera-panel');
const cameraView = document.getElementById('camera-view');
const captureBtn = document.getElementById('capture-photo');
const closeCameraBtn = document.getElementById('close-camera');

const historyTbody = document.getElementById('history-tbody');
const historyEmpty = document.getElementById('history-empty');
const historyTableWrap = document.getElementById('history-table-wrap');
const refreshHistoryBtn = document.getElementById('refresh-history');

// Бэкенд через nginx-proxy — относительный путь
const API_BASE = '';

let stream = null;

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function renderCode(code, uncertainIndices = []) {
  if (!code) {
    codeContentEl.textContent = '-';
    return;
  }

  const uncertain = new Set((uncertainIndices || []).filter((i) => Number.isInteger(i) && i >= 0));
  let html = '';
  for (let i = 0; i < code.length; i += 1) {
    const ch = escapeHtml(code[i]);
    if (uncertain.has(i)) {
      html += `<span class="char uncertain">${ch}</span>`;
    } else {
      html += `<span class="char">${ch}</span>`;
    }
  }
  codeContentEl.innerHTML = html;
}

function setResult(qr, code, uncertainIndices = []) {
  const qrText = qr || '-';
  const codeText = code || '-';

  qrContentEl.textContent = qrText;
  renderCode(code, uncertainIndices);

  metaQrEl.textContent = qr ? 'Найден' : 'Не найден';
  metaCodeEl.textContent = codeText;
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  cameraView.srcObject = null;
  cameraPanel.hidden = true;
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    statusEl.textContent = 'Ошибка: камера в браузере не поддерживается';
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' } },
      audio: false,
    });
    cameraView.srcObject = stream;
    await cameraView.play();
    cameraPanel.hidden = false;
    statusEl.textContent = 'Камера готова';
  } catch (err) {
    statusEl.textContent = `Ошибка камеры: ${err.message}`;
  }
}

// ===== ИСТОРИЯ =====

function formatDate(isoStr) {
  if (!isoStr) return '-';
  const d = new Date(isoStr);
  return d.toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', year: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function renderHistory(records) {
  historyTbody.innerHTML = '';
  if (!records || records.length === 0) {
    historyEmpty.hidden = false;
    historyTableWrap.hidden = true;
    return;
  }
  historyEmpty.hidden = true;
  historyTableWrap.hidden = false;

  for (const rec of records) {
    const tr = document.createElement('tr');
    const pct = Math.round((rec.confidence || 0) * 100);
    tr.innerHTML = `
      <td class="hist-date">${formatDate(rec.scanned_at)}</td>
      <td class="hist-qr">${escapeHtml(rec.qr_content || '—')}</td>
      <td class="hist-code ${rec.code_below_qr ? 'hist-code-found' : ''}">${escapeHtml(rec.code_below_qr || '—')}</td>
      <td class="hist-wb">${rec.wb_above_qr ? '<span class="chip wb-ok">WB</span>' : '<span class="chip wb-no">—</span>'}</td>
      <td class="hist-conf">${pct}%</td>
    `;
    historyTbody.appendChild(tr);
  }
}

async function loadHistory() {
  try {
    const resp = await fetch(`${API_BASE}/history`);
    if (!resp.ok) return;
    const data = await resp.json();
    renderHistory(data);
  } catch (_) {
    // БД может быть недоступна — игнорируем
  }
}

// ===== АНАЛИЗ =====

async function analyzeBlob(blob, previewUrl) {
  preview.src = previewUrl;
  setResult('-', '-', []);
  statusEl.textContent = 'Обработка...';

  const form = new FormData();
  form.append('file', blob, 'capture.jpg');
  form.append('require_wb', 'true');

  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      body: form,
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(errText || `HTTP ${response.status}`);
    }

    const data = await response.json();
    setResult(data.qr_content, data.code_below_qr, data.debug?.code_uncertain_indices || []);
    statusEl.textContent = data.qr_found ? 'Готово' : 'QR не найден';

    // Обновляем историю после успешного скана
    await loadHistory();
  } catch (err) {
    setResult('-', '-', []);
    statusEl.textContent = `Ошибка: ${err.message}`;
  }
}

input.addEventListener('change', async () => {
  const file = input.files?.[0];
  if (!file) {
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  await analyzeBlob(file, objectUrl);
});

openCameraBtn.addEventListener('click', openCamera);
closeCameraBtn.addEventListener('click', stopCamera);
refreshHistoryBtn.addEventListener('click', loadHistory);

captureBtn.addEventListener('click', async () => {
  if (!stream || !cameraView.videoWidth || !cameraView.videoHeight) {
    statusEl.textContent = 'Камера не готова';
    return;
  }

  const canvas = document.createElement('canvas');
  canvas.width = cameraView.videoWidth;
  canvas.height = cameraView.videoHeight;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(cameraView, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    if (!blob) {
      statusEl.textContent = 'Не удалось создать фото';
      return;
    }

    const previewUrl = URL.createObjectURL(blob);
    stopCamera();
    await analyzeBlob(blob, previewUrl);
  }, 'image/jpeg', 0.92);
});

cameraPanel.hidden = true;
window.addEventListener('beforeunload', stopCamera);

// Загружаем историю при старте
loadHistory();
