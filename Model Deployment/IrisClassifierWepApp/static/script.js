/* ─── Constants ─────────────────────────────────────────────────────────────── */
const API_URL = "/predict";

const CLASS_META = {
  "Iris Setosa":     { color: "#a78bfa", emoji: "🌸", bar: "#a78bfa" },
  "Iris Versicolor": { color: "#34d399", emoji: "🌿", bar: "#34d399" },
  "Iris Virginica":  { color: "#fb923c", emoji: "🌺", bar: "#fb923c" },
};

const PRESETS = {
  setosa:     { sl: 5.1, sw: 3.5, pl: 1.4, pw: 0.2 },
  versicolor: { sl: 6.0, sw: 2.7, pl: 4.1, pw: 1.3 },
  virginica:  { sl: 6.7, sw: 3.3, pl: 5.7, pw: 2.1 },
};

/* ─── Slider map ────────────────────────────────────────────────────────────── */
const SLIDERS = [
  { id: "sepal-length", badge: "sl-val", key: "sl" },
  { id: "sepal-width",  badge: "sw-val", key: "sw" },
  { id: "petal-length", badge: "pl-val", key: "pl" },
  { id: "petal-width",  badge: "pw-val", key: "pw" },
];

/* ─── DOM ───────────────────────────────────────────────────────────────────── */
const predictBtn     = document.getElementById("predict-btn");
const predictBtnText = document.getElementById("predict-btn-text");
const predictSpinner = document.getElementById("predict-spinner");
const idleState      = document.getElementById("idle-state");
const resultState    = document.getElementById("result-state");
const errorStateEl   = document.getElementById("error-state");
const errorStateMsgEl= document.getElementById("error-state-msg");
const toast          = document.getElementById("toast");

/* Result elements */
const resultIconWrap = document.getElementById("result-icon-wrap");
const resultEmoji    = document.getElementById("result-emoji");
const resultName     = document.getElementById("result-name");
const confPct        = document.getElementById("conf-pct");
const confFill       = document.getElementById("conf-fill");
const resultDesc     = document.getElementById("result-desc");
const probBars       = document.getElementById("prob-bars");
const measureChips   = document.getElementById("measure-chips");

/* ─── Panel state manager ───────────────────────────────────────────────────── */
// 'idle' | 'result' | 'error'
function showPanel(state, errorMsg = '') {
  idleState.hidden   = state !== 'idle';
  resultState.hidden = state !== 'result';
  errorStateEl.hidden= state !== 'error';
  if (state === 'error' && errorMsg) {
    errorStateMsgEl.textContent = errorMsg;
  }
}

// "Tekrar Dene" butonu idle'a döndürür
document.getElementById("error-retry-btn").addEventListener("click", () => showPanel('idle'));

/* ─── Toast ─────────────────────────────────────────────────────────────────── */
let toastTimer = null;
function showToast(msg, type = "error") {
  toast.textContent = msg;
  toast.className = `toast show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.className = "toast"; }, 4000);
}

/* ─── Slider live update ─────────────────────────────────────────────────────── */
SLIDERS.forEach(({ id, badge }) => {
  const input = document.getElementById(id);
  const badgeEl = document.getElementById(badge);
  if (!input || !badgeEl) return;

  const update = () => {
    badgeEl.textContent = `${parseFloat(input.value).toFixed(1)} cm`;
    // Highlight active badge
    badgeEl.style.borderColor = "rgba(124,106,247,.4)";
    badgeEl.style.background  = "rgba(124,106,247,.08)";
    clearTimeout(input._timer);
    input._timer = setTimeout(() => {
      badgeEl.style.borderColor = "";
      badgeEl.style.background  = "";
    }, 800);
  };

  input.addEventListener("input", update);
  update(); // init
});

/* ─── Read slider values ─────────────────────────────────────────────────────── */
function getValues() {
  return SLIDERS.reduce((acc, { id, key }) => {
    acc[key] = parseFloat(document.getElementById(id).value);
    return acc;
  }, {});
}

/* ─── Set slider values ──────────────────────────────────────────────────────── */
function setValues({ sl, sw, pl, pw }) {
  const map = { "sepal-length": sl, "sepal-width": sw, "petal-length": pl, "petal-width": pw };
  Object.entries(map).forEach(([id, val]) => {
    const el = document.getElementById(id);
    if (el) {
      el.value = val;
      el.dispatchEvent(new Event("input"));
    }
  });
}

/* ─── Presets ────────────────────────────────────────────────────────────────── */
document.querySelectorAll(".preset-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const sl = parseFloat(btn.dataset.sl);
    const sw = parseFloat(btn.dataset.sw);
    const pl = parseFloat(btn.dataset.pl);
    const pw = parseFloat(btn.dataset.pw);
    setValues({ sl, sw, pl, pw });
  });
});

/* ─── Loading state ──────────────────────────────────────────────────────────── */
function setLoading(on) {
  predictBtn.disabled     = on;
  predictBtnText.hidden   = on;
  predictSpinner.hidden   = !on;
}

/* ─── Render Result ──────────────────────────────────────────────────────────── */
function renderResult(data, vals) {
  const meta = CLASS_META[data.predicted_class] || CLASS_META["Iris Setosa"];

  /* Icon + name */
  resultEmoji.textContent = meta.emoji;
  resultName.textContent  = data.predicted_class;
  resultName.style.color  = meta.color;

  resultIconWrap.style.background   = `${meta.color}18`;
  resultIconWrap.style.borderColor  = `${meta.color}40`;

  /* Confidence bar */
  confPct.textContent = `${data.confidence}%`;
  confFill.style.width = "0%";
  requestAnimationFrame(() => requestAnimationFrame(() => {
    confFill.style.background = `linear-gradient(90deg, ${meta.color}, ${meta.bar}aa)`;
    confFill.style.width = `${data.confidence}%`;
  }));

  /* Description */
  resultDesc.textContent = data.description;
  resultDesc.style.borderLeftColor = meta.color;

  /* Prob bars */
  probBars.innerHTML = "";
  Object.entries(data.probabilities).forEach(([cls, pct]) => {
    const m = CLASS_META[cls] || { color: "#a78bfa" };
    const item = document.createElement("div");
    item.className = "prob-item";
    item.innerHTML = `
      <div class="prob-meta">
        <span class="prob-cls">${cls}</span>
        <span class="prob-pct">${pct}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="background:${m.color};"></div>
      </div>
    `;
    probBars.appendChild(item);
  });
  /* Animate fills */
  requestAnimationFrame(() => requestAnimationFrame(() => {
    const fills = probBars.querySelectorAll(".prob-fill");
    Object.values(data.probabilities).forEach((pct, i) => {
      if (fills[i]) fills[i].style.width = `${pct}%`;
    });
  }));

  /* Measure chips */
  const labels = [
    ["Sepal Uz.", vals.sl],
    ["Sepal Gen.", vals.sw],
    ["Petal Uz.", vals.pl],
    ["Petal Gen.", vals.pw],
  ];
  measureChips.innerHTML = labels.map(([lbl, v]) => `
    <div class="m-chip">
      <span class="m-chip-lbl">${lbl}</span>
      <span class="m-chip-val">${parseFloat(v).toFixed(1)}<small style="font-size:.65rem;opacity:.6;font-weight:400"> cm</small></span>
    </div>
  `).join("");

  /* Show result */
  showPanel('result');
}

/* ─── Predict ────────────────────────────────────────────────────────────────── */
predictBtn.addEventListener("click", async () => {
  const vals = getValues();

  setLoading(true);
  try {
    const resp = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sepal_length: vals.sl,
        sepal_width:  vals.sw,
        petal_length: vals.pl,
        petal_width:  vals.pw,
      }),
    });

    if (!resp.ok) {
      let detail = "Sunucu hatası";
      try {
        const json = await resp.json();
        detail = json?.detail || `HTTP ${resp.status}`;
      } catch (_) {}
      throw new Error(detail);
    }

    const data = await resp.json();
    renderResult(data, vals);
    showToast(`${data.predicted_class} — %${data.confidence} güven`, "success");

  } catch (err) {
    showPanel('error', err.message || 'Sunucuya bağlanılamadı.');
    showToast(err.message || 'Bağlantı hatası', 'error');
  } finally {
    setLoading(false);
  }
});
