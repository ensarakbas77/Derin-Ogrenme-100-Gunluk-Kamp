/**
 * script.js — Fetal Sağlık Sınıflandırıcı
 * Yeni sidebar layout ile uyumlu frontend mantığı.
 */

"use strict";

// ── DOM ──────────────────────────────────────────────────
const form          = document.getElementById("predict-form");
const btnPredict    = document.getElementById("btn-predict");
const btnLabel      = document.getElementById("btn-label");
const btnReset      = document.getElementById("btn-reset");
const spinner       = document.getElementById("spinner");
const resultSection = document.getElementById("result-section");
const resultCard    = document.getElementById("result-card");
const resultEmoji   = document.getElementById("result-emoji");
const resultLabel   = document.getElementById("result-label");
const resultMeta    = document.getElementById("result-meta");
const confPct       = document.getElementById("conf-pct");
const probList      = document.getElementById("prob-list");
const errorToast    = document.getElementById("error-toast");
const errorMsg      = document.getElementById("error-msg");

// ── Metadata ──────────────────────────────────────────────
const META = {
  Normal:       { emoji: "💚", label: "Normal",    cls: "normal",       fill: "f-normal" },
  Suspect:      { emoji: "⚠️",  label: "Şüpheli",   cls: "suspect",      fill: "f-suspect" },
  Pathological: { emoji: "🔴", label: "Patolojik", cls: "pathological", fill: "f-pathological" },
};

// ── Helpers ───────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorToast.style.display = "flex";
  resultSection.style.display = "none";
}
function hideError() { errorToast.style.display = "none"; }

function setLoading(on) {
  btnPredict.disabled    = on;
  spinner.style.display  = on ? "block" : "none";
  btnLabel.textContent   = on ? "Analiz ediliyor…" : "Sağlık Durumunu Tahmin Et";
}

function validateInputs(inputs) {
  let ok = true;
  inputs.forEach(el => {
    el.classList.remove("error");
    if (el.value.trim() === "" || isNaN(parseFloat(el.value))) {
      el.classList.add("error");
      ok = false;
    }
  });
  return ok;
}

function animateBar(fillEl, pct) {
  fillEl.style.width = "0%";
  requestAnimationFrame(() => requestAnimationFrame(() => {
    fillEl.style.width = pct.toFixed(2) + "%";
  }));
}

// ── Render result ─────────────────────────────────────────
function renderResult(data) {
  const { predicted_label, predicted_class_index, probabilities_percent } = data;
  const meta = META[predicted_label] || META["Normal"];

  // Card colour class
  resultCard.className = `result-card ${meta.cls}`;

  // Banner
  resultEmoji.textContent = meta.emoji;
  resultLabel.textContent = meta.label;
  resultMeta.textContent  = `Sınıf indeksi: ${predicted_class_index}  ·  11 KTG özelliği kullanıldı`;

  // Confidence chip (winning probability)
  const winPct = probabilities_percent[predicted_label] ?? 0;
  confPct.textContent = winPct.toFixed(1) + "%";

  // Probability rows
  probList.innerHTML = "";
  for (const [eng, pct] of Object.entries(probabilities_percent)) {
    const m = META[eng];
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div class="prob-name">${m ? m.label : eng}</div>
      <div class="prob-track">
        <div class="prob-fill ${m ? m.fill : "f-normal"}" style="width:0%"></div>
      </div>
      <div class="prob-pct">${pct.toFixed(2)}%</div>
    `;
    probList.appendChild(row);
    animateBar(row.querySelector(".prob-fill"), pct);
  }

  resultSection.style.display = "block";
  resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Form submit ───────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();

  const inputs = Array.from(form.querySelectorAll("input[type='number']"));
  if (!validateInputs(inputs)) {
    showError("Lütfen tüm alanları geçerli sayısal değerlerle doldurun.");
    return;
  }

  const payload = {};
  inputs.forEach(el => { payload[el.name] = parseFloat(el.value); });

  setLoading(true);
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showError(err.detail || `Sunucu hatası (${res.status})`);
      return;
    }

    renderResult(await res.json());

  } catch {
    showError("Bağlantı hatası — sunucunun çalıştığından emin olun.");
  } finally {
    setLoading(false);
  }
});

// ── Reset ─────────────────────────────────────────────────
btnReset.addEventListener("click", () => {
  hideError();
  resultSection.style.display = "none";
  form.querySelectorAll("input").forEach(el => el.classList.remove("error"));
});

// ── Clear error on input ──────────────────────────────────
form.querySelectorAll("input").forEach(el => {
  el.addEventListener("input", () => {
    el.classList.remove("error");
    if (errorToast.style.display === "flex") hideError();
  });
});
