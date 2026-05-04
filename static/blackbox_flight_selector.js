/*
AeroTune V1.7 raw Blackbox multi-flight selector.

When blackbox_decode produces multiple CSV files from one raw .BBL/.BFL/.TXT,
AeroTune treats them as Flight 1/N ... Flight N/N and selects Flight N/N by
default as the latest flight. This helper lets the user switch flights without
uploading the raw log again.
*/

(function () {
  "use strict";

  function getConverterReport(json) {
    if (!json || typeof json !== "object") return null;
    if (json.converter_report) return json.converter_report;
    if (json.analysis && json.analysis.converter_report) return json.analysis.converter_report;
    if (json.result && json.result.converter_report) return json.result.converter_report;
    return null;
  }

  function hasMultiFlight(report) {
    return Boolean(
      report &&
      report.converted === true &&
      report.conversion_id &&
      Array.isArray(report.available_flights) &&
      report.available_flights.length > 1
    );
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function currentDroneSize() {
    const el = document.getElementById("droneSize") || document.querySelector("[name='drone_size']");
    return el && el.value ? el.value : "7";
  }

  function currentTuningGoal() {
    const el = document.getElementById("tuningGoal") || document.querySelector("[name='tuning_goal']");
    return el && el.value ? el.value : "efficient";
  }

  function findMount() {
    return (
      document.querySelector("#aerotune-tuning-advice-card") ||
      document.querySelector("#results") ||
      document.querySelector("#analysis-results") ||
      document.querySelector("main") ||
      document.querySelector(".wrap") ||
      document.body
    );
  }

  function insertCard(card) {
    const tuningCard = document.querySelector("#aerotune-tuning-advice-card");
    if (tuningCard && tuningCard.parentNode) {
      tuningCard.parentNode.insertBefore(card, tuningCard);
      return;
    }

    const mount = findMount();
    if (mount.firstChild) {
      mount.insertBefore(card, mount.firstChild);
    } else {
      mount.appendChild(card);
    }
  }

  function scrollToRawFlightSelector() {
    const card = document.getElementById("aerotune-raw-flight-selector-card");
    if (!card) return;

    setTimeout(function () {
      const cardTop = card.getBoundingClientRect().top + window.scrollY - 16;
      window.scrollTo({
        top: Math.max(0, cardTop),
        behavior: "smooth"
      });
    }, 260);
  }

  function selectedFlightIndex(report) {
    const selected = report.available_flights.find((item) => item.is_selected);
    if (selected) return Number(selected.flight_index);
    return Number(report.selected_flight_index || report.available_flights.length || 1);
  }

  function renderFlightSelector(report) {
    if (!hasMultiFlight(report)) return;

    let card = document.getElementById("aerotune-raw-flight-selector-card");
    if (!card) {
      card = document.createElement("section");
      card.id = "aerotune-raw-flight-selector-card";
      insertCard(card);
    }

    const currentIndex = selectedFlightIndex(report);
    const options = report.available_flights.map((flight) => {
      const idx = Number(flight.flight_index);
      const selected = idx === currentIndex ? "selected" : "";
      const latest = flight.is_latest ? " — latest/default" : "";
      const size = Number(flight.size_bytes || 0);
      const sizeMb = size > 0 ? ` · ${(size / 1024 / 1024).toFixed(2)} MB` : "";
      return `<option value="${idx}" ${selected}>${escapeHtml(`Flight ${idx}/${flight.flight_count}${latest}${sizeMb}`)}</option>`;
    }).join("");

    card.innerHTML = `
      <div style="
        margin:20px 0;
        padding:18px;
        border:1px solid rgba(99,214,255,.28);
        border-radius:18px;
        background:rgba(12,23,29,.95);
        color:#e9f4f7;
        box-shadow:0 14px 40px rgba(0,0,0,.24);
        font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      ">
        <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;">
          <div>
            <p style="margin:0 0 6px;font-size:12px;letter-spacing:.08em;text-transform:uppercase;opacity:.7;font-weight:800;">
              Raw Blackbox Multi-Flight Selector
            </p>
            <h2 style="margin:0 0 8px;font-size:22px;">Choose which flight to analyze</h2>
            <p style="margin:0;line-height:1.45;opacity:.9;max-width:760px;">
              This raw Blackbox file produced ${report.available_flights.length} decoded flight profiles.
              AeroTune selected <strong>${escapeHtml(report.selected_flight_label || `Flight ${currentIndex}/${report.available_flights.length}`)}</strong> by default.
              Flight 1/${report.available_flights.length} is treated as the oldest flight, and Flight ${report.available_flights.length}/${report.available_flights.length} is treated as the newest/latest flight.
              You do not need to manually export CSV when raw conversion is set up; AeroTune decodes the raw log locally, then analyzes the selected decoded flight.
            </p>
          </div>
          <span style="padding:7px 10px;border-radius:999px;background:rgba(99,214,255,.12);border:1px solid rgba(99,214,255,.25);font-size:13px;font-weight:800;">
            ${report.available_flights.length} flights detected
          </span>
        </div>

        <label for="aerotune-raw-flight-select" style="display:block;margin:14px 0 6px;color:#8fa5ad;font-size:13px;font-weight:800;">Select decoded flight</label>
        <select id="aerotune-raw-flight-select" style="
          width:100%;
          border-radius:12px;
          border:1px solid rgba(99,214,255,.28);
          background:#0d171d;
          color:#e9f4f7;
          padding:12px 13px;
          font:inherit;
        ">
          ${options}
        </select>

        <button id="aerotune-analyze-selected-flight" type="button" style="
          width:100%;
          margin-top:12px;
          border-radius:12px;
          border:1px solid rgba(99,214,255,.42);
          background:linear-gradient(135deg,#15566a,#12313e);
          color:#e9f4f7;
          padding:12px 13px;
          font:inherit;
          font-weight:900;
          cursor:pointer;
        ">
          Analyze Selected Flight
        </button>

        <div id="aerotune-raw-flight-status" style="margin-top:10px;line-height:1.4;opacity:.82;font-size:13px;">
          Latest flight is selected automatically. Use this selector when one raw .BBL contains multiple flights and you want to analyze a specific session.
        </div>
      </div>
    `;

    scrollToRawFlightSelector();

    const button = document.getElementById("aerotune-analyze-selected-flight");
    const select = document.getElementById("aerotune-raw-flight-select");
    const status = document.getElementById("aerotune-raw-flight-status");

    if (button && select) {
      button.onclick = async function () {
        const selected = Number(select.value || currentIndex);
        const body = new FormData();
        body.append("conversion_id", report.conversion_id);
        body.append("flight_index", String(selected));
        body.append("drone_size", currentDroneSize());
        body.append("tuning_goal", currentTuningGoal());

        button.disabled = true;
        button.textContent = "Analyzing selected flight...";
        if (status) status.textContent = `Analyzing Flight ${selected}/${report.available_flights.length}...`;

        try {
          const response = await fetch("/analyze-converted-flight", {
            method: "POST",
            body,
          });

          const data = await response.json().catch(() => ({}));
          if (!response.ok) {
            throw new Error(data.error || "Selected flight analysis failed.");
          }

          const nextReport = getConverterReport(data);
          if (hasMultiFlight(nextReport)) {
            renderFlightSelector(nextReport);
          }

          if (status) {
            status.textContent = data.message || `Flight ${selected}/${report.available_flights.length} analyzed.`;
          }

          window.dispatchEvent(new CustomEvent("aerotune:raw-flight-selected", { detail: data }));
        } catch (error) {
          if (status) status.textContent = error.message || String(error);
        } finally {
          button.disabled = false;
          button.textContent = "Analyze Selected Flight";
        }
      };
    }
  }

  const originalFetch = window.fetch;
  window.fetch = async function () {
    const response = await originalFetch.apply(this, arguments);

    try {
      const clone = response.clone();
      const contentType = clone.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        clone.json().then((json) => {
          const report = getConverterReport(json);
          if (hasMultiFlight(report)) {
            window.__AEROTUNE_LAST_RAW_FLIGHT_REPORT__ = report;
            renderFlightSelector(report);
          }
        }).catch(() => {});
      }
    } catch (_) {}

    return response;
  };

  window.AeroTuneRenderRawFlightSelector = renderFlightSelector;
})();
