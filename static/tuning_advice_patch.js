/*
AeroTune frontend tuning-advice renderer.

- Watches API JSON responses.
- If tuning_advice exists, renders the card at the TOP of results.
- Adds current Betaflight PID inputs.
- Calculates suggested Betaflight values from percent deltas.
*/

(function () {
  "use strict";

  function findAdvice(obj) {
    if (!obj || typeof obj !== "object") return null;
    if (obj.tuning_advice) return obj.tuning_advice;
    if (obj.analysis && obj.analysis.tuning_advice) return obj.analysis.tuning_advice;
    if (obj.result && obj.result.tuning_advice) return obj.result.tuning_advice;
    if (obj.data && obj.data.tuning_advice) return obj.data.tuning_advice;
    return null;
  }

  function deltaText(value) {
    const n = Number(value || 0);
    if (n > 0) return `+${n}%`;
    if (n < 0) return `${n}%`;
    return "0%";
  }

  function pidInputId(axis, field) {
    return `pid-${axis}-${field}`;
  }

  function readPidValue(axis, field) {
    const el = document.getElementById(pidInputId(axis, field));
    if (!el) return null;

    const raw = String(el.value || "").trim();
    if (raw === "") return null;

    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  }

  function calcNewPid(baseValue, percentDelta) {
    if (baseValue === null || baseValue === undefined) return "";
    const next = Math.round(baseValue * (1 + Number(percentDelta || 0) / 100));
    return String(Math.max(0, next));
  }

  function axisRows(advice) {
    const axes = advice.axes || {};
    const order = ["roll", "pitch", "yaw"];

    return order.map((axis) => {
      const rec = axes[axis] || {};
      const deltas = rec.deltas || {};
      const evidence = rec.evidence || {};
      const action = rec.action || "no_change";

      return `
        <tr>
          <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,.1);font-weight:700;text-transform:capitalize;">${axis}</td>
          <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,.1);">${action}</td>
          <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,.1);">${rec.severity || "unknown"}</td>
          <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,.1);">${deltaText(deltas.p_percent)}</td>
          <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,.1);">${deltaText(deltas.i_percent)}</td>
          <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,.1);">${deltaText(deltas.d_percent)}</td>
          <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,.1);">${deltaText(deltas.dmax_percent)}</td>
          <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,.1);">${deltaText(deltas.ff_percent)}</td>
          <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,.1);font-size:12px;opacity:.85;">
            OS ${evidence.overshoot_score ?? "-"} / BB ${evidence.bounceback_score ?? "-"} / Noise ${evidence.noise_score ?? "-"}
          </td>
        </tr>
      `;
    }).join("");
  }

  function listItems(items) {
    if (!Array.isArray(items)) return "";
    return items.map((x) => `<li style="margin:4px 0;">${String(x)}</li>`).join("");
  }

  function currentPidInputs() {
    const axes = ["roll", "pitch", "yaw"];
    const fields = [
      ["p", "P"],
      ["i", "I"],
      ["dmax", "D Max"],
      ["d", "D"],
      ["ff", "FF"]
    ];

    const rows = axes.map((axis) => {
      const cells = fields.map(([field, label]) => `
        <td style="padding:6px;border-bottom:1px solid rgba(255,255,255,.1);">
          <input
            id="${pidInputId(axis, field)}"
            class="aerotune-pid-input"
            data-pid-input="true"
            data-axis="${axis}"
            data-field="${field}"
            type="number"
            min="0"
            step="1"
            placeholder="${label}"
            style="
              width:74px;
              padding:7px 8px;
              border-radius:8px;
              border:1px solid rgba(255,255,255,.18);
              background:rgba(255,255,255,.08);
              color:#f3f6ff;
            "
          />
        </td>
      `).join("");

      return `
        <tr>
          <td style="padding:6px;border-bottom:1px solid rgba(255,255,255,.1);font-weight:700;text-transform:capitalize;">${axis}</td>
          ${cells}
        </tr>
      `;
    }).join("");

    return `
      <div style="margin-top:18px;padding:14px;border-radius:14px;background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.12);">
        <h3 style="margin:0 0 8px;font-size:17px;">Convert Advice to Betaflight Values</h3>
        <p style="margin:0 0 10px;opacity:.9;line-height:1.4;">
          Enter your current Betaflight PID values after uploading a CSV. AeroTune will estimate the next values from the percentage deltas above.
        </p>

        <div style="overflow-x:auto;">
          <table style="width:100%;border-collapse:collapse;font-size:14px;">
            <thead>
              <tr>
                <th style="text-align:left;padding:6px;border-bottom:1px solid rgba(255,255,255,.2);">Axis</th>
                <th style="text-align:left;padding:6px;border-bottom:1px solid rgba(255,255,255,.2);">P</th>
                <th style="text-align:left;padding:6px;border-bottom:1px solid rgba(255,255,255,.2);">I</th>
                <th style="text-align:left;padding:6px;border-bottom:1px solid rgba(255,255,255,.2);">D Max</th>
                <th style="text-align:left;padding:6px;border-bottom:1px solid rgba(255,255,255,.2);">D</th>
                <th style="text-align:left;padding:6px;border-bottom:1px solid rgba(255,255,255,.2);">FF</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>

        <button
          id="calculate-pid-values"
          type="button"
          style="
            margin-top:12px;
            padding:9px 13px;
            border-radius:10px;
            border:1px solid rgba(255,255,255,.18);
            background:rgba(255,255,255,.12);
            color:#f3f6ff;
            cursor:pointer;
            font-weight:700;
          "
        >
          Calculate Suggested Betaflight Values
        </button>

        <div id="pid-value-output" style="margin-top:12px;"></div>
      </div>
    `;
  }

  function renderCalculatedPidValues(advice) {
    const axes = advice.axes || {};
    const order = ["roll", "pitch", "yaw"];
    const fields = [
      ["p", "p_percent", "P"],
      ["i", "i_percent", "I"],
      ["dmax", "dmax_percent", "D Max"],
      ["d", "d_percent", "D"],
      ["ff", "ff_percent", "FF"]
    ];

    const rows = order.map((axis) => {
      const rec = axes[axis] || {};
      const deltas = rec.deltas || {};

      const cells = fields.map(([field, deltaKey]) => {
        const base = readPidValue(axis, field);
        const delta = Number(deltas[deltaKey] || 0);

        if (base === null) {
          return `<td style="padding:7px;border-bottom:1px solid rgba(255,255,255,.1);opacity:.55;">—</td>`;
        }

        const next = calcNewPid(base, delta);
        const sign = delta > 0 ? "+" : "";

        return `
          <td style="padding:7px;border-bottom:1px solid rgba(255,255,255,.1);">
            <strong>${base} → ${next}</strong>
            <div style="font-size:12px;opacity:.75;">${sign}${delta}%</div>
          </td>
        `;
      }).join("");

      return `
        <tr>
          <td style="padding:7px;border-bottom:1px solid rgba(255,255,255,.1);font-weight:700;text-transform:capitalize;">${axis}</td>
          ${cells}
        </tr>
      `;
    }).join("");

    const out = document.getElementById("pid-value-output");
    if (!out) {
      console.warn("AeroTune PID output container not found.");
      return;
    }

    out.innerHTML = `
      <div style="margin-top:10px;">
        <h4 style="margin:0 0 8px;font-size:15px;">Suggested Betaflight Values</h4>
        <div style="overflow-x:auto;">
          <table style="width:100%;border-collapse:collapse;font-size:14px;">
            <thead>
              <tr>
                <th style="text-align:left;padding:7px;border-bottom:1px solid rgba(255,255,255,.2);">Axis</th>
                <th style="text-align:left;padding:7px;border-bottom:1px solid rgba(255,255,255,.2);">P</th>
                <th style="text-align:left;padding:7px;border-bottom:1px solid rgba(255,255,255,.2);">I</th>
                <th style="text-align:left;padding:7px;border-bottom:1px solid rgba(255,255,255,.2);">D Max</th>
                <th style="text-align:left;padding:7px;border-bottom:1px solid rgba(255,255,255,.2);">D</th>
                <th style="text-align:left;padding:7px;border-bottom:1px solid rgba(255,255,255,.2);">FF</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        <p style="margin:10px 0 0;opacity:.8;font-size:13px;line-height:1.35;">
          Values are rounded to whole Betaflight numbers. Save, reconnect, confirm they stuck, then test one flight before changing again.
        </p>
      </div>
    `;
  }

  function attachCalculatorEvents(advice) {
    const button = document.getElementById("calculate-pid-values");

    if (button) {
      button.onclick = function () {
        renderCalculatedPidValues(advice);
      };
    }

    document.querySelectorAll("[data-pid-input='true']").forEach((input) => {
      input.onkeydown = function (event) {
        if (event.key === "Enter") {
          renderCalculatedPidValues(advice);
        }
      };
    });
  }

  function renderAdvice(advice) {
    if (!advice || typeof advice !== "object") return;

    let mount =
      document.querySelector("#results") ||
      document.querySelector("#analysis-results") ||
      document.querySelector(".results") ||
      document.querySelector("main") ||
      document.body;

    let card = document.querySelector("#aerotune-tuning-advice-card");
    if (!card) {
      card = document.createElement("section");
      card.id = "aerotune-tuning-advice-card";

      if (mount.firstChild) {
        mount.insertBefore(card, mount.firstChild);
      } else {
        mount.appendChild(card);
      }
    }

    card.innerHTML = `
      <div style="
        margin:24px 0;
        padding:18px;
        border:1px solid rgba(255,255,255,.16);
        border-radius:16px;
        background:rgba(20,24,36,.92);
        color:#f3f6ff;
        box-shadow:0 10px 30px rgba(0,0,0,.25);
        font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      ">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
          <h2 style="margin:0;font-size:22px;">AeroTune PID Tuning Advice</h2>
          <span style="font-size:13px;opacity:.8;">${advice.version || ""}</span>
        </div>

        <p style="margin:12px 0 8px;line-height:1.45;">${advice.summary || "No tuning advice summary available."}</p>

        <div style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 0;">
          <span style="padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.1);">Mode: ${advice.mode || "delta_percent"}</span>
          <span style="padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.1);">Confidence: ${advice.confidence || "unknown"}</span>
          <span style="padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.1);">Drone: ${advice.drone_size || "unknown"}"</span>
          <span style="padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.1);">Sample rate: ${advice.sample_rate_hz || "unknown"} Hz</span>
        </div>

        <div style="overflow-x:auto;margin-top:14px;">
          <table style="width:100%;border-collapse:collapse;font-size:14px;">
            <thead>
              <tr>
                <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(255,255,255,.2);">Axis</th>
                <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(255,255,255,.2);">Action</th>
                <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(255,255,255,.2);">Severity</th>
                <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(255,255,255,.2);">P</th>
                <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(255,255,255,.2);">I</th>
                <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(255,255,255,.2);">D</th>
                <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(255,255,255,.2);">D Max</th>
                <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(255,255,255,.2);">FF</th>
                <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(255,255,255,.2);">Evidence</th>
              </tr>
            </thead>
            <tbody>${axisRows(advice)}</tbody>
          </table>
        </div>

        ${currentPidInputs()}

        <h3 style="margin:18px 0 8px;font-size:17px;">Betaflight Steps</h3>
        <ul style="margin:0 0 12px 18px;padding:0;line-height:1.45;">${listItems(advice.betaflight_steps)}</ul>

        <h3 style="margin:18px 0 8px;font-size:17px;">Next Local Test Plan</h3>
        <ul style="margin:0 0 12px 18px;padding:0;line-height:1.45;">${listItems(advice.test_plan)}</ul>

        <h3 style="margin:18px 0 8px;font-size:17px;">Safety / Reality Checks</h3>
        <ul style="margin:0 0 0 18px;padding:0;line-height:1.45;">${listItems(advice.safety)}</ul>
      </div>
    `;

    attachCalculatorEvents(advice);

    // After analysis finishes, automatically bring the user back to the tuning card.
    // This makes the recommendation impossible to miss after clicking Analyze.
    setTimeout(function () {
      const cardTop = card.getBoundingClientRect().top + window.scrollY - 16;
      window.scrollTo({
        top: Math.max(0, cardTop),
        behavior: "smooth"
      });
    }, 150);
  }

  const originalFetch = window.fetch;

  window.fetch = async function () {
    const response = await originalFetch.apply(this, arguments);

    try {
      const clone = response.clone();
      const contentType = clone.headers.get("content-type") || "";

      if (contentType.includes("application/json")) {
        clone.json().then((json) => {
          const advice = findAdvice(json);
          if (advice) {
            window.__AEROTUNE_LAST_TUNING_ADVICE__ = advice;
            renderAdvice(advice);
          }
        }).catch(() => {});
      }
    } catch (_) {}

    return response;
  };

  // Manual fallback for debugging in browser console:
  window.AeroTuneRenderTuningAdvice = renderAdvice;
  window.AeroTuneCalculatePidValues = function () {
    if (window.__AEROTUNE_LAST_TUNING_ADVICE__) {
      renderCalculatedPidValues(window.__AEROTUNE_LAST_TUNING_ADVICE__);
    }
  };
})();
