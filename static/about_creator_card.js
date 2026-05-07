/*
AeroTune creator/about card.

Adds a professional creator section with YouTube link.
*/

(function () {
  "use strict";

  function createCreatorCard() {
    if (document.getElementById("bostromdev-creator-card")) {
      return;
    }

    const card = document.createElement("section");
    card.id = "bostromdev-creator-card";
    card.innerHTML = `
      <div style="
        margin:24px 0;
        padding:20px;
        border:1px solid rgba(255,255,255,.14);
        border-radius:18px;
        background:rgba(18,22,34,.92);
        color:#f3f6ff;
        box-shadow:0 10px 30px rgba(0,0,0,.22);
        font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      ">
        <div style="display:flex;justify-content:space-between;gap:16px;align-items:flex-start;flex-wrap:wrap;">
          <div style="max-width:760px;">
            <p style="
              margin:0 0 8px;
              font-size:12px;
              letter-spacing:.08em;
              text-transform:uppercase;
              opacity:.72;
              font-weight:800;
            ">
              Built by BostromDev
            </p>

            <h2 style="margin:0 0 10px;font-size:24px;line-height:1.2;">
              Practical FPV tuning tools built from real flight testing
            </h2>

            <p style="margin:0 0 12px;line-height:1.55;opacity:.95;">
              AeroTune was built to make Betaflight Blackbox analysis easier to understand for real pilots.
              Instead of throwing raw charts and confusing numbers at the user, it translates gyro, setpoint,
              overshoot, bounceback, and noise behavior into conservative tuning suggestions that can be tested
              one flight at a time.
            </p>

            <p style="margin:0;line-height:1.55;opacity:.9;">
              This project is part of my larger engineering path: combining hands-on mechanical experience,
              FPV flight testing, software development, and aerospace-focused problem solving. The goal is not
              to replace pilot judgment — it is to give pilots better evidence so they can make smarter, safer,
              more repeatable tuning decisions.
            </p>
          </div>

          <div style="display:flex;flex-direction:column;gap:10px;min-width:210px;">
            <a
              href="https://www.youtube.com/@BostromDev"
              target="_blank"
              rel="noopener noreferrer"
              style="
                display:inline-flex;
                align-items:center;
                justify-content:center;
                text-decoration:none;
                padding:11px 14px;
                border-radius:12px;
                background:rgba(255,255,255,.12);
                border:1px solid rgba(255,255,255,.18);
                color:#f3f6ff;
                font-weight:800;
              "
            >
              Visit YouTube Channel
            </a>

            <span style="font-size:13px;line-height:1.4;opacity:.75;">
              Follow build updates, test flights, FPV tuning progress, and engineering projects.
            </span>
          </div>
        </div>
      </div>
    `;

    const explicitMount = document.getElementById("creatorCardMount");
    if (explicitMount) {
      explicitMount.appendChild(card);
      return;
    }

    const main =
      document.querySelector("main") ||
      document.querySelector(".container") ||
      document.body;

    const firstAnalyzerHeading = Array.from(document.querySelectorAll("h1, h2, h3"))
      .find((h) => /analyze|upload|blackbox|aerotune/i.test(h.textContent || ""));

    const targetBlock = firstAnalyzerHeading
      ? firstAnalyzerHeading.closest("section, article, .card, .panel, .upload-card, .glass, .box")
      : null;

    if (targetBlock && targetBlock.parentNode) {
      targetBlock.parentNode.insertBefore(card, targetBlock.nextSibling);
    } else if (main.firstChild) {
      main.insertBefore(card, main.firstChild.nextSibling);
    } else {
      main.appendChild(card);
    }
  }

  document.addEventListener("DOMContentLoaded", createCreatorCard);

  // Safety for pages that render late.
  setTimeout(createCreatorCard, 300);

  window.AeroTuneCreateCreatorCard = createCreatorCard;
})();
