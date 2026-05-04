/*
AeroTune optional hosted-demo helper text.
Keeps the website clear that raw .BBL support works locally when blackbox_decode is available,
while CSV remains the safest fallback workflow.
*/
(function () {
  "use strict";

  function addRunLocallyNote() {
    if (document.getElementById("aerotune-run-locally-note")) return;

    const note = document.createElement("section");
    note.id = "aerotune-run-locally-note";
    note.innerHTML = `
      <div style="margin:18px 0;padding:16px;border-radius:14px;border:1px solid rgba(255,202,95,.45);background:rgba(255,202,95,.08);color:inherit;">
        <h3 style="margin:0 0 8px;">Run AeroTune Locally</h3>
        <p style="margin:0 0 8px;line-height:1.45;">
          AeroTune can analyze Betaflight CSV exports directly. Raw .BBL/.BFL/.TXT logs can also work locally after you follow the README setup and have Betaflight <code>blackbox_decode</code> available.
        </p>
        <p style="margin:0 0 8px;line-height:1.45;opacity:.92;">
          In V1.7, raw Blackbox uploads can expose multiple decoded flights. AeroTune treats Flight 1/N as the oldest and Flight N/N as the newest/latest by default.
        </p>
        <p style="margin:0;line-height:1.45;opacity:.9;">
          You do not need to manually export CSV when raw conversion is set up. CSV is still the safest fallback if raw conversion is not installed or if you prefer choosing the exact flight in Betaflight Blackbox Explorer first.
        </p>
      </div>
    `;

    const main = document.querySelector("main") || document.body;
    main.insertBefore(note, main.firstChild);
  }

  document.addEventListener("DOMContentLoaded", addRunLocallyNote);
  setTimeout(addRunLocallyNote, 300);
})();
