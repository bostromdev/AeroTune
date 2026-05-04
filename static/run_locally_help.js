/*
AeroTune optional hosted-demo helper text.
Keeps the website clear that CSV is recommended and raw logs require local blackbox_decode.
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
          For best results, open your .BBL in Betaflight Blackbox Explorer, select the correct/latest flight, export that flight as CSV, then upload the CSV into AeroTune.
        </p>
        <p style="margin:0;line-height:1.45;opacity:.9;">
          Raw .BBL/.BFL/.TXT logs can work locally when Betaflight blackbox_decode is installed. CSV is recommended because it lets you choose the exact flight before analysis.
        </p>
      </div>
    `;

    const main = document.querySelector("main") || document.body;
    main.insertBefore(note, main.firstChild);
  }

  document.addEventListener("DOMContentLoaded", addRunLocallyNote);
  setTimeout(addRunLocallyNote, 300);
})();
