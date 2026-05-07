/*
AeroTune Phase 1 UI cleanup.

The app now uses a Home Hub plus dedicated tool views, so the old cleanup script
must not hide converter, parser, comparison, or machine-output cards globally.
Those sections are organized by route and advanced <details> panels in index.html.
*/

(function () {
  "use strict";

  function markPhaseOneUiReady() {
    document.documentElement.setAttribute("data-aerotune-ui", "phase-1-hub");
  }

  document.addEventListener("DOMContentLoaded", markPhaseOneUiReady);
  markPhaseOneUiReady();

  // Kept for backward compatibility with older console/debug calls.
  window.AeroTuneCleanupLegacyUi = markPhaseOneUiReady;
})();
