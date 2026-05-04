/*
AeroTune UI cleanup.

Keeps:
- Main analyzer/upload workflow
- PID Tuning Advice card
- PID value calculator
- V1.5 Tune-Change Tracking
- About / YouTube creator card

Hides:
- Converter/optimizer clutter
- Parser/converter reports
- Old summary/machine output cards
- V1.4 comparison cards
*/

(function () {
  "use strict";

  const HIDE_HEADINGS = [
    "Converter / CSV Optimizer",
    "Converter Report",
    "Parser Report",
    "Summary",
    "Pilot Tune Notes",
    "Machine Output",
    "V1.4 Before / After Comparison",
    "V1.4 Comparison Result",
    "Comparison Machine Output"
  ];

  const HIDE_BUTTONS = [
    "Convert / Download AeroTune CSV",
    "Compare Before → After",
    "Copy Cards",
    "Copy Machine Notes"
  ];

  function normalize(text) {
    return String(text || "")
      .replace(/\s+/g, " ")
      .trim()
      .toLowerCase();
  }

  function shouldHideExact(text, list) {
    const cleaned = normalize(text);
    return list.some((target) => cleaned === normalize(target));
  }

  function findCardBlock(el) {
    const block = el.closest(
      "section, article, .card, .panel, .result-card, .feature-card, .dashboard-card, .upload-card, .glass, .box"
    );

    if (block && block !== document.body && block.tagName.toLowerCase() !== "main") {
      return block;
    }

    return null;
  }

  function hideBlockFromHeading(heading) {
    const card = findCardBlock(heading);

    if (card) {
      card.style.display = "none";
      card.setAttribute("data-aerotune-hidden", "legacy-card");
      return;
    }

    let node = heading;
    const toHide = [];

    while (node) {
      if (
        node !== heading &&
        node.nodeType === Node.ELEMENT_NODE &&
        node.tagName &&
        ["H1", "H2"].includes(node.tagName.toUpperCase())
      ) {
        break;
      }

      toHide.push(node);
      node = node.nextElementSibling;
    }

    toHide.forEach((x) => {
      x.style.display = "none";
      x.setAttribute("data-aerotune-hidden", "legacy-card");
    });
  }

  function cleanupLegacyUi() {
    Array.from(document.querySelectorAll("h2, h3, h4")).forEach((heading) => {
      if (shouldHideExact(heading.textContent, HIDE_HEADINGS)) {
        hideBlockFromHeading(heading);
      }
    });

    Array.from(document.querySelectorAll("button, a")).forEach((el) => {
      if (shouldHideExact(el.textContent, HIDE_BUTTONS)) {
        const card = findCardBlock(el);
        if (card) {
          card.style.display = "none";
          card.setAttribute("data-aerotune-hidden", "legacy-control");
        } else {
          el.style.display = "none";
          el.setAttribute("data-aerotune-hidden", "legacy-control");
        }
      }
    });
  }

  document.addEventListener("DOMContentLoaded", cleanupLegacyUi);

  const observer = new MutationObserver(() => cleanupLegacyUi());
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true
  });

  window.AeroTuneCleanupLegacyUi = cleanupLegacyUi;
})();
