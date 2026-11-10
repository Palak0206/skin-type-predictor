/* DERMA / INSIGHT – results page behaviour.
   Everything shown is read from data attributes that Jinja filled with the real model outputs.
   The analysis sequence is a presentation only: predictions already exist when the page loads. */
(function () {
  "use strict";

  const root = document.getElementById("results-root");
  if (!root) return;

  const data = {
    nlp: root.dataset.nlp,
    cnn: root.dataset.cnn,
    hasText: root.dataset.hasText === "true",
    hasImage: root.dataset.hasImage === "true",
    intensities: {
      oiliness: root.dataset.oiliness,
      dryness: root.dataset.dryness,
      sensitivity: root.dataset.sensitivity,
    },
  };

  const STAGE_MS = 1000;
  const LEVELS = { low: 1, medium: 2, high: 3 };
  const BLURBS = {
    oily: "Your prediction suggests a tendency toward oilier skin characteristics.",
    dry: "Your prediction suggests characteristics commonly associated with dry skin.",
    normal: "Your prediction suggests generally balanced skin characteristics.",
    combination: "Your prediction suggests different skin characteristics across different areas.",
  };

  let timers = [];
  let finished = false;

  /* ---------- analysis sequence ---------- */
  function initializeAnalysisSequence() {
    const analysis = document.getElementById("analysis");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!analysis || reduceMotion) { revealResults(false); return; }

    document.body.classList.add("is-analyzing");
    document.getElementById("skip-animation").addEventListener("click", skipAnalysisAnimation);

    const stages = Array.from(document.querySelectorAll(".stage"));
    // Skip stages that do not apply to the submitted inputs.
    stages.forEach((stage) => {
      const id = stage.dataset.stage;
      const missingText = !data.hasText && (id === "preprocess" || id === "nlp");
      const missingImage = !data.hasImage && id === "image";
      if (missingText || missingImage) {
        stage.classList.add("is-skipped");
        stage.dataset.skipped = "true";
        stage.querySelector(".stage-state").textContent = "Not needed";
      }
    });

    const queue = stages.filter((s) => !s.dataset.skipped);
    queue.forEach((stage, i) => {
      timers.push(setTimeout(() => showAnalysisStage(stage, queue[i - 1]), i * STAGE_MS));
    });
    timers.push(setTimeout(() => revealResults(true), queue.length * STAGE_MS + 300));
  }

  function showAnalysisStage(stage, previous) {
    if (previous) setStageState(previous, "is-done", "Complete");
    setStageState(stage, "is-active", "In progress");
    const title = stage.querySelector("strong").textContent;
    document.getElementById("analysis-status").textContent = title + "…";
  }

  function setStageState(stage, cls, label) {
    stage.classList.remove("is-active", "is-done");
    stage.classList.add(cls);
    stage.querySelector(".stage-state").textContent = label;
  }

  function skipAnalysisAnimation() { revealResults(false); }

  function revealResults(animated) {
    if (finished) return;
    finished = true;
    timers.forEach(clearTimeout);
    timers = [];

    document.querySelectorAll(".stage:not([data-skipped])").forEach((s) => setStageState(s, "is-done", "Complete"));
    const analysis = document.getElementById("analysis");
    const finish = () => {
      if (analysis) analysis.hidden = true;
      document.body.classList.remove("is-analyzing");
      document.body.classList.add("is-revealed");
      applyIntensityLevels();
      const title = document.getElementById("report-title");
      if (title) title.focus({ preventScroll: true });
      window.scrollTo(0, 0);
    };
    if (animated && analysis) {
      analysis.classList.add("is-leaving");
      setTimeout(finish, 600);
    } else {
      finish();
    }
  }

  /* ---------- results visuals ---------- */
  function initializeSkinVisualization() {
    const visual = document.getElementById("skin-visual");
    if (!visual) return;
    // CNN output drives the illustration; fall back to the NLP type, clearly labelled.
    const fromCnn = data.cnn !== "";
    const type = (fromCnn ? data.cnn : data.nlp).trim().toLowerCase();
    visual.dataset.skin = BLURBS[type] ? type : "none";

    document.getElementById("hero-source").textContent = fromCnn
      ? "Image model (CNN) prediction"
      : data.nlp ? "Text model (NLP) prediction — no image prediction available" : "";
    document.getElementById("hero-blurb").textContent =
      BLURBS[type] || "Submit a description or an image to see a skin type illustration.";
  }

  function initializeAgreementNote() {
    const note = document.getElementById("agreement");
    if (!note || !data.nlp || !data.cnn) return;
    const same = data.nlp.trim().toLowerCase() === data.cnn.trim().toLowerCase();
    note.textContent = same
      ? "Your text-based and image-based predictions agree."
      : "Your text-based and image-based predictions differ. The models analyze different inputs, so both results are shown.";
    note.hidden = false;
  }

  function applyIntensityLevels() {
    document.querySelectorAll(".meter").forEach((meter) => {
      const level = LEVELS[(meter.dataset.value || "").trim().toLowerCase()] || 0;
      meter.dataset.level = level;
      meter.querySelector(".meter-fill").style.width = (level / 3) * 100 + "%";
      meter.querySelectorAll(".seg").forEach((seg, i) => seg.classList.toggle("is-current", i + 1 === level));
    });
  }

  /* ---------- image preview ---------- */
  function initializeImagePreview() {
    const open = document.getElementById("open-image");
    const dialog = document.getElementById("image-dialog");
    if (!open || !dialog || typeof dialog.showModal !== "function") return;
    open.addEventListener("click", () => dialog.showModal());
    document.getElementById("close-image").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
  }

  initializeSkinVisualization();
  initializeAgreementNote();
  initializeImagePreview();
  initializeAnalysisSequence();
})();