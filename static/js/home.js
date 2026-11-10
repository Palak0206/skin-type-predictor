/* =========================================================
   DERMA / INSIGHT homepage script (vanilla JavaScript)
   1. Navigation   2. Skin cards   3. How it works
   4. Steps        5. Textarea     6. Image upload
   7. Skin Buddy   8. Submit + rain
   Each part is its own function, started at the bottom.
   ========================================================= */

/* ---------- Small helpers ---------- */
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* Section the user is currently viewing ("home", "explore" or "analyze").
   Skin Buddy reads this to pick a message. */
let currentSection = "home";
let onSectionChange = () => {};   // replaced by Skin Buddy in setupSkinBuddy()

/* ---------- 1. Navigation ---------- */
function setupNavigation() {
    const links = $$(".nav-links a");
    const sections = $$("main section[id]");
    const toggle = $("#nav-toggle");
    const menu = $("#nav-links");

    // Highlight the link of the section that is in the middle of the screen
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            currentSection = entry.target.id;
            links.forEach((link) => {
                const isActive = link.dataset.section === currentSection;
                link.classList.toggle("is-active", isActive);
                if (isActive) link.setAttribute("aria-current", "page");
                else link.removeAttribute("aria-current");
            });
            onSectionChange();
        });
    }, { rootMargin: "-45% 0px -50% 0px" });
    sections.forEach((section) => observer.observe(section));

    // Mobile menu
    function setMenu(open) {
        menu.classList.toggle("is-open", open);
        toggle.setAttribute("aria-expanded", String(open));
        toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    }
    toggle.addEventListener("click", () => setMenu(!menu.classList.contains("is-open")));
    links.forEach((link) => link.addEventListener("click", () => setMenu(false)));
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") setMenu(false); });
}

/* ---------- 2. Skin type cards ---------- */
function setupSkinCards() {
    const cards = $$(".skin-card");
    const panel = $("#skin-detail");
    const articles = $$("[data-detail]");
    const closeButton = $("#skin-detail-close");

    function showSkin(skin) {
        panel.hidden = false;
        panel.dataset.skin = skin;               // CSS uses this to switch the visualization
        articles.forEach((article) => { article.hidden = article.dataset.detail !== skin; });
        cards.forEach((card) => {
            const selected = card.dataset.skin === skin;
            card.setAttribute("aria-expanded", String(selected));
            const label = card.querySelector(".skin-more");
            label.textContent = selected ? "Showing details" : "Explore";
        });
        if (prefersReducedMotion()) return;
        panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function hideSkin(returnFocusTo) {
        panel.hidden = true;
        panel.dataset.skin = "none";
        articles.forEach((article) => { article.hidden = true; });
        cards.forEach((card) => {
            card.setAttribute("aria-expanded", "false");
            card.querySelector(".skin-more").textContent = "Explore";
        });
        if (returnFocusTo) returnFocusTo.focus();
    }

    cards.forEach((card) => {
        card.addEventListener("click", () => {
            // Clicking the selected card again closes the panel
            if (card.getAttribute("aria-expanded") === "true") hideSkin(card);
            else showSkin(card.dataset.skin);
        });
    });

    closeButton.addEventListener("click", () => {
        const selectedCard = cards.find((card) => card.getAttribute("aria-expanded") === "true");
        hideSkin(selectedCard);
    });

    panel.addEventListener("keydown", (event) => {
        if (event.key !== "Escape") return;
        const selectedCard = cards.find((card) => card.getAttribute("aria-expanded") === "true");
        hideSkin(selectedCard);
    });
}

/* ---------- 3. "How does it work?" ---------- */
function setupHowItWorks() {
    const toggle = $("#how-toggle");
    const more = $("#how-more");

    toggle.addEventListener("click", () => {
        const opening = more.hidden;
        more.hidden = !opening;
        toggle.setAttribute("aria-expanded", String(opening));
        toggle.textContent = opening ? "Hide details" : "Show details";
    });
}

/* ---------- 4. Three-step guide ---------- */
const STEP_TIPS = {
    1: "Mention shine, tightness, flaking, or how your skin feels by evening.",
    2: "Face a window, avoid filters, and keep your whole face in the frame.",
    3: "Press the button once. The real prediction runs on the server, then the result page opens."
};

function setupSteps() {
    const steps = $$(".step");
    const tipText = $("#tip-text");
    const tipIcon = $("#tip-icon");
    const progressBar = $("#progress-bar");

    function selectStep(number) {
        steps.forEach((step) => {
            const active = step.dataset.step === String(number);
            step.classList.toggle("is-active", active);
            if (active) step.setAttribute("aria-current", "step");
            else step.removeAttribute("aria-current");
        });
        tipText.textContent = STEP_TIPS[number];
        tipIcon.textContent = number;
        progressBar.style.width = (number / 3) * 100 + "%";
    }

    steps.forEach((step) => step.addEventListener("click", () => selectStep(step.dataset.step)));

    // The guide also follows what the user is doing in the form
    $("#skin_description").addEventListener("focus", () => selectStep(1));
    $("#skin_image").addEventListener("focus", () => selectStep(2));
    $("#dropzone").addEventListener("mouseenter", () => selectStep(2));
    $("#predict-button").addEventListener("focus", () => selectStep(3));
    $("#predict-button").addEventListener("mouseenter", () => selectStep(3));
}

/* ---------- 5. Textarea: counter and clear button ---------- */
function setupTextarea() {
    const textarea = $("#skin_description");
    const counter = $("#char-count");
    const clearButton = $("#clear-text");

    function refresh() {
        counter.textContent = textarea.value.length;
        clearButton.hidden = textarea.value.length === 0;
        textarea.classList.toggle("has-text", textarea.value.trim().length > 0);
    }

    textarea.addEventListener("input", refresh);
    clearButton.addEventListener("click", () => {
        textarea.value = "";
        refresh();
        textarea.focus();
    });
    refresh();   // also handles text the browser restores after Back
}

/* ---------- 6. Image upload ---------- */
const ALLOWED_TYPES = ["image/png", "image/jpeg", "image/webp"];
const ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"];

function setupImageUpload() {
    const upload = $("#upload");
    const input = $("#skin_image");          // the REAL file input sent to Flask
    const preview = $("#upload-preview");
    const previewImage = $("#preview-image");
    const previewName = $("#preview-name");
    const previewSize = $("#preview-size");
    const errorBox = $("#upload-error");
    const removeButton = $("#remove-image");
    let objectUrl = null;                    // remembered so it can be revoked later

    function isValidImage(file) {
        const name = file.name.toLowerCase();
        return ALLOWED_TYPES.includes(file.type) || (file.type === "" && ALLOWED_EXTENSIONS.some((ext) => name.endsWith(ext)));
    }

    function formatSize(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    function releaseObjectUrl() {
        if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    }

    function showError(message) {
        errorBox.textContent = message;
        errorBox.hidden = false;
    }

    function clearImage() {
        input.value = "";                    // empties the real file input
        releaseObjectUrl();
        previewImage.removeAttribute("src");
        previewName.textContent = "";
        previewSize.textContent = "";
        preview.hidden = true;
        upload.classList.remove("has-file");
    }

    function showPreview(file) {
        releaseObjectUrl();
        objectUrl = URL.createObjectURL(file);
        previewImage.src = objectUrl;
        previewName.textContent = file.name;
        previewSize.textContent = formatSize(file.size);
        preview.hidden = false;
        upload.classList.add("has-file");
    }

    function handleFile(file) {
        errorBox.hidden = true;
        if (!file) { clearImage(); return; }
        if (!isValidImage(file)) {
            clearImage();
            showError("That file type is not supported. Please choose a PNG, JPG, or WEBP image.");
            return;
        }
        showPreview(file);
    }

    // Normal file picker
    input.addEventListener("change", () => {
        // If the browser emptied the input (for example the picker was cancelled),
        // handleFile(undefined) resets the preview so it never shows a file that will not be sent.
        handleFile(input.files[0]);
    });

    removeButton.addEventListener("click", () => {
        errorBox.hidden = true;
        clearImage();
        input.focus();   // keyboard users can pick another image right away
    });

    // Drag and drop
    ["dragenter", "dragover"].forEach((type) => {
        upload.addEventListener(type, (event) => {
            event.preventDefault();
            upload.classList.add("is-dragging");
        });
    });
    upload.addEventListener("dragleave", (event) => {
        if (!upload.contains(event.relatedTarget)) upload.classList.remove("is-dragging");
    });
    upload.addEventListener("drop", (event) => {
        event.preventDefault();
        upload.classList.remove("is-dragging");
        const file = event.dataTransfer.files[0];
        if (!file) return;
        if (!isValidImage(file)) {
            clearImage();
            showError("That file type is not supported. Please choose a PNG, JPG, or WEBP image.");
            return;
        }
        // Put the dropped file into the real input so Flask receives it
        const transfer = new DataTransfer();
        transfer.items.add(file);
        input.files = transfer.files;
        handleFile(file);
    });

    // Stop the browser from opening a file that is dropped outside the upload area
    ["dragover", "drop"].forEach((type) => {
        window.addEventListener(type, (event) => { if (!upload.contains(event.target)) event.preventDefault(); });
    });

    window.addEventListener("pagehide", releaseObjectUrl);

    // Back/forward cache: show the preview again if the browser kept the file
    if (input.files.length > 0) handleFile(input.files[0]);
}

/* ---------- 7. Skin Buddy ---------- */
const BUDDY_MESSAGES = {
    home: ["Hey! I'll help you explore your skin.", "Scroll down or tap a button to begin."],
    explore: ["Curious about your skin type? Tap a card to learn more.", "Each card opens a short explanation."],
    analyze: ["Ready? Describe your skin or upload a photo.", "You can use text, a photo, or both."]
};

function setupSkinBuddy() {
    const button = $("#buddy-button");
    const bubble = $("#buddy-bubble");
    const message = $("#buddy-message");
    const closeButton = $("#buddy-close");
    let messageIndex = 0;

    function showMessage() {
        const list = BUDDY_MESSAGES[currentSection];
        message.textContent = list[messageIndex % list.length];
        bubble.hidden = false;
    }

    button.addEventListener("click", () => {
        // First click on a section shows the main message; later clicks show the next one
        if (!bubble.hidden) messageIndex += 1;
        showMessage();
        button.classList.remove("is-happy");
        void button.offsetWidth;              // restarts the wiggle animation
        button.classList.add("is-happy");
    });

    closeButton.addEventListener("click", () => {
        bubble.hidden = true;
        button.focus();
    });

    // Buddy never opens by itself. If the bubble is already open, it updates for the new section.
    onSectionChange = () => {
        messageIndex = 0;
        if (!bubble.hidden) showMessage();
    };
}

/* ---------- 8. Submit and rain transition ---------- */
function setupSubmit() {
    const form = $("#predict-form");
    const textarea = $("#skin_description");
    const fileInput = $("#skin_image");
    const hint = $("#form-hint");
    const button = $("#predict-button");
    const label = $("#predict-label");
    const rainLayer = $("#rain-layer");
    const RAIN_DELAY = 900;   // ms the rain plays before the real submit continues
    let isSubmitting = false;

    function createRain() {
        rainLayer.innerHTML = "";
        const colors = ["is-blue", "is-lavender", "is-muted"];
        for (let i = 0; i < 70; i++) {
            const drop = document.createElement("span");
            drop.className = "drop " + colors[i % colors.length];
            drop.style.left = Math.random() * 100 + "%";
            drop.style.height = 14 + Math.random() * 26 + "px";
            drop.style.animationDuration = 0.9 + Math.random() * 1.1 + "s";
            drop.style.animationDelay = -Math.random() * 2 + "s";
            rainLayer.appendChild(drop);
        }
    }

    function resetState() {
        isSubmitting = false;
        document.body.classList.remove("is-analyzing");
        button.classList.remove("is-loading");
        button.removeAttribute("aria-busy");
        label.textContent = "Predict my skin type";
        rainLayer.innerHTML = "";
    }

    form.addEventListener("submit", (event) => {
        if (isSubmitting) { event.preventDefault(); return; }

        // Nothing to analyze: ask for at least one input.
        // (Delete this block if you prefer Flask to handle empty submissions itself.)
        const hasText = textarea.value.trim().length > 0;
        const hasImage = fileInput.files.length > 0;
        if (!hasText && !hasImage) {
            event.preventDefault();
            hint.textContent = "Add a short description of your skin, upload a photo, or do both.";
            hint.hidden = false;
            textarea.focus();
            return;
        }
        hint.hidden = true;

        // Valid: start the animation, then continue with the REAL form submission to Flask
        isSubmitting = true;
        button.classList.add("is-loading");
        button.setAttribute("aria-busy", "true");
        label.textContent = "Analyzing your skin…";

        if (prefersReducedMotion()) return;   // submit normally with no delay

        event.preventDefault();
        createRain();
        document.body.classList.add("is-analyzing");
        window.setTimeout(() => form.submit(), RAIN_DELAY);
    });

    // If the user comes back with the browser's Back button, reset the page
    window.addEventListener("pageshow", (event) => { if (event.persisted) resetState(); });
}

/* ---------- Start everything ---------- */
setupNavigation();
setupSkinCards();
setupHowItWorks();
setupSteps();
setupTextarea();
setupImageUpload();
setupSkinBuddy();
setupSubmit();