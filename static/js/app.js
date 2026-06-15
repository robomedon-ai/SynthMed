// ══════════════════════════════════════════
// Medical Imaging Lab — frontend bootstrap
// PASD (placental MRI) + TopBrain (cerebrovascular)
// WSI/PANDA code removed 2026-06-07
// ══════════════════════════════════════════

// ── View navigation (MRI Viewer / Modules / Results) ──

const ALL_VIEWS = ["home", "mri", "cta", "tofmra", "modules", "results"];

function switchView(view, sub) {
    ALL_VIEWS.forEach(v => {
        const el = document.getElementById(`view-${v}`);
        if (el) el.style.display = (v === view) ? "" : "none";
    });
    document.querySelectorAll(".nav-btn").forEach(b => {
        b.classList.toggle("active", b.dataset.view === view);
    });
    // Trigger lazy init for views that need it
    if (view === "mri"    && typeof initMriViewer === "function") initMriViewer();
    if (view === "cta"    && typeof initCvViewer  === "function") initCvViewer("cta",    "ct");
    if (view === "tofmra" && typeof initCvViewer  === "function") initCvViewer("tofmra", "mr");
    // Optional deep-link into a specific module
    if (view === "modules" && sub && typeof openModule === "function") openModule(sub);
    window.scrollTo(0, 0);
}

document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
});

// Brand logo + home cards navigate
document.querySelectorAll("[data-view].home-card, .brand[data-view]").forEach(el => {
    el.addEventListener("click", () => switchView(el.dataset.view, el.dataset.sub));
});

// Default landing view on load (honor #hash deep-links, e.g. /#results)
document.addEventListener("DOMContentLoaded", () => {
    const [v, sub] = (location.hash || "").replace("#", "").split("/");
    switchView(ALL_VIEWS.includes(v) ? v : "home", sub);
});


// ── Modules view: card open / back ──

const MODULE_TITLES = {
    "pasd-gen":           "PASD MRI Generation",
    "pasd-xmod":          "Cross-Modality Translation (CycleGAN)",
    "cv-mr2ct":           "MRA → CTA Translation (TopBrain)",
    "cv-mask2ct":         "Vessel-Mask → CTA (SPADE, TopBrain)",
    "prostate-t2-adc":    "T2 → ADC Translation (prostate158)",
    "prostate-mask-t2":   "Anatomy Mask → T2 (prostate158)",
    "tstr-seg":           "Segmentation: Realism vs Relevance (TSTR)",
};

let currentModule = null;

function openModule(moduleId) {
    currentModule = moduleId;

    document.getElementById("modules-grid").style.display = "none";
    document.getElementById("module-detail").style.display = "";
    document.getElementById("module-detail-title").textContent =
        MODULE_TITLES[moduleId] || moduleId;

    // All current modules operate on PASD/TopBrain data — no WSI image needed.
    const ctxBox = document.getElementById("module-image-ctx");
    const noImg  = document.getElementById("module-no-image");
    if (ctxBox) ctxBox.style.display = "none";
    if (noImg)  noImg.style.display = "none";

    document.querySelectorAll(".module-panel").forEach(p => p.style.display = "none");
    const panel = document.getElementById(`module-panel-${moduleId}`);
    if (panel) panel.style.display = "";

    // Lazy initializers
    if (moduleId === "pasd-gen"          && typeof initPasdGen        === "function") initPasdGen();
    if (moduleId === "pasd-xmod"         && typeof initPasdXmod       === "function") initPasdXmod();
    if (moduleId === "cv-mr2ct"          && typeof initCvMr2Ct        === "function") initCvMr2Ct();
    if (moduleId === "cv-mask2ct"        && typeof initCvMask2Ct      === "function") initCvMask2Ct();
    if (moduleId === "prostate-t2-adc"   && typeof initProstateT2Adc  === "function") initProstateT2Adc();
    if (moduleId === "prostate-mask-t2"  && typeof initProstateMaskT2 === "function") initProstateMaskT2();
    if (moduleId === "tstr-seg"          && typeof initTstrSeg        === "function") initTstrSeg();
}

function closeModule() {
    currentModule = null;
    document.getElementById("module-detail").style.display = "none";
    document.getElementById("modules-grid").style.display = "";
}

document.querySelectorAll(".module-card").forEach(card => {
    card.addEventListener("click", () => openModule(card.dataset.module));
});

const backBtn = document.getElementById("btn-module-back");
if (backBtn) backBtn.addEventListener("click", closeModule);


// ══════════════════════════════════════════
// MRI Viewer (PASD dataset)
// ══════════════════════════════════════════

let mriInitialized = false;
let mriState = {
    modality: "BTFE",
    subject: null,
    axis: "axial",
    axisLen: 1,
    sliceIdx: 0,
    maskAlpha: 0.45,
    showMask: true,
    displayMode: "both",      // "both" | "image_only" | "mask_only"
    displayMode3D: "both",
    wlAuto: true,
    wl: 128,
    ww: 255,
    info: null,
    outline: false,
    cinePlaying: false,
    cineTimer: null,
    areas: null,
    compareSubject: null,
    subjects: [],
    zoom: 1, panX: 0, panY: 0,
    dragging: false, dragStartX: 0, dragStartY: 0, dragStartPanX: 0, dragStartPanY: 0,
};
let nv = null;

function initMriViewer() {
    mriInitialized = true;
    document.getElementById("mri-modality").addEventListener("change", e => {
        mriState.modality = e.target.value;
        loadMriSubjects();
    });
    document.getElementById("mri-search").addEventListener("input", renderMriSubjectList);
    document.querySelectorAll(".mri-axis-tab").forEach(tab => {
        tab.addEventListener("click", () => switchMriAxis(tab.dataset.axis));
    });
    document.getElementById("mri-slice-slider").addEventListener("input", e => {
        mriState.sliceIdx = parseInt(e.target.value, 10);
        updateMriSlice();
    });
    document.getElementById("mri-alpha-slider").addEventListener("input", e => {
        mriState.maskAlpha = parseFloat(e.target.value);
        document.getElementById("mri-alpha-label").textContent = mriState.maskAlpha.toFixed(2);
        updateMriSlice();
    });
    // 2D display-mode segmented control (Image+GT / Image only / GT only)
    document.querySelectorAll("#mri-display-mode button").forEach(b => {
        b.addEventListener("click", () => setMriDisplayMode(b.dataset.mode));
    });
    const wlSlider = document.getElementById("mri-wl-slider");
    const wwSlider = document.getElementById("mri-ww-slider");
    wlSlider.addEventListener("input", e => {
        mriState.wlAuto = false;
        mriState.wl = parseInt(e.target.value, 10);
        document.getElementById("mri-wl-label").textContent = mriState.wl;
        updateMriSlice();
    });
    wwSlider.addEventListener("input", e => {
        mriState.wlAuto = false;
        mriState.ww = parseInt(e.target.value, 10);
        document.getElementById("mri-ww-label").textContent = mriState.ww;
        updateMriSlice();
    });
    document.getElementById("mri-wl-auto").addEventListener("click", () => {
        mriState.wlAuto = true;
        document.getElementById("mri-wl-label").textContent = "auto";
        document.getElementById("mri-ww-label").textContent = "auto";
        updateMriSlice();
    });
    // 3D display-mode segmented control
    document.querySelectorAll("#mri-3d-display-mode button").forEach(b => {
        b.addEventListener("click", () => setMri3dDisplayMode(b.dataset.mode));
    });
    document.getElementById("mri-3d-render-mode").addEventListener("change", e => {
        if (!nv) return;
        // 0 = MPR, 1 = render
        nv.setSliceType(e.target.value === "volume" ? nv.sliceTypeRender : nv.sliceTypeMultiplanar);
    });
    document.getElementById("mri-3d-reset").addEventListener("click", () => {
        if (nv) nv.scene.crosshairPos = [0.5, 0.5, 0.5];
        if (nv) nv.updateGLVolume();
    });

    // Outline-only mode
    document.getElementById("mri-outline-mode").addEventListener("change", e => {
        mriState.outline = e.target.checked;
        updateMriSlice();
    });

    // Zoom + pan
    document.getElementById("mri-zoom-in").addEventListener("click",  () => mriZoomBy(1.25));
    document.getElementById("mri-zoom-out").addEventListener("click", () => mriZoomBy(1 / 1.25));
    document.getElementById("mri-zoom-fit").addEventListener("click", () => mriZoomFit());
    document.querySelectorAll("#mri-slice-img, #mri-slice-img-b").forEach(img => {
        img.addEventListener("dblclick", () => mriZoomFit());
        img.addEventListener("mousedown", onMriDragStart);
    });
    window.addEventListener("mousemove", onMriDragMove);
    window.addEventListener("mouseup",   onMriDragEnd);

    // Cine play / pause
    document.getElementById("mri-play-btn").addEventListener("click", toggleCine);

    // Download current slice as PNG
    document.getElementById("mri-download-btn").addEventListener("click", () => {
        if (!mriState.subject) return;
        const alpha = mriState.showMask ? mriState.maskAlpha : 0;
        const params = new URLSearchParams({
            mask_alpha: alpha, format: "png", download: "1",
        });
        if (mriState.outline) params.set("outline", "1");
        if (!mriState.wlAuto) {
            params.set("wl", mriState.wl);
            params.set("ww", mriState.ww);
        }
        const url = `/api/mri/${mriState.modality}/${mriState.subject}` +
                    `/slice/${mriState.axis}/${mriState.sliceIdx}?${params}`;
        const a = document.createElement("a");
        a.href = url;
        a.download = "";
        document.body.appendChild(a);
        a.click();
        a.remove();
    });

    // Sparkline click → jump slice
    document.getElementById("mri-area-sparkline").addEventListener("click", e => {
        if (!mriState.areas) return;
        const canvas = e.currentTarget;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const idx = Math.floor((x / rect.width) * mriState.areas.length);
        setMriSliceIdx(Math.max(0, Math.min(mriState.areas.length - 1, idx)));
    });

    // Mouse wheel scrubbing over the 2D stage
    document.getElementById("mri-slice-stage").addEventListener("wheel", e => {
        if (!mriState.subject) return;
        e.preventDefault();
        if (e.shiftKey) {
            // Shift+wheel → scrub slices
            const step = e.deltaY > 0 ? 1 : -1;
            setMriSliceIdx(mriState.sliceIdx + step);
            return;
        }
        // Plain wheel → zoom around cursor
        const factor = e.deltaY > 0 ? 1 / 1.15 : 1.15;
        const img = document.getElementById("mri-slice-img");
        const r = img.getBoundingClientRect();
        mriZoomBy(factor, e.clientX - (r.left + r.width / 2),
                          e.clientY - (r.top  + r.height / 2));
    }, { passive: false });

    // Compare subject selector
    document.getElementById("mri-compare-select").addEventListener("change", e => {
        mriState.compareSubject = e.target.value || null;
        applyMriCompareLayout();
        updateMriSlice();
    });

    // Keyboard shortcuts (only when MRI view is visible)
    document.addEventListener("keydown", onMriKeyDown);

    loadMriSubjects();
}

function applyMriCompareLayout() {
    const paneB = document.getElementById("mri-pane-b");
    const labelA = document.getElementById("mri-pane-a-label");
    if (mriState.compareSubject) {
        paneB.style.display = "";
        labelA.textContent = mriState.subject || "-";
        document.getElementById("mri-pane-b-label").textContent = mriState.compareSubject;
    } else {
        paneB.style.display = "none";
        labelA.textContent = mriState.subject || "-";
    }
}

function populateMriCompareSelect() {
    const sel = document.getElementById("mri-compare-select");
    const prev = sel.value;
    sel.innerHTML = '<option value="">Compare: off</option>';
    for (const s of mriState.subjects) {
        if (s.subject === mriState.subject) continue;  // skip self
        const opt = document.createElement("option");
        opt.value = s.subject;
        opt.textContent = `vs ${s.subject}`;
        sel.appendChild(opt);
    }
    // Restore previous selection if still valid; else reset
    if (prev && mriState.subjects.some(s => s.subject === prev && s.subject !== mriState.subject)) {
        sel.value = prev;
    } else {
        sel.value = "";
        mriState.compareSubject = null;
    }
}

function isMriViewActive() {
    const v = document.getElementById("view-mri");
    return v && v.style.display !== "none";
}

function onMriKeyDown(e) {
    if (!isMriViewActive() || !mriState.subject) return;
    const tag = (e.target.tagName || "").toUpperCase();
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;

    const AXES = ["axial", "sagittal", "coronal"];
    const i = AXES.indexOf(mriState.axis);
    switch (e.key) {
        case "ArrowLeft":  setMriSliceIdx(mriState.sliceIdx - 1); e.preventDefault(); break;
        case "ArrowRight": setMriSliceIdx(mriState.sliceIdx + 1); e.preventDefault(); break;
        case "ArrowUp":    switchMriAxis(AXES[(i + AXES.length - 1) % AXES.length]); e.preventDefault(); break;
        case "ArrowDown":  switchMriAxis(AXES[(i + 1) % AXES.length]); e.preventDefault(); break;
        case "m": case "M": {
            // Cycle Both → Image only → GT only → Both
            const cycle = {both: "image_only", image_only: "mask_only", mask_only: "both"};
            setMriDisplayMode(cycle[mriState.displayMode] || "both");
            e.preventDefault(); break;
        }
        case " ": toggleCine(); e.preventDefault(); break;
    }
}

function setMriSliceIdx(idx) {
    idx = Math.max(0, Math.min(mriState.axisLen - 1, idx));
    if (idx === mriState.sliceIdx) return;
    mriState.sliceIdx = idx;
    document.getElementById("mri-slice-slider").value = idx;
    updateMriSlice();
}

// ── Zoom / pan helpers (MRI 2D slice; syncs both compare panes) ──

const MRI_ZOOM_MIN = 1.0;
const MRI_ZOOM_MAX = 12.0;

function mriApplyTransform() {
    const t = `translate(${mriState.panX}px, ${mriState.panY}px) scale(${mriState.zoom})`;
    ["mri-slice-img", "mri-slice-img-b"].forEach(id => {
        const img = document.getElementById(id);
        if (img) img.style.transform = t;
    });
    const zoomed = mriState.zoom > 1.001;
    document.querySelectorAll("#mri-2d-panel .mri-slice-pane").forEach(p =>
        p.classList.toggle("cv-zoomed", zoomed));
}

function mriZoomBy(factor, cursorOffsetX = 0, cursorOffsetY = 0) {
    const oldZ = mriState.zoom;
    const newZ = Math.max(MRI_ZOOM_MIN, Math.min(MRI_ZOOM_MAX, oldZ * factor));
    if (newZ === oldZ) return;
    const ratio = newZ / oldZ;
    mriState.panX = (mriState.panX - cursorOffsetX) * ratio + cursorOffsetX;
    mriState.panY = (mriState.panY - cursorOffsetY) * ratio + cursorOffsetY;
    mriState.zoom = newZ;
    if (newZ <= 1.001) { mriState.panX = 0; mriState.panY = 0; }
    mriApplyTransform();
}

function mriZoomFit() {
    mriState.zoom = 1; mriState.panX = 0; mriState.panY = 0;
    mriApplyTransform();
}

function onMriDragStart(e) {
    if (mriState.zoom <= 1.001) return;
    mriState.dragging = true;
    mriState.dragStartX = e.clientX; mriState.dragStartY = e.clientY;
    mriState.dragStartPanX = mriState.panX; mriState.dragStartPanY = mriState.panY;
    document.querySelectorAll("#mri-2d-panel .mri-slice-pane").forEach(p =>
        p.classList.add("cv-dragging"));
    e.preventDefault();
}

function onMriDragMove(e) {
    if (!mriState.dragging) return;
    mriState.panX = mriState.dragStartPanX + (e.clientX - mriState.dragStartX);
    mriState.panY = mriState.dragStartPanY + (e.clientY - mriState.dragStartY);
    mriApplyTransform();
}

function onMriDragEnd() {
    if (!mriState.dragging) return;
    mriState.dragging = false;
    document.querySelectorAll("#mri-2d-panel .mri-slice-pane").forEach(p =>
        p.classList.remove("cv-dragging"));
}


function toggleCine() {
    const btn = document.getElementById("mri-play-btn");
    if (mriState.cinePlaying) {
        clearInterval(mriState.cineTimer);
        mriState.cineTimer = null;
        mriState.cinePlaying = false;
        btn.innerHTML = "&#9658;";  // play
        btn.classList.remove("active");
    } else {
        mriState.cinePlaying = true;
        btn.innerHTML = "&#10074;&#10074;";  // pause
        btn.classList.add("active");
        mriState.cineTimer = setInterval(() => {
            const next = (mriState.sliceIdx + 1) % mriState.axisLen;
            setMriSliceIdx(next);
        }, 160);
    }
}

function stopCine() {
    if (mriState.cinePlaying) toggleCine();
}

async function loadAreaSparkline() {
    if (!mriState.subject) return;
    const res = await fetch(`/api/mri/${mriState.modality}/${mriState.subject}` +
                            `/mask_areas?axis=${mriState.axis}`);
    const data = await res.json();
    mriState.areas = data.areas;
    drawAreaSparkline();
}

function drawAreaSparkline() {
    const canvas = document.getElementById("mri-area-sparkline");
    if (!canvas || !mriState.areas) return;
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || canvas.parentElement.clientWidth;
    const cssH = 36;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssW, cssH);

    const areas = mriState.areas;
    const n = areas.length;
    const maxA = Math.max(1, ...areas);
    const barW = cssW / n;
    for (let i = 0; i < n; i++) {
        const h = (areas[i] / maxA) * (cssH - 4);
        ctx.fillStyle = i === mriState.sliceIdx ? "#ff4081" : "#4a5cb8";
        ctx.fillRect(i * barW, cssH - h - 1, Math.max(1, barW - 0.5), h);
    }
    // Current-slice marker line
    const cx = (mriState.sliceIdx + 0.5) * barW;
    ctx.strokeStyle = "#ff80ab";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, 0); ctx.lineTo(cx, cssH);
    ctx.stroke();
}

async function loadMriSubjects() {
    const list = document.getElementById("mri-subject-list");
    list.innerHTML = '<div class="mri-subject-item">Loading...</div>';
    const res = await fetch(`/api/mri/subjects?modality=${mriState.modality}`);
    const data = await res.json();
    mriState.subjects = data.subjects;
    document.getElementById("mri-count").textContent = `${data.subjects.length} subjects`;
    renderMriSubjectList();
    populateMriCompareSelect();
}

function renderMriSubjectList() {
    const list = document.getElementById("mri-subject-list");
    const q = document.getElementById("mri-search").value.trim().toLowerCase();
    const filtered = mriState.subjects.filter(s => !q || s.subject.toLowerCase().includes(q));
    list.innerHTML = "";
    for (const s of filtered) {
        const item = document.createElement("div");
        item.className = "mri-subject-item";
        if (s.subject === mriState.subject) item.classList.add("active");
        const thumbUrl = `/api/mri/${mriState.modality}/${s.subject}/thumb`;
        item.innerHTML = `
            <img loading="lazy" src="${thumbUrl}" alt="thumb">
            <div class="meta">
                <div class="id">${s.subject}</div>
                <div class="details">${s.slice_count} slices (${s.first_slice}-${s.last_slice})</div>
            </div>
            ${s.has_mask ? '<span class="badge-mask">GT</span>' : ''}
        `;
        item.addEventListener("click", () => selectMriSubject(s.subject));
        list.appendChild(item);
    }
}

async function selectMriSubject(subject) {
    stopCine();
    mriState.subject = subject;
    mriState.zoom = 1; mriState.panX = 0; mriState.panY = 0;
    mriApplyTransform();
    document.querySelectorAll("#mri-subject-list .mri-subject-item").forEach(el => {
        el.classList.toggle("active", el.querySelector(".id")?.textContent === subject);
    });

    // Fetch info, populate panel
    const res = await fetch(`/api/mri/${mriState.modality}/${subject}/info`);
    const info = await res.json();
    mriState.info = info;
    document.getElementById("mri-info-subject").textContent = subject;
    document.getElementById("mri-info-modality").textContent = info.modality;
    document.getElementById("mri-info-shape").textContent = `${info.shape[0]}×${info.shape[1]}×${info.shape[2]}`;
    document.getElementById("mri-info-spacing").textContent = info.spacing_mm.map(x => x.toFixed(2)).join(" / ");
    document.getElementById("mri-info-mask").textContent = info.mask_voxel_count.toLocaleString();
    document.getElementById("mri-info-panel").style.display = "";
    document.getElementById("mri-empty").style.display = "none";
    document.getElementById("mri-workspace").style.display = "";

    populateMriCompareSelect();
    applyMriCompareLayout();

    switchMriAxis(mriState.axis);
    await loadMri3D();
}

function switchMriAxis(axis) {
    mriState.axis = axis;
    document.querySelectorAll(".mri-axis-tab").forEach(t => {
        t.classList.toggle("active", t.dataset.axis === axis);
    });
    if (!mriState.info) return;
    const n = mriState.info.axis_lengths[axis];
    mriState.axisLen = n;
    mriState.sliceIdx = Math.floor(n / 2);
    const slider = document.getElementById("mri-slice-slider");
    slider.min = 0;
    slider.max = n - 1;
    slider.value = mriState.sliceIdx;
    mriState.areas = null;  // refetch for new axis
    updateMriSlice();
    loadAreaSparkline();
}

function buildSliceUrl(subject, sliceIdx) {
    const alpha = mriState.maskAlpha;
    const params = new URLSearchParams({ mask_alpha: alpha });
    if (mriState.outline) params.set("outline", "1");
    if (mriState.displayMode !== "both") params.set("mode", mriState.displayMode);
    if (!mriState.wlAuto) {
        params.set("wl", mriState.wl);
        params.set("ww", mriState.ww);
    }
    return `/api/mri/${mriState.modality}/${subject}` +
           `/slice/${mriState.axis}/${sliceIdx}?${params}`;
}

function setMriDisplayMode(mode) {
    mriState.displayMode = mode;
    document.querySelectorAll("#mri-display-mode button").forEach(b =>
        b.classList.toggle("active", b.dataset.mode === mode));
    // Mask alpha only meaningful in "both" mode
    const alphaRow = document.getElementById("mri-alpha-row");
    if (alphaRow) alphaRow.style.opacity = (mode === "both") ? "1" : "0.4";
    const alphaSlider = document.getElementById("mri-alpha-slider");
    if (alphaSlider) alphaSlider.disabled = (mode !== "both");
    updateMriSlice();
}

function setMri3dDisplayMode(mode) {
    mriState.displayMode3D = mode;
    document.querySelectorAll("#mri-3d-display-mode button").forEach(b =>
        b.classList.toggle("active", b.dataset.mode === mode));
    if (nv && nv.volumes && nv.volumes.length > 1) {
        const imgOp  = (mode === "mask_only") ? 0 : 1;
        const maskOp = (mode === "image_only") ? 0 : (mode === "mask_only" ? 1 : 0.55);
        nv.setOpacity(0, imgOp);
        nv.setOpacity(1, maskOp);
    }
}

function updateMriSlice() {
    if (!mriState.subject) return;
    document.getElementById("mri-slice-img").src =
        buildSliceUrl(mriState.subject, mriState.sliceIdx);
    document.getElementById("mri-slice-label").textContent =
        `${mriState.sliceIdx + 1} / ${mriState.axisLen}`;

    if (mriState.compareSubject) {
        const cmp = mriState.subjects.find(s => s.subject === mriState.compareSubject);
        // Clamp compare slice idx to its own slice count for the current axis.
        // We don't have per-axis lengths cached for the compare subject, but for
        // axial the slice_count is exact; for sag/cor it's always 512 since all
        // slices are 512x512.
        const maxIdx = mriState.axis === "axial"
            ? (cmp ? cmp.slice_count - 1 : mriState.axisLen - 1)
            : (mriState.axisLen - 1);
        const cmpIdx = Math.min(mriState.sliceIdx, maxIdx);
        document.getElementById("mri-slice-img-b").src =
            buildSliceUrl(mriState.compareSubject, cmpIdx);
    }

    drawAreaSparkline();
    syncTo3DCrosshair();
}

// Sync flags to break feedback loops between 2D slider and 3D crosshair
let mriSyncingFromNv = false;
let mriSyncingFromUi = false;

async function loadMri3D() {
    const status = document.getElementById("mri-3d-status");
    status.textContent = "Loading volume...";
    status.style.display = "";

    const baseUrl = `/api/mri/${mriState.modality}/${mriState.subject}`;
    const imgUrl  = `${baseUrl}/image.nii.gz`;
    const maskUrl = `${baseUrl}/mask.nii.gz`;

    if (!nv) {
        if (typeof niivue === "undefined") {
            status.textContent = "NiiVue failed to load (check network).";
            return;
        }
        nv = new niivue.Niivue({
            backColor: [0, 0, 0, 1],
            show3Dcrosshair: true,
            crosshairColor: [1, 0.5, 0, 1],
            onLocationChange: (data) => {
                if (mriSyncingFromUi || !mriState.info || !data || !data.vox) return;
                const [vx, vy, vz] = data.vox;
                const [H, W, _D] = mriState.info.shape;
                let idx;
                if (mriState.axis === "axial")    idx = vz;
                else if (mriState.axis === "sagittal") idx = vx;
                else                              idx = (H - 1) - vy;  // coronal (Y flipped on export)
                idx = Math.round(idx);
                if (idx === mriState.sliceIdx) return;
                mriSyncingFromNv = true;
                setMriSliceIdx(idx);
                mriSyncingFromNv = false;
            },
        });
        nv.attachToCanvas(document.getElementById("mri-3d-canvas"));
    }

    try {
        await nv.loadVolumes([
            { url: imgUrl, colormap: "gray", opacity: 1.0 },
            { url: maskUrl, colormap: "red", opacity: 0.55,
              cal_min: 0.5, cal_max: 1.0 },
        ]);
        const mode = document.getElementById("mri-3d-render-mode").value;
        nv.setSliceType(mode === "volume" ? nv.sliceTypeRender : nv.sliceTypeMultiplanar);
        // Apply the user's persisted display-mode choice
        setMri3dDisplayMode(mriState.displayMode3D);
        status.style.display = "none";
        syncTo3DCrosshair();  // initial crosshair from current slice
    } catch (err) {
        console.error(err);
        status.textContent = "Failed to load 3D volume: " + err.message;
    }
}

function syncTo3DCrosshair() {
    if (!nv || !nv.volumes || !nv.volumes.length || !mriState.info) return;
    if (mriSyncingFromNv) return;
    const [H, W, D] = mriState.info.shape;
    // NiiVue volume after our export: dims (W, H_flipped, D), so frac order is [W, H, D].
    let frac = (nv.scene && nv.scene.crosshairPos)
        ? [...nv.scene.crosshairPos] : [0.5, 0.5, 0.5];
    if (mriState.axis === "axial")    frac[2] = (mriState.sliceIdx + 0.5) / D;
    else if (mriState.axis === "sagittal") frac[0] = (mriState.sliceIdx + 0.5) / W;
    else /* coronal */                frac[1] = (((H - 1 - mriState.sliceIdx)) + 0.5) / H;
    mriSyncingFromUi = true;
    nv.scene.crosshairPos = frac;
    if (typeof nv.drawScene === "function") nv.drawScene();
    mriSyncingFromUi = false;
}


// ══════════════════════════════════════════
// CTA / TOF-MRA Viewer (TopBrain dataset)
// Shared init for both tabs — call with (prefix, modality).
// ══════════════════════════════════════════

const _cvViewers = {};   // {prefix: state}
const _cvNv = {};        // {prefix: NiiVue instance}

function initCvViewer(prefix, modality) {
    if (_cvViewers[prefix]?.initialized) return;
    const state = {
        prefix, modality,
        subjects: [],
        subject: null,
        info: null,
        axis: "axial",
        axisLen: 1,
        sliceIdx: 128,
        maskAlpha: 0.5,
        showMask: true,
        displayMode: "both",      // "both" | "image_only" | "mask_only"
        displayMode3D: "both",
        outline: false,
        zoom: 1, panX: 0, panY: 0,
        dragging: false, dragStartX: 0, dragStartY: 0, dragStartPanX: 0, dragStartPanY: 0,
        wl: modality === "ct" ? 200 : null,
        ww: modality === "ct" ? 700 : null,
        windowPreset: modality === "ct" ? "vessel" : "auto",
        areas: null,
        cinePlaying: false,
        cineTimer: null,
        labels: null,
        labelById: null,
        initialized: true,
    };
    _cvViewers[prefix] = state;

    const $ = sel => document.getElementById(`${prefix}-${sel}`);

    // ── Search ──
    $("search").addEventListener("input", () => renderCvSubjectList(prefix));

    // ── CT window preset dropdown ──
    if (modality === "ct") {
        $("window").addEventListener("change", e => {
            state.windowPreset = e.target.value;
            applyCvWindowPreset(prefix);
            updateCvSlice(prefix);
        });
    }

    // ── Axis tabs ──
    document.querySelectorAll(`#view-${prefix} .mri-axis-tab`).forEach(tab => {
        tab.addEventListener("click", () => switchCvAxis(prefix, tab.dataset.axis));
    });

    // ── Slice slider + sparkline ──
    $("slice-slider").addEventListener("input", e => {
        state.sliceIdx = parseInt(e.target.value, 10);
        updateCvSlice(prefix);
    });
    $("area-sparkline").addEventListener("click", e => onCvSparklineClick(prefix, e));

    // ── 2D display mode (segmented) + alpha + outline ──
    document.querySelectorAll(`#${prefix}-display-mode button`).forEach(b => {
        b.addEventListener("click", () => setCvDisplayMode(prefix, b.dataset.mode));
    });
    $("alpha-slider").addEventListener("input", e => {
        state.maskAlpha = parseFloat(e.target.value);
        $("alpha-label").textContent = state.maskAlpha.toFixed(2);
        updateCvSlice(prefix);
    });
    $("outline-mode").addEventListener("change", e => {
        state.outline = e.target.checked;
        updateCvSlice(prefix);
    });

    // ── WL/WW sliders (CT only) ──
    if (modality === "ct") {
        $("wl-slider").addEventListener("input", e => {
            state.wl = parseInt(e.target.value, 10);
            state.windowPreset = "manual";
            $("window").value = "manual";
            $("wl-label").textContent = state.wl;
            updateCvSlice(prefix);
        });
        $("ww-slider").addEventListener("input", e => {
            state.ww = parseInt(e.target.value, 10);
            state.windowPreset = "manual";
            $("window").value = "manual";
            $("ww-label").textContent = state.ww;
            updateCvSlice(prefix);
        });
    }

    // ── 3D controls ──
    document.querySelectorAll(`#${prefix}-3d-display-mode button`).forEach(b => {
        b.addEventListener("click", () => setCv3dDisplayMode(prefix, b.dataset.mode));
    });
    $("3d-render-mode").addEventListener("change", e => {
        const nv = _cvNv[prefix];
        if (!nv) return;
        nv.setSliceType(e.target.value === "volume"
            ? nv.sliceTypeRender : nv.sliceTypeMultiplanar);
    });
    $("3d-reset").addEventListener("click", () => {
        const nv = _cvNv[prefix];
        if (nv) { nv.scene.crosshairPos = [0.5, 0.5, 0.5]; nv.updateGLVolume(); }
    });

    // ── Cine play + download ──
    $("play-btn").addEventListener("click", () => toggleCvCine(prefix));
    $("download-btn").addEventListener("click", () => downloadCvSlice(prefix));

    // ── Zoom + pan ──
    $("zoom-in").addEventListener("click",  () => cvZoomBy(prefix, 1.25));
    $("zoom-out").addEventListener("click", () => cvZoomBy(prefix, 1 / 1.25));
    $("zoom-fit").addEventListener("click", () => cvZoomFit(prefix));
    const sliceImg = $("slice-img");
    sliceImg.addEventListener("dblclick", () => cvZoomFit(prefix));
    sliceImg.addEventListener("mousedown", e => onCvDragStart(prefix, e));
    window.addEventListener("mousemove",   e => onCvDragMove(prefix, e));
    window.addEventListener("mouseup",     ()  => onCvDragEnd(prefix));

    // ── Keyboard + wheel (wheel handles BOTH zoom (with ctrl) and scrub) ──
    $("2d-panel").addEventListener("keydown", e => onCvKey(prefix, e));
    $("2d-panel").addEventListener("wheel", e => onCvWheel(prefix, e));

    // ── Load data ──
    loadCvSubjects(prefix);
    loadCvLegend(prefix);
}


async function loadCvSubjects(prefix) {
    const state = _cvViewers[prefix];
    try {
        const r = await fetch("/api/cv_view/subjects");
        const d = await r.json();
        state.subjects = d.subjects;
        renderCvSubjectList(prefix);
    } catch (e) {
        document.getElementById(`${prefix}-count`).textContent = "Failed to load patients";
    }
}


async function loadCvLegend(prefix) {
    const state = _cvViewers[prefix];
    try {
        const r = await fetch(`/api/cv_view/labels/${state.modality}`);
        const d = await r.json();
        state.labels = d.labels;
        state.labelById = Object.fromEntries(d.labels.map(l => [l.id, l]));
        renderCvLegend(prefix);
    } catch (e) { /* non-fatal */ }
}


function renderCvLegend(prefix) {
    const state = _cvViewers[prefix];
    if (!state.labels) return;
    document.getElementById(`${prefix}-legend-count`).textContent =
        `${state.labels.length} labels`;
    const list = document.getElementById(`${prefix}-legend-list`);
    list.innerHTML = state.labels.map(l => {
        const [r, g, b] = l.color;
        return `<div class="cv-legend-row" data-label-id="${l.id}">
            <span class="cv-legend-swatch" style="background: rgb(${r},${g},${b})"></span>
            <span class="cv-legend-id">${l.id}</span>
            <span class="cv-legend-name">${l.name}</span>
        </div>`;
    }).join("");
    // Dim rows for labels not present in current subject
    refreshCvLegendActive(prefix);
}


function refreshCvLegendActive(prefix) {
    const state = _cvViewers[prefix];
    const present = new Set(state.info?.unique_labels ?? []);
    document.querySelectorAll(`#${prefix}-legend-list .cv-legend-row`).forEach(row => {
        const id = parseInt(row.dataset.labelId, 10);
        row.classList.toggle("cv-legend-absent", state.info && !present.has(id));
    });
}


function renderCvSubjectList(prefix) {
    const state = _cvViewers[prefix];
    const list = document.getElementById(`${prefix}-subject-list`);
    const q = (document.getElementById(`${prefix}-search`).value || "").trim().toLowerCase();
    const filtered = state.subjects.filter(s => !q || s.patient.toLowerCase().includes(q));
    document.getElementById(`${prefix}-count`).textContent =
        `${filtered.length} patient${filtered.length === 1 ? "" : "s"}`;
    list.innerHTML = filtered.map(s => `
        <div class="image-item cv-subject-item${state.subject === s.patient ? " active" : ""}"
             data-patient="${s.patient}">
            <img class="thumb" src="/api/cv_view/${state.modality}/${s.patient}/thumb" alt="">
            <div class="details">
                <div class="id">sub-${s.patient}</div>
                <div class="meta">${s.n_slices} slices w/ tissue</div>
            </div>
        </div>
    `).join("");
    list.querySelectorAll(".cv-subject-item").forEach(item => {
        item.addEventListener("click", () => selectCvSubject(prefix, item.dataset.patient));
    });
}


async function selectCvSubject(prefix, patient) {
    const state = _cvViewers[prefix];
    state.subject = patient;
    state.zoom = 1; state.panX = 0; state.panY = 0;
    cvApplyTransform(prefix);
    renderCvSubjectList(prefix);

    document.getElementById(`${prefix}-empty`).style.display = "none";
    document.getElementById(`${prefix}-workspace`).style.display = "";
    document.getElementById(`${prefix}-info-panel`).style.display = "";

    // Fetch volume info
    const r = await fetch(`/api/cv_view/${state.modality}/${patient}/info`);
    const info = await r.json();
    state.info = info;
    state.axisLen = info.axis_lengths[state.axis];
    state.sliceIdx = Math.floor(state.axisLen / 2);

    document.getElementById(`${prefix}-info-subject`).textContent = `sub-${patient}`;
    document.getElementById(`${prefix}-info-shape`).textContent =
        `${info.shape[0]} × ${info.shape[1]} × ${info.shape[2]}`;
    document.getElementById(`${prefix}-info-spacing`).textContent =
        info.spacing_mm.map(x => x.toFixed(2)).join(" × ");
    document.getElementById(`${prefix}-info-mask`).textContent =
        info.mask_voxel_count.toLocaleString();
    document.getElementById(`${prefix}-info-labels`).textContent =
        info.unique_labels.length + " of " + (state.labels?.length ?? "—");

    document.getElementById(`${prefix}-slice-slider`).max = state.axisLen - 1;
    document.getElementById(`${prefix}-slice-slider`).value = state.sliceIdx;

    refreshCvLegendActive(prefix);
    await loadCvMaskAreas(prefix);
    updateCvSlice(prefix);
    loadCvVolume3D(prefix);

    // Focus 2D panel for keyboard shortcuts
    document.getElementById(`${prefix}-2d-panel`).focus();
}


async function loadCvMaskAreas(prefix) {
    const state = _cvViewers[prefix];
    try {
        const r = await fetch(`/api/cv_view/${state.modality}/${state.subject}/mask_areas?axis=${state.axis}`);
        const d = await r.json();
        state.areas = d.areas;
        drawCvSparkline(prefix);
    } catch (e) { state.areas = null; }
}


function drawCvSparkline(prefix) {
    const state = _cvViewers[prefix];
    const canvas = document.getElementById(`${prefix}-area-sparkline`);
    const ctx = canvas.getContext("2d");
    const w = canvas.clientWidth;
    const h = canvas.height;
    canvas.width = w;
    ctx.clearRect(0, 0, w, h);
    if (!state.areas) return;
    const n = state.areas.length;
    const mx = Math.max(...state.areas) || 1;
    ctx.fillStyle = "#a78bfa";
    for (let i = 0; i < n; i++) {
        const x = (i / n) * w;
        const barW = Math.max(1, w / n);
        const barH = (state.areas[i] / mx) * (h - 2);
        ctx.fillRect(x, h - barH, barW, barH);
    }
    // Cursor
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    const cx = (state.sliceIdx / n) * w;
    ctx.beginPath();
    ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
    ctx.stroke();
}


function onCvSparklineClick(prefix, e) {
    const state = _cvViewers[prefix];
    if (!state.areas) return;
    const canvas = document.getElementById(`${prefix}-area-sparkline`);
    const r = canvas.getBoundingClientRect();
    const frac = (e.clientX - r.left) / r.width;
    state.sliceIdx = Math.max(0, Math.min(state.axisLen - 1,
        Math.round(frac * state.axisLen)));
    document.getElementById(`${prefix}-slice-slider`).value = state.sliceIdx;
    updateCvSlice(prefix);
}


function applyCvWindowPreset(prefix) {
    const state = _cvViewers[prefix];
    if (state.modality !== "ct") return;
    const PRESETS = {
        vessel: {wl: 200, ww: 700},
        brain:  {wl: 40,  ww: 80},
        bone:   {wl: 600, ww: 1500},
        wide:   {wl: 50,  ww: 1200},
    };
    if (state.windowPreset === "auto") {
        state.wl = null; state.ww = null;
        document.getElementById(`${prefix}-wl-label`).textContent = "auto";
        document.getElementById(`${prefix}-ww-label`).textContent = "auto";
        return;
    }
    if (state.windowPreset === "manual") return;
    const p = PRESETS[state.windowPreset];
    if (!p) return;
    state.wl = p.wl; state.ww = p.ww;
    document.getElementById(`${prefix}-wl-slider`).value = p.wl;
    document.getElementById(`${prefix}-ww-slider`).value = p.ww;
    document.getElementById(`${prefix}-wl-label`).textContent = p.wl;
    document.getElementById(`${prefix}-ww-label`).textContent = p.ww;
}


function switchCvAxis(prefix, axis) {
    const state = _cvViewers[prefix];
    state.axis = axis;
    state.axisLen = state.info?.axis_lengths?.[axis] ?? 256;
    state.sliceIdx = Math.min(state.sliceIdx, state.axisLen - 1);
    document.querySelectorAll(`#view-${prefix} .mri-axis-tab`).forEach(t =>
        t.classList.toggle("active", t.dataset.axis === axis));
    document.getElementById(`${prefix}-slice-slider`).max = state.axisLen - 1;
    document.getElementById(`${prefix}-slice-slider`).value = state.sliceIdx;
    loadCvMaskAreas(prefix);
    updateCvSlice(prefix);
}


function updateCvSlice(prefix) {
    const state = _cvViewers[prefix];
    if (!state.subject) return;
    const q = new URLSearchParams({mask_alpha: state.maskAlpha});
    if (state.wl != null && state.ww != null) {
        q.set("wl", state.wl); q.set("ww", state.ww);
    }
    if (state.outline) q.set("outline", "1");
    if (state.displayMode !== "both") q.set("mode", state.displayMode);
    const url = `/api/cv_view/${state.modality}/${state.subject}/slice/${state.axis}/${state.sliceIdx}?${q}`;
    document.getElementById(`${prefix}-slice-img`).src = url;
    document.getElementById(`${prefix}-slice-label`).textContent =
        `${state.sliceIdx} / ${state.axisLen - 1}`;
    document.getElementById(`${prefix}-pane-a-label`).textContent =
        `sub-${state.subject} · ${state.axis} · ${state.sliceIdx}`;
    drawCvSparkline(prefix);
}


function setCvDisplayMode(prefix, mode) {
    const state = _cvViewers[prefix];
    state.displayMode = mode;
    document.querySelectorAll(`#${prefix}-display-mode button`).forEach(b =>
        b.classList.toggle("active", b.dataset.mode === mode));
    // Dim alpha control when not in "both" mode
    const alphaRow = document.getElementById(`${prefix}-alpha-row`);
    if (alphaRow) alphaRow.style.opacity = (mode === "both") ? "1" : "0.4";
    const alphaSlider = document.getElementById(`${prefix}-alpha-slider`);
    if (alphaSlider) alphaSlider.disabled = (mode !== "both");
    updateCvSlice(prefix);
}


function setCv3dDisplayMode(prefix, mode) {
    const state = _cvViewers[prefix];
    state.displayMode3D = mode;
    document.querySelectorAll(`#${prefix}-3d-display-mode button`).forEach(b =>
        b.classList.toggle("active", b.dataset.mode === mode));
    const nv = _cvNv[prefix];
    if (nv && nv.volumes && nv.volumes.length > 1) {
        const imgOp  = (mode === "mask_only") ? 0 : 1;
        const maskOp = (mode === "image_only") ? 0 : (mode === "mask_only" ? 1 : 0.65);
        nv.setOpacity(0, imgOp);
        nv.setOpacity(1, maskOp);
    }
}


function downloadCvSlice(prefix) {
    const state = _cvViewers[prefix];
    if (!state.subject) return;
    const q = new URLSearchParams({mask_alpha: state.maskAlpha,
                                    format: "png", download: "1"});
    if (state.wl != null && state.ww != null) {
        q.set("wl", state.wl); q.set("ww", state.ww);
    }
    if (state.outline) q.set("outline", "1");
    if (state.displayMode !== "both") q.set("mode", state.displayMode);
    window.location.href = `/api/cv_view/${state.modality}/${state.subject}/slice/${state.axis}/${state.sliceIdx}?${q}`;
}


function toggleCvCine(prefix) {
    const state = _cvViewers[prefix];
    state.cinePlaying = !state.cinePlaying;
    document.getElementById(`${prefix}-play-btn`).innerHTML =
        state.cinePlaying ? "&#9208;" : "&#9658;";
    if (state.cinePlaying) {
        state.cineTimer = setInterval(() => {
            state.sliceIdx = (state.sliceIdx + 1) % state.axisLen;
            document.getElementById(`${prefix}-slice-slider`).value = state.sliceIdx;
            updateCvSlice(prefix);
        }, 90);
    } else if (state.cineTimer) {
        clearInterval(state.cineTimer); state.cineTimer = null;
    }
}


function onCvKey(prefix, e) {
    const state = _cvViewers[prefix];
    const axes = ["axial", "sagittal", "coronal"];
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        const step = e.key === "ArrowRight" ? 1 : -1;
        state.sliceIdx = Math.max(0, Math.min(state.axisLen - 1, state.sliceIdx + step));
        document.getElementById(`${prefix}-slice-slider`).value = state.sliceIdx;
        updateCvSlice(prefix);
        e.preventDefault();
    } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        const i = axes.indexOf(state.axis);
        const next = axes[(i + (e.key === "ArrowDown" ? 1 : axes.length - 1)) % axes.length];
        switchCvAxis(prefix, next);
        e.preventDefault();
    } else if (e.key === "m" || e.key === "M") {
        // Cycle Both → Image only → GT only → Both
        const cycle = {both: "image_only", image_only: "mask_only", mask_only: "both"};
        setCvDisplayMode(prefix, cycle[state.displayMode] || "both");
    } else if (e.key === " ") {
        toggleCvCine(prefix);
        e.preventDefault();
    }
}


function onCvWheel(prefix, e) {
    e.preventDefault();
    const state = _cvViewers[prefix];
    if (e.shiftKey) {
        // Shift+wheel → scrub slices
        const step = e.deltaY > 0 ? 1 : -1;
        state.sliceIdx = Math.max(0, Math.min(state.axisLen - 1, state.sliceIdx + step));
        document.getElementById(`${prefix}-slice-slider`).value = state.sliceIdx;
        updateCvSlice(prefix);
        return;
    }
    // Plain wheel → zoom around cursor
    const factor = e.deltaY > 0 ? 1 / 1.15 : 1.15;
    const img = document.getElementById(`${prefix}-slice-img`);
    const r = img.getBoundingClientRect();
    cvZoomBy(prefix, factor, e.clientX - (r.left + r.width / 2),
                              e.clientY - (r.top  + r.height / 2));
}


// ── Zoom / pan helpers (2D slice) ──

const CV_ZOOM_MIN = 1.0;
const CV_ZOOM_MAX = 12.0;

function cvApplyTransform(prefix) {
    const state = _cvViewers[prefix];
    const img = document.getElementById(`${prefix}-slice-img`);
    if (!img) return;
    // Order matters: translate first, then scale, so pan is in display units
    img.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
    const pane = img.closest(".mri-slice-pane");
    if (pane) pane.classList.toggle("cv-zoomed", state.zoom > 1.001);
}

function cvZoomBy(prefix, factor, cursorOffsetX = 0, cursorOffsetY = 0) {
    const state = _cvViewers[prefix];
    const oldZoom = state.zoom;
    const newZoom = Math.max(CV_ZOOM_MIN, Math.min(CV_ZOOM_MAX, oldZoom * factor));
    if (newZoom === oldZoom) return;
    // Keep the point under the cursor fixed during zoom.
    // pan' = pan + (cursorOffset - pan) * (1 - newZoom/oldZoom)
    const ratio = newZoom / oldZoom;
    state.panX = (state.panX - cursorOffsetX) * ratio + cursorOffsetX;
    state.panY = (state.panY - cursorOffsetY) * ratio + cursorOffsetY;
    state.zoom = newZoom;
    if (newZoom <= 1.001) { state.panX = 0; state.panY = 0; }
    cvApplyTransform(prefix);
}

function cvZoomFit(prefix) {
    const state = _cvViewers[prefix];
    state.zoom = 1; state.panX = 0; state.panY = 0;
    cvApplyTransform(prefix);
}

function onCvDragStart(prefix, e) {
    const state = _cvViewers[prefix];
    if (state.zoom <= 1.001) return;
    state.dragging = true;
    state.dragStartX = e.clientX; state.dragStartY = e.clientY;
    state.dragStartPanX = state.panX; state.dragStartPanY = state.panY;
    const pane = document.getElementById(`${prefix}-slice-img`).closest(".mri-slice-pane");
    if (pane) pane.classList.add("cv-dragging");
    e.preventDefault();
}

function onCvDragMove(prefix, e) {
    const state = _cvViewers[prefix];
    if (!state.dragging) return;
    state.panX = state.dragStartPanX + (e.clientX - state.dragStartX);
    state.panY = state.dragStartPanY + (e.clientY - state.dragStartY);
    cvApplyTransform(prefix);
}

function onCvDragEnd(prefix) {
    const state = _cvViewers[prefix];
    if (!state.dragging) return;
    state.dragging = false;
    const pane = document.getElementById(`${prefix}-slice-img`).closest(".mri-slice-pane");
    if (pane) pane.classList.remove("cv-dragging");
}


async function loadCvVolume3D(prefix) {
    const state = _cvViewers[prefix];
    const statusEl = document.getElementById(`${prefix}-3d-status`);
    statusEl.textContent = "Loading 3D volume…";
    statusEl.style.display = "";
    try {
        if (!window.niivue) {
            statusEl.textContent = "NiiVue not available";
            return;
        }
        if (!_cvNv[prefix]) {
            _cvNv[prefix] = new niivue.Niivue({
                show3Dcrosshair: true,
                backColor: [0.06, 0.07, 0.1, 1],
                isColorbar: false,
            });
            await _cvNv[prefix].attachTo(`${prefix}-3d-canvas`);
            _cvNv[prefix].setSliceType(_cvNv[prefix].sliceTypeRender);
        }
        const nv = _cvNv[prefix];
        const imageUrl = `/api/cv_view/${state.modality}/${state.subject}/image.nii.gz`;
        const maskUrl  = `/api/cv_view/${state.modality}/${state.subject}/mask.nii.gz`;
        await nv.loadVolumes([
            {url: imageUrl, colormap: "gray", opacity: 1},
            {url: maskUrl,  colormap: "warm", cal_min: 0.5, cal_max: 42, opacity: 0.65},
        ]);
        // Apply user's persisted display-mode choice
        setCv3dDisplayMode(prefix, state.displayMode3D);
        statusEl.style.display = "none";
    } catch (e) {
        statusEl.textContent = "3D load failed: " + e.message;
    }
}


// ══════════════════════════════════════════
// PASD MRI Generation module
// ══════════════════════════════════════════

// Static metadata for every generator. Numbers from REPORT.md (BTFE eval).
const PASD_MODEL_META = {
    pix2pix: {
        full_name: "Pix2pix",
        subtitle: "Conditional GAN, mask → image",
        paper: "Isola et al., CVPR 2017",
        params: "54 M generator + 2.8 M discriminator",
        training: "60 epochs · AdamW · L1 (×100) + GAN",
        stochastic: false,
        metrics: [
            {label: "FID",       value: "108.81", note: "best"},
            {label: "Mask Dice", value: "0.926",  note: "best"},
            {label: "SSIM",      value: "0.417",  note: "best"},
            {label: "PSNR",      value: "15.77 dB"},
        ],
        verdict: "Best paired pixel match; deterministic outputs.",
    },
    spade: {
        full_name: "SPADE-GAN v2",
        subtitle: "Spatially-adaptive normalization",
        paper: "Park et al., CVPR 2019",
        params: "24 M generator + 2.8 M discriminator",
        training: "200 epochs · L1=0 · hinge GAN + VGG + FM",
        stochastic: true,
        metrics: [
            {label: "FID",       value: "222.88"},
            {label: "Mask Dice", value: "0.817 ± 0.22", note: "high variance"},
            {label: "SSIM",      value: "0.312"},
            {label: "NN-LPIPS",  value: "0.326",  note: "best privacy"},
        ],
        verdict: "Best privacy of any model; stochastic samples can drift off-mask.",
    },
    ldm: {
        full_name: "Latent Diffusion Model",
        subtitle: "SD-VAE + conditional UNet, v-prediction",
        paper: "Rombach et al., CVPR 2022",
        params: "101 M UNet + 84 M VAE",
        training: "30 k steps · DPMSolver++ at sample · CFG p=0.1",
        stochastic: true,
        metrics: [
            {label: "FID",       value: "118.91", note: "g=1.5"},
            {label: "Mask Dice", value: "0.911"},
            {label: "SSIM",      value: "0.370"},
            {label: "PSNR",      value: "13.57 dB"},
        ],
        verdict: "Realistic stochastic samples; second-best mask faithfulness.",
    },
    cyclegan: {
        full_name: "CycleGAN BTFE ↔ TSE",
        subtitle: "Unpaired cross-modality translation",
        paper: "Zhu et al., ICCV 2017",
        params: "11.4 M generator × 2 + 2.8 M discriminator × 2",
        training: "ep 30 (rolled back; final ckpt had D-collapse)",
        stochastic: false,
        metrics: [
            {label: "FID BTFE→TSE",  value: "115.05"},
            {label: "FID TSE→BTFE",  value: "126.30"},
            {label: "Cycle L1",      value: "0.013",  note: "1.3 % round-trip error"},
            {label: "Identity L1",   value: "0.010"},
        ],
        verdict: "🏆 Best TSTR augmentation. Outperforms mask-conditional synth at every data size.",
    },
};

function renderPasdModelInfo(elId, modelKey) {
    const el = document.getElementById(elId);
    if (!el) return;
    const m = PASD_MODEL_META[modelKey];
    if (!m) { el.innerHTML = ""; return; }
    const metricChips = m.metrics.map(x =>
        `<span class="pasd-info-metric"><b>${x.label}</b> ${x.value}` +
        (x.note ? ` <em>(${x.note})</em>` : "") + `</span>`
    ).join("");
    el.innerHTML = `
        <div class="pasd-info-header">
            <div>
                <span class="pasd-info-name">${m.full_name}</span>
                <span class="pasd-info-sub">${m.subtitle}</span>
            </div>
            <span class="pasd-info-paper">${m.paper}</span>
        </div>
        <div class="pasd-info-row"><span class="pasd-info-label">Params</span> ${m.params}</div>
        <div class="pasd-info-row"><span class="pasd-info-label">Training</span> ${m.training}</div>
        <div class="pasd-info-row"><span class="pasd-info-label">Sampling</span> ${m.stochastic ? "stochastic (different output each run)" : "deterministic"}</div>
        <div class="pasd-info-metrics">${metricChips}</div>
        <div class="pasd-info-verdict">${m.verdict}</div>
    `;
}

let pasdGenInitialized = false;
let pasdGenState = {
    source: "dataset",
    modality: "BTFE",
    subject: null,
    sliceIdx: null,
    uploadedMaskB64: null,
    subjectsByMod: {},  // modality -> [{subject, slice_count, first_slice, last_slice, ...}]
    status: null,
};

async function initPasdGen() {
    if (!pasdGenInitialized) {
        pasdGenInitialized = true;
        wirePasdGenEvents();
        await loadPasdGenStatus();
        await loadPasdGenSubjects(pasdGenState.modality);
    }
    refreshPasdGenModelStatus();
    applyPasdModelControlsVisibility();
    renderPasdModelInfo("pasd-model-info",
        document.getElementById("pasd-gen-model").value);
}

function wirePasdGenEvents() {
    document.querySelectorAll(".pasd-source-tab").forEach(t => {
        t.addEventListener("click", () => switchPasdSource(t.dataset.pasdSource));
    });
    document.getElementById("pasd-gen-modality").addEventListener("change", e => {
        pasdGenState.modality = e.target.value;
        refreshPasdGenModelStatus();
        loadPasdGenSubjects(e.target.value);
    });
    document.getElementById("pasd-gen-subject").addEventListener("change", e => {
        pasdGenState.subject = e.target.value;
        populatePasdSliceSelect();
    });
    document.getElementById("pasd-gen-slice").addEventListener("change", e => {
        pasdGenState.sliceIdx = parseInt(e.target.value, 10);
        renderPasdMaskPreview();
    });
    document.getElementById("pasd-gen-upload").addEventListener("change", onPasdUploadChange);
    document.getElementById("pasd-gen-btn").addEventListener("click", runPasdGen);
    document.getElementById("pasd-compare-btn").addEventListener("click", runPasdCompareAll);
    document.getElementById("pasd-gen-model").addEventListener("change", () => {
        refreshPasdGenModelStatus();
        applyPasdModelControlsVisibility();
        renderPasdModelInfo("pasd-model-info",
            document.getElementById("pasd-gen-model").value);
    });

    // Download-all: hit the session ZIP endpoint
    document.getElementById("pasd-gen-download-all").addEventListener("click", () => {
        const sid = pasdGenState.lastSessionId;
        if (!sid) {
            alert("Generate some samples first.");
            return;
        }
        const a = document.createElement("a");
        a.href = `/api/pasd_gen/download_session/${sid}`;
        a.download = "";
        document.body.appendChild(a); a.click(); a.remove();
    });

    // Lightbox close handlers (click backdrop, click X, press Esc)
    const lb = document.getElementById("pasd-lightbox");
    document.getElementById("pasd-lightbox-close").addEventListener("click", closePasdLightbox);
    lb.querySelector(".pasd-lightbox-backdrop").addEventListener("click", closePasdLightbox);
    document.addEventListener("keydown", e => {
        if (e.key === "Escape" && lb.style.display !== "none") closePasdLightbox();
    });
}

function openPasdLightbox(src, downloadHref, caption, fname) {
    const lb = document.getElementById("pasd-lightbox");
    document.getElementById("pasd-lightbox-img").src = src;
    document.getElementById("pasd-lightbox-caption").textContent = caption;
    const dl = document.getElementById("pasd-lightbox-download");
    dl.href = downloadHref;
    dl.setAttribute("download", fname);
    lb.style.display = "";
}

function closePasdLightbox() {
    document.getElementById("pasd-lightbox").style.display = "none";
}

// Click-to-enlarge for every module's source / mask preview image.
// Delegated so it covers all current and future modules automatically.
document.addEventListener("click", (e) => {
    const img = e.target.closest(".pasd-mask-frame img");
    if (!img) return;
    const src = img.getAttribute("src");
    if (!src) return;                       // empty / not-yet-loaded preview
    const wrap = img.closest(".pasd-mask-preview");
    const cap = wrap && wrap.querySelector(".pasd-mask-caption");
    openPasdLightbox(src, src, cap ? cap.textContent.trim() : "Preview", "preview.png");
});

function applyPasdModelControlsVisibility() {
    const isLdm = document.getElementById("pasd-gen-model").value === "ldm";
    document.querySelectorAll(".pasd-ldm-row").forEach(r => {
        r.style.display = isLdm ? "" : "none";
    });
}

async function loadPasdGenStatus() {
    const res = await fetch("/api/pasd_gen/status");
    pasdGenState.status = await res.json();
}

function refreshPasdGenModelStatus() {
    const el = document.getElementById("pasd-gen-model-status");
    const btn = document.getElementById("pasd-gen-btn");
    if (!pasdGenState.status) { el.textContent = ""; return; }
    const model = document.getElementById("pasd-gen-model").value;
    const info = pasdGenState.status[pasdGenState.modality]?.[model];
    const labels = {pix2pix: "Pix2pix", spade: "SPADE-GAN", ldm: "LDM"};
    const trainCmds = {
        pix2pix: `python train_pasd_pix2pix.py --modality ${pasdGenState.modality}`,
        spade:   `python train_pasd_spade.py --modality ${pasdGenState.modality}`,
        ldm:     `python train_pasd_ldm.py --vae_ckpt pasd_models/vae_${pasdGenState.modality.toLowerCase()}/latest.pt --modality ${pasdGenState.modality}`,
    };
    const label = labels[model] || model;
    const cmd = trainCmds[model] || "";
    if (info?.available) {
        el.textContent = `${label} checkpoint loaded (${info.size_mb} MB · ${info.path})`;
        el.className = "pasd-status ok";
        btn.disabled = false;
    } else {
        el.textContent = `No trained ${label} checkpoint for ${pasdGenState.modality} yet. Run: ${cmd}`;
        el.className = "pasd-status warn";
        btn.disabled = true;
    }
}

function switchPasdSource(source) {
    pasdGenState.source = source;
    document.querySelectorAll(".pasd-source-tab").forEach(t => {
        t.classList.toggle("active", t.dataset.pasdSource === source);
    });
    document.querySelectorAll(".pasd-source-panel").forEach(p => {
        p.style.display = p.dataset.pasdPanel === source ? "" : "none";
    });
}

async function loadPasdGenSubjects(modality) {
    if (pasdGenState.subjectsByMod[modality]) {
        populatePasdSubjectSelect();
        return;
    }
    const res = await fetch(`/api/mri/subjects?modality=${modality}`);
    const data = await res.json();
    pasdGenState.subjectsByMod[modality] = data.subjects;
    populatePasdSubjectSelect();
}

function populatePasdSubjectSelect() {
    const sel = document.getElementById("pasd-gen-subject");
    sel.innerHTML = "";
    const subjects = pasdGenState.subjectsByMod[pasdGenState.modality] || [];
    for (const s of subjects) {
        if (!s.has_mask) continue;
        const opt = document.createElement("option");
        opt.value = s.subject;
        opt.textContent = `${s.subject} (${s.slice_count} slices)`;
        sel.appendChild(opt);
    }
    if (subjects.length > 0) {
        pasdGenState.subject = sel.value;
        populatePasdSliceSelect();
    }
}

function populatePasdSliceSelect() {
    const sel = document.getElementById("pasd-gen-slice");
    sel.innerHTML = "";
    const subjects = pasdGenState.subjectsByMod[pasdGenState.modality] || [];
    const s = subjects.find(x => x.subject === pasdGenState.subject);
    if (!s) return;
    for (let i = s.first_slice; i <= s.last_slice; i++) {
        const opt = document.createElement("option");
        opt.value = i;
        opt.textContent = i;
        sel.appendChild(opt);
    }
    // Default to the middle slice
    const mid = Math.floor((s.first_slice + s.last_slice) / 2);
    sel.value = mid;
    pasdGenState.sliceIdx = mid;
    renderPasdMaskPreview();
}

function renderPasdMaskPreview() {
    if (!pasdGenState.subject || pasdGenState.sliceIdx == null) return;
    const bg = document.getElementById("pasd-gen-mask-bg");
    const ov = document.getElementById("pasd-gen-mask-overlay");
    bg.src = `/api/mri/${pasdGenState.modality}/${pasdGenState.subject}` +
             `/slice/axial/${pasdGenState.sliceIdx - getPasdFirstSlice()}?mask_alpha=0`;
    ov.src = `/api/pasd_gen/mask_preview?modality=${pasdGenState.modality}` +
             `&subject=${pasdGenState.subject}&slice_idx=${pasdGenState.sliceIdx}`;
}

function getPasdFirstSlice() {
    const subjects = pasdGenState.subjectsByMod[pasdGenState.modality] || [];
    const s = subjects.find(x => x.subject === pasdGenState.subject);
    return s ? s.first_slice : 0;
}

function onPasdUploadChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
        pasdGenState.uploadedMaskB64 = ev.target.result;
        document.getElementById("pasd-gen-uploaded").src = ev.target.result;
    };
    reader.readAsDataURL(file);
}

async function runPasdGen() {
    const btn = document.getElementById("pasd-gen-btn");
    btn.disabled = true;
    btn.textContent = "Generating...";
    try {
        const model = document.getElementById("pasd-gen-model").value;
        const body = {
            modality: pasdGenState.modality,
            model: model,
            num_samples: parseInt(document.getElementById("pasd-gen-num").value, 10),
            mask_source: pasdGenState.source,
            perturb: document.getElementById("pasd-gen-perturb").checked,
        };
        const seed = document.getElementById("pasd-gen-seed").value;
        if (seed !== "") body.seed = parseInt(seed, 10);
        if (model === "ldm") {
            body.num_inference_steps =
                parseInt(document.getElementById("pasd-gen-steps").value, 10);
            body.guidance_scale =
                parseFloat(document.getElementById("pasd-gen-guidance").value);
        }

        if (pasdGenState.source === "dataset") {
            body.subject = pasdGenState.subject;
            body.slice_idx = pasdGenState.sliceIdx;
        } else {
            if (!pasdGenState.uploadedMaskB64) {
                alert("Upload a mask PNG first.");
                return;
            }
            body.mask_b64 = pasdGenState.uploadedMaskB64;
        }

        const res = await fetch("/api/pasd_gen/sample", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok) {
            alert("Generation failed: " + (data.error || res.statusText));
            return;
        }
        renderPasdGenGallery(data);
    } finally {
        btn.disabled = false;
        btn.textContent = "Generate";
    }
}

async function runPasdCompareAll() {
    const cmpBtn = document.getElementById("pasd-compare-btn");
    const genBtn = document.getElementById("pasd-gen-btn");
    cmpBtn.disabled = true; genBtn.disabled = true;
    cmpBtn.textContent = "Running…";
    try {
        // Build the shared mask payload once
        const sharedBody = {
            modality: pasdGenState.modality,
            num_samples: 1,
            mask_source: pasdGenState.source,
            perturb: false,        // compare = one mask, no perturb
        };
        if (pasdGenState.source === "dataset") {
            sharedBody.subject = pasdGenState.subject;
            sharedBody.slice_idx = pasdGenState.sliceIdx;
        } else {
            if (!pasdGenState.uploadedMaskB64) {
                alert("Upload a mask PNG first."); return;
            }
            sharedBody.mask_b64 = pasdGenState.uploadedMaskB64;
        }
        const seed = document.getElementById("pasd-gen-seed").value;
        if (seed !== "") sharedBody.seed = parseInt(seed, 10);

        const status = pasdGenState.status?.[pasdGenState.modality] || {};
        const wanted = ["pix2pix", "spade", "ldm"].filter(m => status[m]?.available);
        if (wanted.length === 0) {
            alert("No models available for this modality."); return;
        }
        // Fire all in parallel
        const calls = wanted.map(model => {
            const body = {...sharedBody, model};
            if (model === "ldm") {
                body.num_inference_steps = 25;
                body.guidance_scale = 1.5;
            }
            return fetch("/api/pasd_gen/sample", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(body),
            }).then(r => r.json().then(d => ({model, ok: r.ok, data: d})));
        });
        const results = await Promise.all(calls);
        renderPasdCompareAllGallery(results, sharedBody);
    } finally {
        cmpBtn.disabled = false; genBtn.disabled = false;
        cmpBtn.textContent = "Compare all generators";
    }
}

function renderPasdCompareAllGallery(results, sharedBody) {
    const section = document.getElementById("pasd-gen-results-section");
    const gallery = document.getElementById("pasd-gen-gallery");
    section.style.display = "";
    gallery.innerHTML = "";

    const card = document.createElement("div");
    card.className = "pasd-gen-card pasd-compare-card";

    // Pick mask + real URLs from the FIRST successful result (they share inputs)
    const first = results.find(r => r.ok)?.data;
    if (!first) {
        gallery.innerHTML = "<p>All generators failed.</p>";
        return;
    }
    const maskSrc = sharedBody.mask_source === "upload"
        ? pasdGenState.uploadedMaskB64 : first.mask_url;
    const realSrc = first.real_url || null;
    pasdGenState.lastSessionId = first.session_id;
    pasdGenState.lastModel = "compare";

    const cells = [{label: "GT mask", src: maskSrc, cls: "pasd-panel-mask"}];
    for (const r of results) {
        if (!r.ok) {
            cells.push({label: r.model + " (failed)", src: "", cls: "pasd-panel-synth",
                        meta: {error: r.data?.error || "request failed"}});
            continue;
        }
        const sid = r.data.session_id;
        const synth = `/api/pasd_gen/sample_image/${sid}/0`;
        cells.push({
            label: PASD_MODEL_META[r.model]?.full_name || r.model,
            src: synth,
            cls: "pasd-panel-synth",
            meta: r.data.samples?.[0]?.metrics || null,
            sid,
        });
    }
    if (realSrc) cells.push({label: "Real MRI", src: realSrc, cls: "pasd-panel-real"});

    for (const c of cells) {
        const cell = document.createElement("div");
        cell.className = `pasd-gen-cell ${c.cls}`;
        const fname = `${c.label.replace(/[^\w]+/g, "_")}.png`;
        const downloadHref = c.sid
            ? `/api/pasd_gen/sample_image/${c.sid}/0?format=png&download=1`
            : c.src;
        cell.innerHTML = `
            <div class="pasd-cell-label">${c.label}</div>
            <div class="pasd-cell-frame">
                ${c.src ? `<img src="${c.src}" alt="${c.label}">` : `<div class="pasd-cell-err">${c.meta?.error || "no image"}</div>`}
                ${c.src ? `<button class="pasd-cell-zoom" title="View larger">&#128269;</button>
                          <a class="pasd-cell-dl" title="Download" href="${downloadHref}" download="${fname}">&#11015;</a>` : ""}
            </div>
        `;
        if (c.src) {
            const open = () => openPasdLightbox(c.src, downloadHref, c.label, fname);
            cell.querySelector(".pasd-cell-frame img").addEventListener("click", open);
            cell.querySelector(".pasd-cell-zoom").addEventListener("click", e => {
                e.stopPropagation(); open();
            });
        }
        // Metric chips per cell
        if (c.meta && !c.meta.error) {
            const bits = [];
            if (c.meta.mask_dice !== undefined) bits.push(`Dice <b>${c.meta.mask_dice.toFixed(3)}</b>`);
            if (c.meta.psnr      !== undefined) bits.push(`PSNR <b>${c.meta.psnr.toFixed(1)} dB</b>`);
            if (c.meta.ssim      !== undefined) bits.push(`SSIM <b>${c.meta.ssim.toFixed(3)}</b>`);
            if (bits.length) {
                const m = document.createElement("div");
                m.className = "pasd-cell-metrics";
                m.innerHTML = bits.join(" · ");
                cell.appendChild(m);
            }
        }
        card.appendChild(cell);
    }
    gallery.appendChild(card);

    const hint = document.createElement("p");
    hint.className = "pasd-gen-results-hint";
    hint.textContent = "Same input, every trained generator. Hover any image to zoom or download.";
    gallery.appendChild(hint);

    section.scrollIntoView({behavior: "smooth", block: "nearest"});
}

function renderPasdGenGallery(data) {
    const section = document.getElementById("pasd-gen-results-section");
    const gallery = document.getElementById("pasd-gen-gallery");
    section.style.display = "";
    gallery.innerHTML = "";
    // Remember the latest session for the "Download all" button
    pasdGenState.lastSessionId = data.session_id;
    pasdGenState.lastModel = data.model;

    // Three rendering modes:
    //   perturb=on              → per-sample (perturbed_mask | synthetic)
    //   perturb=off, dataset    → (input_mask | synthetic | real)
    //   perturb=off, upload     → (input_mask | synthetic)
    const perSampleMaskUrls = data.sample_mask_urls || null;
    const sharedMaskSrc = data.mask_source === "upload"
        ? pasdGenState.uploadedMaskB64
        : data.mask_url;
    const realSrc = data.real_url || null;

    for (let i = 0; i < data.num_samples; i++) {
        const card = document.createElement("div");
        card.className = "pasd-gen-card";

        const synthetic = `/api/pasd_gen/sample_image/${data.session_id}/${i}`;
        const maskForThis = perSampleMaskUrls
            ? perSampleMaskUrls[i]
            : sharedMaskSrc;
        const maskLabel = perSampleMaskUrls
            ? `GT mask #${i + 1}`
            : "GT mask";

        const panels = [
            {label: maskLabel, src: maskForThis, cls: "pasd-panel-mask"},
            {label: `Synthetic #${i + 1}`, src: synthetic, cls: "pasd-panel-synth"},
        ];
        if (realSrc && !perSampleMaskUrls) {
            panels.push({label: "Real MRI", src: realSrc, cls: "pasd-panel-real"});
        }

        for (const p of panels) {
            const cell = document.createElement("div");
            cell.className = `pasd-gen-cell ${p.cls}`;
            // Build a sensible download filename + the URL to download
            const fname = `${p.cls.replace("pasd-panel-", "")}_${(i+1).toString().padStart(2,"0")}.png`;
            // For synthetic, prefer the PNG endpoint so the download is lossless
            const downloadHref = p.cls === "pasd-panel-synth"
                ? `/api/pasd_gen/sample_image/${data.session_id}/${i}?format=png&download=1`
                : p.src;
            cell.innerHTML = `
                <div class="pasd-cell-label">${p.label}</div>
                <div class="pasd-cell-frame">
                    <img src="${p.src}" alt="${p.label}">
                    <button class="pasd-cell-zoom" title="View larger">&#128269;</button>
                    <a class="pasd-cell-dl" title="Download PNG" href="${downloadHref}" download="${fname}">&#11015;</a>
                </div>
            `;
            const imgEl = cell.querySelector(".pasd-cell-frame img");
            const zoomBtn = cell.querySelector(".pasd-cell-zoom");
            const open = () => openPasdLightbox(p.src, downloadHref, `${p.label} — sample ${i+1}`, fname);
            imgEl.addEventListener("click", open);
            zoomBtn.addEventListener("click", e => { e.stopPropagation(); open(); });
            card.appendChild(cell);
        }

        // Per-sample metrics row
        const m = data.samples[i] && data.samples[i].metrics;
        if (m && !m.error) {
            const metricRow = document.createElement("div");
            metricRow.className = "pasd-gen-metrics";
            const items = [];
            if (m.mask_dice !== undefined) {
                items.push(`<span class="pasd-metric" title="Dice between the trained segmenter's prediction on the synthetic image and the input mask. Higher = synthetic anatomy matches the mask better.">Mask Dice <b>${m.mask_dice.toFixed(3)}</b></span>`);
            }
            if (m.psnr !== undefined) {
                items.push(`<span class="pasd-metric" title="Peak SNR vs the paired real slice. Higher = pixel-closer to the actual MRI from the same mask.">PSNR <b>${m.psnr.toFixed(2)} dB</b></span>`);
            }
            if (m.ssim !== undefined) {
                items.push(`<span class="pasd-metric" title="Structural similarity vs the paired real slice (0–1).">SSIM <b>${m.ssim.toFixed(3)}</b></span>`);
            }
            metricRow.innerHTML = items.join("");
            card.appendChild(metricRow);
        } else if (m && m.error) {
            const errRow = document.createElement("div");
            errRow.className = "pasd-gen-metrics pasd-gen-metrics-err";
            errRow.textContent = "Metrics: " + m.error;
            card.appendChild(errRow);
        }

        gallery.appendChild(card);
    }

    // Footer hint
    const hint = document.createElement("p");
    hint.className = "pasd-gen-results-hint";
    if (perSampleMaskUrls) {
        hint.textContent = "Each row is an independent (mask, image) pair — the mask was rotated/flipped/scaled from the base, then fed to the generator to produce the matching synthetic slice.";
    } else if (data.real_url) {
        hint.textContent = "GT mask = generator input  ·  Synthetic = model output  ·  Real MRI = the actual slice the GT came from.";
    } else {
        hint.textContent = "GT mask = generator input  ·  Synthetic = model output. (Upload source has no paired real MRI.)";
    }
    gallery.appendChild(hint);

    section.scrollIntoView({behavior: "smooth", block: "nearest"});
}


// ══════════════════════════════════════════
// Cross-Modality Translation (CycleGAN)
// ══════════════════════════════════════════

let pasdXmodInitialized = false;
let pasdXmodState = {
    source: "dataset",
    direction: "BTFE_to_TSE",
    subject: null,
    sliceIdx: null,
    uploadedB64: null,
    subjectsByMod: {},
};

async function initPasdXmod() {
    if (!pasdXmodInitialized) {
        pasdXmodInitialized = true;
        wirePasdXmodEvents();
        await refreshPasdXmodStatus();
        await loadPasdXmodSubjects(currentSourceModality());
    }
    renderPasdModelInfo("pasd-xmod-info", "cyclegan");
}

function currentSourceModality() {
    return pasdXmodState.direction === "BTFE_to_TSE" ? "BTFE" : "TSE";
}

function wirePasdXmodEvents() {
    document.querySelectorAll("[data-pasd-xmod-source]").forEach(t => {
        t.addEventListener("click", () => {
            pasdXmodState.source = t.dataset.pasdXmodSource;
            document.querySelectorAll("[data-pasd-xmod-source]").forEach(x =>
                x.classList.toggle("active", x === t));
            document.querySelectorAll("[data-pasd-xmod-panel]").forEach(p =>
                p.style.display = p.dataset.pasdXmodPanel === pasdXmodState.source ? "" : "none");
        });
    });
    document.getElementById("pasd-xmod-direction").addEventListener("change", e => {
        pasdXmodState.direction = e.target.value;
        loadPasdXmodSubjects(currentSourceModality());
    });
    document.getElementById("pasd-xmod-subject").addEventListener("change", e => {
        pasdXmodState.subject = e.target.value;
        populatePasdXmodSliceSelect();
    });
    document.getElementById("pasd-xmod-slice").addEventListener("change", e => {
        pasdXmodState.sliceIdx = parseInt(e.target.value, 10);
        renderPasdXmodInputPreview();
    });
    document.getElementById("pasd-xmod-upload").addEventListener("change", e => {
        const file = e.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = ev => {
            pasdXmodState.uploadedB64 = ev.target.result;
            document.getElementById("pasd-xmod-uploaded-preview").src = ev.target.result;
        };
        reader.readAsDataURL(file);
    });
    document.getElementById("pasd-xmod-btn").addEventListener("click", runPasdXmod);
}

async function refreshPasdXmodStatus() {
    const res = await fetch("/api/pasd_gen/status");
    const status = await res.json();
    const cg = status.cyclegan;
    const el = document.getElementById("pasd-xmod-status");
    const btn = document.getElementById("pasd-xmod-btn");
    if (cg?.available) {
        el.textContent = `CycleGAN checkpoint loaded (${cg.size_mb} MB · ${cg.path})`;
        el.className = "pasd-status ok";
        btn.disabled = false;
    } else {
        el.textContent = "No trained CycleGAN yet. Run: python train_pasd_cyclegan.py";
        el.className = "pasd-status warn";
        btn.disabled = true;
    }
}

async function loadPasdXmodSubjects(modality) {
    if (!pasdXmodState.subjectsByMod[modality]) {
        const r = await fetch(`/api/mri/subjects?modality=${modality}`);
        const d = await r.json();
        pasdXmodState.subjectsByMod[modality] = d.subjects;
    }
    const sel = document.getElementById("pasd-xmod-subject");
    sel.innerHTML = "";
    for (const s of pasdXmodState.subjectsByMod[modality]) {
        const opt = document.createElement("option");
        opt.value = s.subject;
        opt.textContent = `${s.subject} (${s.slice_count} slices)`;
        sel.appendChild(opt);
    }
    if (sel.options.length) {
        pasdXmodState.subject = sel.value;
        populatePasdXmodSliceSelect();
    }
}

function populatePasdXmodSliceSelect() {
    const sel = document.getElementById("pasd-xmod-slice");
    sel.innerHTML = "";
    const modality = currentSourceModality();
    const s = (pasdXmodState.subjectsByMod[modality] || [])
              .find(x => x.subject === pasdXmodState.subject);
    if (!s) return;
    for (let i = s.first_slice; i <= s.last_slice; i++) {
        const opt = document.createElement("option");
        opt.value = i; opt.textContent = i;
        sel.appendChild(opt);
    }
    const mid = Math.floor((s.first_slice + s.last_slice) / 2);
    sel.value = mid;
    pasdXmodState.sliceIdx = mid;
    renderPasdXmodInputPreview();
}

function renderPasdXmodInputPreview() {
    if (!pasdXmodState.subject || pasdXmodState.sliceIdx == null) return;
    const modality = currentSourceModality();
    const s = (pasdXmodState.subjectsByMod[modality] || [])
              .find(x => x.subject === pasdXmodState.subject);
    if (!s) return;
    const pos = pasdXmodState.sliceIdx - s.first_slice;
    const img = document.getElementById("pasd-xmod-input-preview");
    img.src = `/api/mri/${modality}/${pasdXmodState.subject}/slice/axial/${pos}?mask_alpha=0`;
}

async function runPasdXmod() {
    const btn = document.getElementById("pasd-xmod-btn");
    btn.disabled = true; btn.textContent = "Translating...";
    try {
        const body = {
            direction: pasdXmodState.direction,
            source: pasdXmodState.source,
        };
        if (pasdXmodState.source === "dataset") {
            body.subject = pasdXmodState.subject;
            body.slice_idx = pasdXmodState.sliceIdx;
        } else {
            if (!pasdXmodState.uploadedB64) {
                alert("Upload an image first.");
                return;
            }
            body.image_b64 = pasdXmodState.uploadedB64;
        }
        const r = await fetch("/api/pasd_xmod/translate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(body),
        });
        const data = await r.json();
        if (!r.ok) {
            alert("Translation failed: " + (data.error || r.statusText));
            return;
        }
        renderPasdXmodResult(data);
    } finally {
        btn.disabled = false; btn.textContent = "Translate";
    }
}

function renderPasdXmodResult(data) {
    const section = document.getElementById("pasd-xmod-results-section");
    const pair = document.getElementById("pasd-xmod-pair");
    section.style.display = "";
    pair.innerHTML = "";
    const cells = [
        {label: `Input (${data.source_modality})`,  src: data.input_url,  cls: "pasd-panel-mask"},
        {label: `Synthetic (${data.target_modality})`, src: data.output_url, cls: "pasd-panel-synth"},
    ];
    for (const c of cells) {
        const div = document.createElement("div");
        div.className = `pasd-gen-cell ${c.cls}`;
        const fname = `${c.label.split(" ")[0].toLowerCase()}_${data.session_id}.png`;
        div.innerHTML = `
            <div class="pasd-cell-label">${c.label}</div>
            <div class="pasd-cell-frame">
                <img src="${c.src}" alt="${c.label}">
                <button class="pasd-cell-zoom" title="View larger">&#128269;</button>
                <a class="pasd-cell-dl" title="Download" href="${c.src}" download="${fname}">&#11015;</a>
            </div>
        `;
        const open = () => openPasdLightbox(c.src, c.src, c.label, fname);
        div.querySelector(".pasd-cell-frame img").addEventListener("click", open);
        div.querySelector(".pasd-cell-zoom").addEventListener("click", e => {
            e.stopPropagation(); open();
        });
        pair.appendChild(div);
    }
    section.scrollIntoView({behavior: "smooth", block: "nearest"});
}


// ══════════════════════════════════════════
// Cerebrovascular: MRA → CTA (TopBrain) module
// ══════════════════════════════════════════

let cvInitialized = false;
let cvState = {
    source: "dataset",
    patient: null,
    z: null,
    patients: [],
    uploadedB64: null,
};

// Static metadata for the CV pix2pix model
PASD_MODEL_META["cv_pix2pix_mr2ct"] = {
    full_name: "Pix2pix MRA → CTA",
    subtitle: "Paired modality translation, brain angiography",
    paper: "Isola et al. 2017 + TopCoW data, Koch et al. 2024",
    params: "54 M generator + 2.8 M discriminator",
    training: "60 ep on 4 690 paired axial slices (20/2/3 patient split)",
    stochastic: false,
    metrics: [
        {label: "Slices", value: "4 690"},
        {label: "Spacing", value: "0.5 mm³"},
        {label: "Modality", value: "TOF-MRA → CTA"},
    ],
    verdict: "Direct PDF-cited approach for compensating CTA scarcity.",
};

PASD_MODEL_META["cv_ldm_mr2ct"] = {
    full_name: "LDM MRA → CTA",
    subtitle: "Latent diffusion, MR channel-concat conditioning",
    paper: "Rombach et al. 2022 + Koch et al. 2024 (TopCoW diffusion)",
    params: "101 M UNet + 84 M VAE",
    training: "Goal 30 k steps · v-prediction · DPMSolver++ · CFG p=0.1",
    stochastic: true,
    metrics: [
        {label: "Slices", value: "4 690"},
        {label: "Latent", value: "(4, 32, 32) SD-VAE"},
        {label: "Conditioning", value: "MR latent channel-concat (8-ch input)"},
        {label: "FID ↓", value: "95.70", note: "best"},
    ],
    verdict: "PDF-recommended SOTA: paired diffusion translation. Stochastic — get multiple diverse CTAs from one MR.",
};

PASD_MODEL_META["cv_ldm_combined"] = {
    full_name: "LDM combined (MR + mask → CTA)",
    subtitle: "Latent diffusion, dual conditioning",
    paper: "Rombach et al. 2022 (extended for multi-class mask channel-concat)",
    params: "101.2 M UNet + 84 M VAE (in_channels=49)",
    training: "30 k steps · v-prediction · DPMSolver++ · CFG p=0.1 joint",
    stochastic: true,
    metrics: [
        {label: "Slices", value: "4 690"},
        {label: "Input", value: "4 noisy + 4 MR + 41 mask one-hot = 49 channels"},
        {label: "MSE (final)", value: "0.134"},
    ],
    verdict: "Adds vessel-mask topology constraint on top of MR conditioning. Tests whether masks add measurable value over MR alone.",
};

PASD_MODEL_META["cv_cyclegan_mr_ct"] = {
    full_name: "CycleGAN MR ↔ CT",
    subtitle: "Unpaired cross-modality translation",
    paper: "Zhu et al. 2017",
    params: "11.4 M G × 2 + 2.8 M D × 2",
    training: "60 ep · LSGAN + cycle (×10) + identity (×5) · image buffer · balance-guarded checkpoint",
    stochastic: false,
    metrics: [
        {label: "Train", value: "4 690 MR / 4 690 CT (sampled independently)"},
        {label: "D balance at best ckpt", value: "D_A=0.18, D_B=0.16"},
        {label: "Cycle L1", value: "≈ 0.7 (low — strong anatomy preservation)"},
    ],
    verdict: "Anatomy-preserving via cycle consistency. Bidirectional. Tests whether the PASD xmod finding generalizes to TopBrain.",
};

async function initCvMr2Ct() {
    if (!cvInitialized) {
        cvInitialized = true;
        wireCvEvents();
        await refreshCvStatus();
        await loadCvPatients();
    }
    renderPasdModelInfo("cv-model-info", "cv_pix2pix_mr2ct");
}

function wireCvEvents() {
    document.querySelectorAll("[data-cv-source]").forEach(t => {
        t.addEventListener("click", () => {
            cvState.source = t.dataset.cvSource;
            document.querySelectorAll("[data-cv-source]").forEach(x =>
                x.classList.toggle("active", x === t));
            document.querySelectorAll("[data-cv-panel]").forEach(p =>
                p.style.display = p.dataset.cvPanel === cvState.source ? "" : "none");
        });
    });
    document.getElementById("cv-patient").addEventListener("change", e => {
        cvState.patient = e.target.value;
        populateCvSliceSelect();
    });
    document.getElementById("cv-slice").addEventListener("change", e => {
        cvState.z = parseInt(e.target.value, 10);
        renderCvInputPreview();
    });
    document.getElementById("cv-upload").addEventListener("change", e => {
        const file = e.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = ev => {
            cvState.uploadedB64 = ev.target.result;
            document.getElementById("cv-uploaded-preview").src = ev.target.result;
        };
        reader.readAsDataURL(file);
    });
    document.getElementById("cv-generate-btn").addEventListener("click", runCvTranslate);
    document.getElementById("cv-compare-btn").addEventListener("click", runCvCompareBoth);
    document.getElementById("cv-model").addEventListener("change", () => {
        const m = document.getElementById("cv-model").value;
        const metaKey = m === "ldm"          ? "cv_ldm_mr2ct"
                      : m === "ldm_combined" ? "cv_ldm_combined"
                      : m === "cyclegan"     ? "cv_cyclegan_mr_ct"
                      : "cv_pix2pix_mr2ct";
        renderPasdModelInfo("cv-model-info", metaKey);
        // Both LDM variants use the same inference-steps/CFG controls
        const isLdm = (m === "ldm" || m === "ldm_combined");
        document.querySelectorAll(".cv-ldm-row").forEach(r =>
            r.style.display = isLdm ? "" : "none");
        refreshCvStatus();
    });
}

async function refreshCvStatus() {
    const res = await fetch("/api/pasd_gen/status");
    const status = await res.json();
    const selected = document.getElementById("cv-model")?.value || "pix2pix";
    const key = selected === "ldm"          ? "ldm_mr2ct"
              : selected === "ldm_combined" ? "ldm_combined"
              : selected === "cyclegan"     ? "cyclegan_mr_ct"
              : "pix2pix_mr2ct";
    const info = status.cv?.[key];
    const el = document.getElementById("cv-status");
    const btn = document.getElementById("cv-generate-btn");
    const labels = {
        pix2pix_mr2ct: "Pix2pix MRA→CTA",
        ldm_mr2ct: "LDM MRA→CTA",
        ldm_combined: "LDM combined (MR + mask)",
        cyclegan_mr_ct: "CycleGAN MR↔CT",
    };
    const cmds = {
        pix2pix_mr2ct: "python train_cv_pix2pix.py",
        ldm_mr2ct: "python train_cv_vae.py && python train_cv_ldm.py --vae_ckpt cv_models/vae/latest.pt",
        ldm_combined: "python train_cv_ldm_combined.py --vae_ckpt cv_models/vae/latest.pt",
        cyclegan_mr_ct: "python train_cv_cyclegan.py",
    };
    if (info?.available) {
        el.textContent = `${labels[key]} loaded (${info.size_mb} MB · ${info.path})`;
        el.className = "pasd-status ok";
        btn.disabled = false;
    } else {
        el.textContent = `No trained ${labels[key]} checkpoint yet. Run: ${cmds[key]}`;
        el.className = "pasd-status warn";
        btn.disabled = true;
    }
}

async function loadCvPatients() {
    const r = await fetch("/api/cv/patients");
    const d = await r.json();
    cvState.patients = d.patients;
    const sel = document.getElementById("cv-patient");
    sel.innerHTML = "";
    for (const p of d.patients) {
        const opt = document.createElement("option");
        opt.value = p.patient;
        opt.textContent = `sub${p.patient} (${p.n_slices} slices)`;
        sel.appendChild(opt);
    }
    if (sel.options.length) {
        cvState.patient = sel.value;
        populateCvSliceSelect();
    }
}

function populateCvSliceSelect() {
    const sel = document.getElementById("cv-slice");
    sel.innerHTML = "";
    const p = cvState.patients.find(x => x.patient === cvState.patient);
    if (!p) return;
    // 256³ grid → z in [0, 255]
    for (let z = 0; z < 256; z += 2) {       // step 2 to keep dropdown manageable
        const opt = document.createElement("option");
        opt.value = z; opt.textContent = z;
        sel.appendChild(opt);
    }
    const mid = 128;
    sel.value = mid;
    cvState.z = mid;
    renderCvInputPreview();
}

function renderCvInputPreview() {
    if (!cvState.patient || cvState.z == null) return;
    document.getElementById("cv-mr-preview").src =
        `/api/cv/${cvState.patient}/slice/mr/${cvState.z}`;
}

async function runCvTranslate() {
    const btn = document.getElementById("cv-generate-btn");
    btn.disabled = true; btn.textContent = "Generating…";
    try {
        const body = {model: document.getElementById("cv-model").value};
        if (body.model === "ldm") {
            body.num_inference_steps =
                parseInt(document.getElementById("cv-ldm-steps").value, 10);
            body.guidance_scale =
                parseFloat(document.getElementById("cv-ldm-guidance").value);
        }
        if (cvState.source === "dataset") {
            body.patient = cvState.patient;
            body.z = cvState.z;
        } else {
            if (!cvState.uploadedB64) {
                alert("Upload an image first."); return;
            }
            body.image_b64 = cvState.uploadedB64;
        }
        const r = await fetch("/api/cv/translate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(body),
        });
        const data = await r.json();
        if (!r.ok) {
            alert("Translation failed: " + (data.error || r.statusText));
            return;
        }
        renderCvResult(data);
    } finally {
        btn.disabled = false; btn.textContent = "Generate CTA";
    }
}

function renderCvResult(data) {
    const section = document.getElementById("cv-results-section");
    const pair = document.getElementById("cv-result-pair");
    section.style.display = "";
    pair.innerHTML = "";
    const cells = [
        {label: "Input (MRA)",  src: data.input_url,  cls: "pasd-panel-mask"},
        {label: "Synthetic CTA", src: data.output_url, cls: "pasd-panel-synth"},
    ];
    // Add real CTA reference if dataset source — we can fetch it from the slice API
    if (data.patient != null && data.z != null) {
        cells.push({
            label: "Real CTA (reference)",
            src: `/api/cv/${data.patient}/slice/ct/${data.z}`,
            cls: "pasd-panel-real",
        });
    }
    for (const c of cells) {
        const div = document.createElement("div");
        div.className = `pasd-gen-cell ${c.cls}`;
        const fname = `cv_${c.label.split(" ")[0].toLowerCase()}_${data.session_id}.png`;
        div.innerHTML = `
            <div class="pasd-cell-label">${c.label}</div>
            <div class="pasd-cell-frame">
                <img src="${c.src}" alt="${c.label}">
                <button class="pasd-cell-zoom" title="View larger">&#128269;</button>
                <a class="pasd-cell-dl" title="Download" href="${c.src}" download="${fname}">&#11015;</a>
            </div>
        `;
        const open = () => openPasdLightbox(c.src, c.src, c.label, fname);
        div.querySelector(".pasd-cell-frame img").addEventListener("click", open);
        div.querySelector(".pasd-cell-zoom").addEventListener("click", e => {
            e.stopPropagation(); open();
        });
        pair.appendChild(div);
    }
    section.scrollIntoView({behavior: "smooth", block: "nearest"});
}


// ══════════════════════════════════════════
// Prostate158: T2 → ADC (pix2pix) module
// ══════════════════════════════════════════

let prostateInitialized = false;
let prostateState = { patient: null, z: null, patients: [] };

PASD_MODEL_META["prostate_pix2pix_t2_adc"] = {
    full_name: "Pix2pix T2 → ADC (prostate158)",
    subtitle: "Paired same-grid generator",
    paper: "Isola et al. 2017 + Adams et al. 2022 (prostate158)",
    params: "54 M U-Net + PatchGAN",
    training: "60 ep · L1=100 · 3061 train slices · 256² @ 0.5 mm",
    stochastic: false,
    metrics: [
        {label: "Patients", value: "119 train + 20 val + 19 test"},
        {label: "Final L1",  value: "7.4 (start 18.8)"},
        {label: "D balance", value: "0.37 / 0.31 (real / fake)"},
        {label: "Test FID",  value: "81.6"},
        {label: "Test SSIM", value: "0.545"},
    ],
    verdict: "First-pass paired translation. FID better than TopBrain pix2pix (126.9). SSIM weaker. LDM to compare.",
};

PASD_MODEL_META["prostate_cyclegan_t2_adc"] = {
    full_name: "CycleGAN T2 ↔ ADC (prostate158)",
    subtitle: "Unpaired-style training, anatomy-preserving",
    paper: "Zhu et al. 2017 + Adams et al. 2022 (prostate158)",
    params: "11.4 M generator × 2 directions",
    training: "100 ep · LSGAN + cycle (λ=10) + identity (λ=5) · best_G.pt (balance-guarded)",
    stochastic: false,
    metrics: [
        {label: "Patients",       value: "119 train + 20 val + 19 test"},
        {label: "Final balance",  value: "0.149 (no D-collapse)"},
        {label: "D_A / D_B",      value: "0.20 / 0.085"},
    ],
    verdict: "Cycle-consistency learning on paired data sampled as unpaired. Late training avoided D-collapse. Tests whether the PASD cross-modality win replicates on prostate.",
};

PASD_MODEL_META["prostate_ldm_t2_adc"] = {
    full_name: "LDM T2 → ADC (prostate158)",
    subtitle: "Latent diffusion, T2-conditioned",
    paper: "Rombach et al. 2022 + Adams et al. 2022 (prostate158)",
    params: "120 M UNet + 84 M VAE",
    training: "VAE 8k steps + LDM 30k steps · v-prediction · DPMSolver++ · CFG p=0.1",
    stochastic: true,
    metrics: [
        {label: "Patients",   value: "119 train + 20 val + 19 test"},
        {label: "MSE (final)", value: "0.216"},
        {label: "Input",      value: "4 noisy ADC + 4 T2 latent = 8 channels"},
    ],
    verdict: "Stochastic alternative to pix2pix. Higher MSE than TopBrain LDM (0.135 vs 0.216) — ADC harder to predict than CT. Eval comparison pending.",
};


async function initProstateT2Adc() {
    if (!prostateInitialized) {
        prostateInitialized = true;
        wireProstateEvents();
        await refreshProstateStatus();
        await loadProstatePatients();
    }
    renderPasdModelInfo("prostate-model-info", "prostate_pix2pix_t2_adc");
}


function wireProstateEvents() {
    document.getElementById("prostate-patient").addEventListener("change", e => {
        prostateState.patient = e.target.value;
        populateProstateSliceSelect();
    });
    document.getElementById("prostate-slice").addEventListener("change", e => {
        prostateState.z = parseInt(e.target.value, 10);
        renderProstateInputPreview();
    });
    document.getElementById("prostate-generate-btn")
        .addEventListener("click", runProstateTranslate);
    document.getElementById("prostate-model").addEventListener("change", () => {
        const m = document.getElementById("prostate-model").value;
        const metaKey = m === "ldm"      ? "prostate_ldm_t2_adc"
                      : m === "pix2pix"  ? "prostate_pix2pix_t2_adc"
                      : m === "cyclegan" ? "prostate_cyclegan_t2_adc"
                      : "prostate_" + m;
        renderPasdModelInfo("prostate-model-info", metaKey);
        document.querySelectorAll(".prostate-ldm-row").forEach(r =>
            r.style.display = (m === "ldm") ? "" : "none");
        refreshProstateStatus();
    });
}


async function refreshProstateStatus() {
    const res = await fetch("/api/pasd_gen/status");
    const status = await res.json();
    const selected = document.getElementById("prostate-model")?.value || "pix2pix";
    const key = selected === "ldm"      ? "ldm_t2_adc"
              : selected === "pix2pix"  ? "pix2pix_t2_adc"
              : selected === "cyclegan" ? "cyclegan_t2_adc"
              : selected;
    const info = status.prostate?.[key];
    const el = document.getElementById("prostate-status");
    const btn = document.getElementById("prostate-generate-btn");
    const labels = {
        pix2pix_t2_adc:  "Pix2pix T2→ADC",
        ldm_t2_adc:      "LDM T2→ADC",
        cyclegan_t2_adc: "CycleGAN T2↔ADC",
    };
    const cmds = {
        pix2pix_t2_adc:  "python train_prostate_pix2pix.py",
        ldm_t2_adc:      "python train_prostate_vae.py && python train_prostate_ldm.py --vae_ckpt prostate_models/vae/latest.pt",
        cyclegan_t2_adc: "python train_prostate_cyclegan.py",
    };
    if (info?.available) {
        el.textContent = `${labels[key]} loaded (${info.size_mb} MB · ${info.path})`;
        el.className = "pasd-status ok";
        btn.disabled = false;
    } else {
        el.textContent = `No trained ${labels[key]} checkpoint yet. Run: ${cmds[key]}`;
        el.className = "pasd-status warn";
        btn.disabled = true;
    }
}


async function loadProstatePatients() {
    const r = await fetch("/api/prostate/patients");
    const d = await r.json();
    prostateState.patients = d.patients;
    const sel = document.getElementById("prostate-patient");
    sel.innerHTML = "";
    for (const p of d.patients) {
        const opt = document.createElement("option");
        opt.value = p.patient;
        opt.textContent = `sub-${p.patient} · ${p.split} · ${p.n_slices}/${p.z_total} tissue slices`;
        sel.appendChild(opt);
    }
    if (sel.options.length) {
        prostateState.patient = sel.value;
        populateProstateSliceSelect();
    }
}


function populateProstateSliceSelect() {
    const sel = document.getElementById("prostate-slice");
    sel.innerHTML = "";
    const p = prostateState.patients.find(x => x.patient === prostateState.patient);
    if (!p) return;
    for (let z = 0; z < p.z_total; z++) {
        const opt = document.createElement("option");
        opt.value = z; opt.textContent = z;
        sel.appendChild(opt);
    }
    // Pick a mid-prostate slice (typically z ≈ 40-50% of volume)
    const mid = Math.floor(p.z_total / 2);
    sel.value = mid;
    prostateState.z = mid;
    renderProstateInputPreview();
}


function renderProstateInputPreview() {
    if (!prostateState.patient || prostateState.z == null) return;
    document.getElementById("prostate-t2-preview").src =
        `/api/prostate/${prostateState.patient}/slice/t2/${prostateState.z}`;
}


async function runProstateTranslate() {
    const btn = document.getElementById("prostate-generate-btn");
    btn.disabled = true; btn.textContent = "Generating…";
    try {
        const body = {
            patient: prostateState.patient,
            z: prostateState.z,
            model: document.getElementById("prostate-model").value,
        };
        if (body.model === "ldm") {
            body.num_inference_steps =
                parseInt(document.getElementById("prostate-ldm-steps").value, 10);
            body.guidance_scale =
                parseFloat(document.getElementById("prostate-ldm-guidance").value);
            body.n_ensemble =
                parseInt(document.getElementById("prostate-ldm-ensemble").value, 10);
        }
        const r = await fetch("/api/prostate/translate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(body),
        });
        const data = await r.json();
        if (!r.ok) {
            alert("Translation failed: " + (data.error || r.statusText));
            return;
        }
        renderProstateResult(data);
    } finally {
        btn.disabled = false; btn.textContent = "Generate ADC";
    }
}


function renderProstateResult(data) {
    const section = document.getElementById("prostate-results-section");
    const pair = document.getElementById("prostate-result-pair");
    section.style.display = "";
    pair.innerHTML = "";
    const cells = [
        {label: "Input (T2)",        src: data.input_url,    cls: "pasd-panel-mask"},
        {label: "Synthetic ADC",     src: data.output_url,   cls: "pasd-panel-synth"},
        {label: "Real ADC (reference)", src: data.real_adc_url, cls: "pasd-panel-real"},
    ];
    for (const c of cells) {
        const div = document.createElement("div");
        div.className = `pasd-gen-cell ${c.cls}`;
        const fname = `prostate_${c.label.split(" ")[0].toLowerCase()}_${data.session_id}.png`;
        div.innerHTML = `
            <div class="pasd-cell-label">${c.label}</div>
            <div class="pasd-cell-frame">
                <img src="${c.src}" alt="${c.label}">
                <button class="pasd-cell-zoom" title="View larger">&#128269;</button>
                <a class="pasd-cell-dl" title="Download" href="${c.src}" download="${fname}">&#11015;</a>
            </div>
        `;
        const open = () => openPasdLightbox(c.src, c.src, c.label, fname);
        div.querySelector(".pasd-cell-frame img").addEventListener("click", open);
        div.querySelector(".pasd-cell-zoom").addEventListener("click", e => {
            e.stopPropagation(); open();
        });
        pair.appendChild(div);
    }
    section.scrollIntoView({behavior: "smooth", block: "nearest"});
}


// ══════════════════════════════════════════
// Prostate158: Anatomy Mask → T2 (SPADE) module
// ══════════════════════════════════════════

let prostateSpadeInitialized = false;
let prostateSpadeState = { patient: null, z: null, patients: [] };

PASD_MODEL_META["prostate_spade_mask_t2"] = {
    full_name: "SPADE T2 (anatomy mask → T2, prostate158)",
    subtitle: "Multi-class mask conditioning (3 classes)",
    paper: "Park et al. 2019 + Adams et al. 2022 (prostate158)",
    params: "24.4 M generator + 2.8 M discriminator",
    training: "ep 70 (pre-collapse) · L1=0 · hinge GAN + VGG + FM · EMA-applied",
    stochastic: true,
    metrics: [
        {label: "Patients",   value: "119 train + 20 val + 19 test"},
        {label: "Mask coverage", value: "~30% of slice (PZ + CG)"},
        {label: "Train health", value: "D collapsed at ep ~85; rolled back to ep 70"},
    ],
    verdict: "Tests the TopBrain-SPADE-sparsity hypothesis: prostate masks are denser than TopBrain's vessel masks, should give SPADE more signal. Visual: generates believable T2 anatomy around the mask region.",
};


async function initProstateMaskT2() {
    if (!prostateSpadeInitialized) {
        prostateSpadeInitialized = true;
        wireProstateSpadeEvents();
        await refreshProstateSpadeStatus();
        await loadProstateSpadePatients();
    }
    renderPasdModelInfo("prostate-spade-model-info", "prostate_spade_mask_t2");
}


function wireProstateSpadeEvents() {
    document.getElementById("prostate-spade-patient").addEventListener("change", e => {
        prostateSpadeState.patient = e.target.value;
        populateProstateSpadeSliceSelect();
    });
    document.getElementById("prostate-spade-slice").addEventListener("change", e => {
        prostateSpadeState.z = parseInt(e.target.value, 10);
        renderProstateSpadeMaskPreview();
    });
    document.getElementById("prostate-spade-generate-btn")
        .addEventListener("click", runProstateSpadeGenerate);
}


async function refreshProstateSpadeStatus() {
    const res = await fetch("/api/pasd_gen/status");
    const status = await res.json();
    const info = status.prostate?.spade_mask_t2;
    const el = document.getElementById("prostate-spade-status");
    const btn = document.getElementById("prostate-spade-generate-btn");
    if (info?.available) {
        el.textContent = `SPADE loaded (${info.size_mb} MB · ${info.path})`;
        el.className = "pasd-status ok";
        btn.disabled = false;
    } else {
        el.textContent = "No trained SPADE checkpoint. Run: python train_prostate_spade.py";
        el.className = "pasd-status warn";
        btn.disabled = true;
    }
}


async function loadProstateSpadePatients() {
    const r = await fetch("/api/prostate/patients");
    const d = await r.json();
    prostateSpadeState.patients = d.patients;
    const sel = document.getElementById("prostate-spade-patient");
    sel.innerHTML = "";
    for (const p of d.patients) {
        const opt = document.createElement("option");
        opt.value = p.patient;
        opt.textContent = `sub-${p.patient} · ${p.split} · ${p.n_slices}/${p.z_total} tissue slices`;
        sel.appendChild(opt);
    }
    if (sel.options.length) {
        prostateSpadeState.patient = sel.value;
        populateProstateSpadeSliceSelect();
    }
}


function populateProstateSpadeSliceSelect() {
    const sel = document.getElementById("prostate-spade-slice");
    sel.innerHTML = "";
    const p = prostateSpadeState.patients.find(x => x.patient === prostateSpadeState.patient);
    if (!p) return;
    for (let z = 0; z < p.z_total; z++) {
        const opt = document.createElement("option");
        opt.value = z; opt.textContent = z;
        sel.appendChild(opt);
    }
    const mid = Math.floor(p.z_total / 2);
    sel.value = mid;
    prostateSpadeState.z = mid;
    renderProstateSpadeMaskPreview();
}


function renderProstateSpadeMaskPreview() {
    if (!prostateSpadeState.patient || prostateSpadeState.z == null) return;
    document.getElementById("prostate-spade-mask-preview").src =
        `/api/prostate/${prostateSpadeState.patient}/slice/t2_anatomy/${prostateSpadeState.z}`;
}


async function runProstateSpadeGenerate() {
    const btn = document.getElementById("prostate-spade-generate-btn");
    btn.disabled = true; btn.textContent = "Generating…";
    try {
        const body = {
            patient: prostateSpadeState.patient,
            z: prostateSpadeState.z,
        };
        const seed = document.getElementById("prostate-spade-seed").value;
        if (seed !== "") body.seed = parseInt(seed, 10);
        const r = await fetch("/api/prostate/spade_generate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(body),
        });
        const data = await r.json();
        if (!r.ok) {
            alert("Generation failed: " + (data.error || r.statusText));
            return;
        }
        renderProstateSpadeResult(data);
    } finally {
        btn.disabled = false; btn.textContent = "Generate T2";
    }
}


function renderProstateSpadeResult(data) {
    const section = document.getElementById("prostate-spade-results-section");
    const pair = document.getElementById("prostate-spade-result-pair");
    section.style.display = "";
    pair.innerHTML = "";
    const cells = [
        {label: "Input (mask)",     src: data.input_url,     cls: "pasd-panel-mask"},
        {label: "Synthetic T2",     src: data.output_url,    cls: "pasd-panel-synth"},
        {label: "Real T2 (reference)", src: data.real_t2_url, cls: "pasd-panel-real"},
    ];
    for (const c of cells) {
        const div = document.createElement("div");
        div.className = `pasd-gen-cell ${c.cls}`;
        const fname = `prostate_spade_${c.label.split(" ")[0].toLowerCase()}_${data.session_id}.png`;
        div.innerHTML = `
            <div class="pasd-cell-label">${c.label}</div>
            <div class="pasd-cell-frame">
                <img src="${c.src}" alt="${c.label}">
                <button class="pasd-cell-zoom" title="View larger">&#128269;</button>
                <a class="pasd-cell-dl" title="Download" href="${c.src}" download="${fname}">&#11015;</a>
            </div>
        `;
        const open = () => openPasdLightbox(c.src, c.src, c.label, fname);
        div.querySelector(".pasd-cell-frame img").addEventListener("click", open);
        div.querySelector(".pasd-cell-zoom").addEventListener("click", e => {
            e.stopPropagation(); open();
        });
        pair.appendChild(div);
    }
    section.scrollIntoView({behavior: "smooth", block: "nearest"});
}


// ══════════════════════════════════════════
// Segmentation (TSTR): realism vs relevance
// ══════════════════════════════════════════

let tstrSegInit = false;

function initTstrSeg() {
    if (!tstrSegInit) {
        tstrSegInit = true;
        document.getElementById("tstr-seg-run").addEventListener("click", runTstrSeg);
        document.getElementById("tstr-seg-dataset").addEventListener("change", () => {
            // reset slice index sensibly per dataset
            const def = {prostate: 240, topbrain: 384, pasd: 170};
            document.getElementById("tstr-seg-idx").value =
                def[document.getElementById("tstr-seg-dataset").value] ?? 0;
        });
    }
}

async function runTstrSeg() {
    const ds = document.getElementById("tstr-seg-dataset").value;
    const idx = parseInt(document.getElementById("tstr-seg-idx").value, 10);
    const btn = document.getElementById("tstr-seg-run");
    const status = document.getElementById("tstr-seg-status");
    btn.disabled = true; btn.textContent = "Running…";
    status.textContent = "Running all segmenters on the real slice…";
    status.className = "pasd-status";
    try {
        const r = await fetch(`/api/tstr/${ds}/segment/${idx}`);
        const d = await r.json();
        if (!r.ok) { status.textContent = "Failed: " + (d.error || r.statusText);
                     status.className = "pasd-status warn"; return; }
        renderTstrSeg(d);
        status.textContent = `Done — ${d.panels.filter(p=>'dice'in p).length} segmenters on ${ds} slice ${idx}.`;
        status.className = "pasd-status ok";
    } finally { btn.disabled = false; btn.textContent = "Run all segmenters"; }
}

function renderTstrSeg(data) {
    const section = document.getElementById("tstr-seg-results-section");
    const pair = document.getElementById("tstr-seg-pair");
    section.style.display = "";
    pair.innerHTML = "";
    // sort: input, gt first, then conditions by Dice desc
    const head = data.panels.filter(p => !('dice' in p));
    const conds = data.panels.filter(p => 'dice' in p).sort((a,b) => b.dice - a.dice);
    const best = conds.length ? conds[0].key : null;
    for (const p of [...head, ...conds]) {
        const div = document.createElement("div");
        div.className = "pasd-gen-cell";
        const cap = ('dice' in p)
            ? `${p.label} · Dice ${p.dice.toFixed(3)}${p.key===best?' ★':''}`
            : p.label;
        div.innerHTML = `
            <div class="pasd-cell-label">${cap}</div>
            <div class="pasd-cell-frame">
                <img src="${p.url}" alt="${p.label}">
                <button class="pasd-cell-zoom" title="View larger">&#128269;</button>
            </div>`;
        const open = () => openPasdLightbox(p.url, p.url, cap, `tstr_${p.key}.png`);
        div.querySelector("img").addEventListener("click", open);
        div.querySelector(".pasd-cell-zoom").addEventListener("click", e => { e.stopPropagation(); open(); });
        pair.appendChild(div);
    }
    section.scrollIntoView({behavior: "smooth", block: "nearest"});
}


// ══════════════════════════════════════════
// Cerebrovascular SPADE: Vessel-Mask → CTA
// ══════════════════════════════════════════

let cvSpadeInitialized = false;
let cvSpadeState = { patient: null, z: null, patients: [] };

PASD_MODEL_META["cv_spade_mask2ct"] = {
    full_name: "SPADE-GAN multi-class (mask → CTA)",
    subtitle: "41-channel vessel-mask conditioning",
    paper: "Park et al. 2019 + TopBrain MICCAI 2025",
    params: "25 M generator + 2.8 M discriminator",
    training: "100 ep · L1=0 · hinge GAN + VGG + FM · CT mask labels 0-40",
    stochastic: true,
    metrics: [
        {label: "Classes", value: "41 vessel labels"},
        {label: "Output", value: "1-channel CTA, 256×256"},
        {label: "Train slices", value: "4 690"},
    ],
    verdict: "Generates anatomy-controlled CTA from a vessel skeleton. Stochastic z gives different appearance per draw.",
};

async function initCvMask2Ct() {
    if (!cvSpadeInitialized) {
        cvSpadeInitialized = true;
        wireCvSpadeEvents();
        await refreshCvSpadeStatus();
        await loadCvSpadePatients();
    }
    renderPasdModelInfo("cvspade-model-info", "cv_spade_mask2ct");
}

function wireCvSpadeEvents() {
    document.getElementById("cvspade-patient").addEventListener("change", e => {
        cvSpadeState.patient = e.target.value;
        populateCvSpadeSliceSelect();
    });
    document.getElementById("cvspade-slice").addEventListener("change", e => {
        cvSpadeState.z = parseInt(e.target.value, 10);
        renderCvSpadeMaskPreview();
    });
    document.getElementById("cvspade-generate-btn").addEventListener("click", runCvSpadeGenerate);
}

async function refreshCvSpadeStatus() {
    const res = await fetch("/api/pasd_gen/status");
    const status = await res.json();
    const info = status.cv?.spade_mask2ct;
    const el = document.getElementById("cvspade-status");
    const btn = document.getElementById("cvspade-generate-btn");
    if (info?.available) {
        el.textContent = `SPADE mask→CT loaded (${info.size_mb} MB · ${info.path})`;
        el.className = "pasd-status ok";
        btn.disabled = false;
    } else {
        el.textContent = "No trained checkpoint yet. Run: python train_cv_spade.py";
        el.className = "pasd-status warn";
        btn.disabled = true;
    }
}

async function loadCvSpadePatients() {
    if (cvSpadeState.patients.length) {
        populateCvSpadePatientSelect();
        return;
    }
    const r = await fetch("/api/cv/patients");
    const d = await r.json();
    cvSpadeState.patients = d.patients;
    populateCvSpadePatientSelect();
}

function populateCvSpadePatientSelect() {
    const sel = document.getElementById("cvspade-patient");
    sel.innerHTML = "";
    for (const p of cvSpadeState.patients) {
        const opt = document.createElement("option");
        opt.value = p.patient;
        opt.textContent = `sub${p.patient} (${p.n_slices} slices)`;
        sel.appendChild(opt);
    }
    if (sel.options.length) {
        cvSpadeState.patient = sel.value;
        populateCvSpadeSliceSelect();
    }
}

function populateCvSpadeSliceSelect() {
    const sel = document.getElementById("cvspade-slice");
    sel.innerHTML = "";
    for (let z = 0; z < 256; z += 2) {
        const opt = document.createElement("option");
        opt.value = z; opt.textContent = z;
        sel.appendChild(opt);
    }
    sel.value = 128;
    cvSpadeState.z = 128;
    renderCvSpadeMaskPreview();
}

function renderCvSpadeMaskPreview() {
    if (!cvSpadeState.patient || cvSpadeState.z == null) return;
    document.getElementById("cvspade-mask-preview").src =
        `/api/cv/${cvSpadeState.patient}/slice/ct_mask/${cvSpadeState.z}`;
}

async function runCvSpadeGenerate() {
    const btn = document.getElementById("cvspade-generate-btn");
    btn.disabled = true; btn.textContent = "Generating…";
    try {
        const body = {patient: cvSpadeState.patient, z: cvSpadeState.z};
        const seed = document.getElementById("cvspade-seed").value;
        if (seed !== "") body.seed = parseInt(seed, 10);
        const r = await fetch("/api/cv/spade_generate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(body),
        });
        const data = await r.json();
        if (!r.ok) {
            alert("Generation failed: " + (data.error || r.statusText));
            return;
        }
        renderCvSpadeResult(data);
    } finally {
        btn.disabled = false; btn.textContent = "Generate CTA from mask";
    }
}

function renderCvSpadeResult(data) {
    const section = document.getElementById("cvspade-results-section");
    const gallery = document.getElementById("cvspade-result-gallery");
    section.style.display = "";
    gallery.innerHTML = "";
    const card = document.createElement("div");
    card.className = "pasd-gen-card";
    const cells = [
        {label: "Vessel mask (input)",   src: data.input_url,   cls: "pasd-panel-mask"},
        {label: "Synthetic CTA",         src: data.output_url,  cls: "pasd-panel-synth"},
        {label: "Real CTA (reference)",  src: data.real_ct_url, cls: "pasd-panel-real"},
    ];
    for (const c of cells) {
        const cell = document.createElement("div");
        cell.className = `pasd-gen-cell ${c.cls}`;
        const fname = `cvspade_${c.label.split(" ")[0].toLowerCase()}_${data.session_id}.png`;
        cell.innerHTML = `
            <div class="pasd-cell-label">${c.label}</div>
            <div class="pasd-cell-frame">
                <img src="${c.src}" alt="${c.label}">
                <button class="pasd-cell-zoom" title="View larger">&#128269;</button>
                <a class="pasd-cell-dl" title="Download" href="${c.src}" download="${fname}">&#11015;</a>
            </div>
        `;
        const open = () => openPasdLightbox(c.src, c.src, c.label, fname);
        cell.querySelector(".pasd-cell-frame img").addEventListener("click", open);
        cell.querySelector(".pasd-cell-zoom").addEventListener("click", e => {
            e.stopPropagation(); open();
        });
        card.appendChild(cell);
    }
    gallery.appendChild(card);
    section.scrollIntoView({behavior: "smooth", block: "nearest"});
}


async function runCvCompareBoth() {
    const cmpBtn = document.getElementById("cv-compare-btn");
    const genBtn = document.getElementById("cv-generate-btn");
    cmpBtn.disabled = true; genBtn.disabled = true;
    cmpBtn.textContent = "Running…";
    try {
        const sharedBody = {};
        if (cvState.source === "dataset") {
            sharedBody.patient = cvState.patient;
            sharedBody.z = cvState.z;
        } else {
            if (!cvState.uploadedB64) {
                alert("Upload an image first."); return;
            }
            sharedBody.image_b64 = cvState.uploadedB64;
        }
        const seed = document.getElementById("cv-ldm-seed")?.value;

        const status = await (await fetch("/api/pasd_gen/status")).json();
        const wanted = ["pix2pix", "ldm"].filter(m =>
            status.cv?.[m === "pix2pix" ? "pix2pix_mr2ct" : "ldm_mr2ct"]?.available);
        if (wanted.length < 2) {
            alert("Need both pix2pix and LDM trained to compare. Available: " + wanted.join(", "));
            return;
        }
        const calls = wanted.map(model => {
            const body = {...sharedBody, model};
            if (model === "ldm") {
                body.num_inference_steps = parseInt(document.getElementById("cv-ldm-steps").value, 10);
                body.guidance_scale      = parseFloat(document.getElementById("cv-ldm-guidance").value);
                if (seed) body.seed = parseInt(seed, 10);
            }
            return fetch("/api/cv/translate", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(body),
            }).then(r => r.json().then(d => ({model, ok: r.ok, data: d})));
        });
        const results = await Promise.all(calls);
        renderCvCompareGallery(results, sharedBody);
    } finally {
        cmpBtn.disabled = false; genBtn.disabled = false;
        cmpBtn.textContent = "Compare both generators";
    }
}

function renderCvCompareGallery(results, sharedBody) {
    const section = document.getElementById("cv-results-section");
    const pair = document.getElementById("cv-result-pair");
    section.style.display = "";
    pair.innerHTML = "";

    const first = results.find(r => r.ok)?.data;
    if (!first) { pair.innerHTML = "<p>Both generators failed.</p>"; return; }

    const cells = [
        {label: "Input (MRA)", src: first.input_url, cls: "pasd-panel-mask"},
    ];
    for (const r of results) {
        if (!r.ok) {
            cells.push({label: `${r.model} (failed)`, src: "", cls: "pasd-panel-synth",
                        meta: {error: r.data?.error || "request failed"}});
        } else {
            cells.push({
                label: r.model === "ldm" ? "LDM CTA" : "Pix2pix CTA",
                src: r.data.output_url,
                cls: "pasd-panel-synth",
                sid: r.data.session_id,
            });
        }
    }
    if (first.patient != null && first.z != null) {
        cells.push({
            label: "Real CTA (reference)",
            src: `/api/cv/${first.patient}/slice/ct/${first.z}`,
            cls: "pasd-panel-real",
        });
    }

    for (const c of cells) {
        const div = document.createElement("div");
        div.className = `pasd-gen-cell ${c.cls}`;
        const fname = `cv_${c.label.replace(/\s+/g, "_").toLowerCase()}.png`;
        if (c.src) {
            div.innerHTML = `
                <div class="pasd-cell-label">${c.label}</div>
                <div class="pasd-cell-frame">
                    <img src="${c.src}" alt="${c.label}">
                    <button class="pasd-cell-zoom" title="View larger">&#128269;</button>
                    <a class="pasd-cell-dl" title="Download" href="${c.src}" download="${fname}">&#11015;</a>
                </div>`;
            const open = () => openPasdLightbox(c.src, c.src, c.label, fname);
            div.querySelector(".pasd-cell-frame img").addEventListener("click", open);
            div.querySelector(".pasd-cell-zoom").addEventListener("click", e => {
                e.stopPropagation(); open();
            });
        } else {
            div.innerHTML = `
                <div class="pasd-cell-label">${c.label}</div>
                <div class="pasd-cell-frame"><div class="pasd-cell-err">${c.meta.error}</div></div>`;
        }
        pair.appendChild(div);
    }
    section.scrollIntoView({behavior: "smooth", block: "nearest"});
}
