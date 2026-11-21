const dataset = [];
let defaultThreshold = 0.05;
const SLIDER_MAX = 1000;
const ANNOTATOR_STORAGE_KEY = "patcher.annotatorId";

const annotatorState = {
  ids: [],
  activeId: null,
};

const patchState = {
  slideIdx: null,
  slideId: null,
  gridRows: 0,
  gridColumns: 0,
  levelWidth: 0,
  levelHeight: 0,
  patchSize: 0,
  patches: [],
  patchesByKey: new Map(),
  annotatedPatchCount: 0,
  manualEditCount: 0,
  hoveredKey: null,
  isDirty: false,
  isLoading: false,
  isSaving: false,
  isPainting: false,
  paintValue: null,
  paintedKeys: new Set(),
  lastPaintedCoords: null,
  activeLoadId: 0,
  pendingSlideIdx: null,
};

function patchKey(row, column) {
  return `${row},${column}`;
}

function getElements() {
  return {
    appFrame: document.querySelector(".app-frame"),
    annotatorList: document.querySelector("#annotator-list"),
    slidesList: document.querySelector("#slides-list"),
    slideTitle: document.querySelector("#slide-title"),
    annotationCount: document.querySelector("#annotation-count"),
    prevSlideButton: document.querySelector("#prev-slide-button"),
    nextSlideButton: document.querySelector("#next-slide-button"),
    thumbnailStage: document.querySelector("#thumbnail-stage"),
    thumbnail: document.querySelector("#thumbnail"),
    overlay: document.querySelector("#thumbnail-overlay"),
    patch: document.querySelector("#patch"),
    mouseCoords: document.querySelector("#mouse-coords"),
    thresholdSlider: document.querySelector("#threshold-slider"),
    thresholdInput: document.querySelector("#threshold-input"),
    outlineOnlyToggle: document.querySelector("#outline-only-toggle"),
    saveButton: document.querySelector("#save-button"),
    statusMessage: document.querySelector("#status-message"),
  };
}

function getUrlAnnotatorId() {
  return new URLSearchParams(window.location.search).get("annotator");
}

function setUrlAnnotatorId(annotatorId) {
  const url = new URL(window.location.href);
  url.searchParams.set("annotator", annotatorId);
  window.history.replaceState({}, "", url);
}

function getStoredAnnotatorId() {
  try {
    return window.localStorage.getItem(ANNOTATOR_STORAGE_KEY);
  } catch (error) {
    console.error("Failed to read annotator from local storage", error);
    return null;
  }
}

function storeAnnotatorId(annotatorId) {
  try {
    window.localStorage.setItem(ANNOTATOR_STORAGE_KEY, annotatorId);
  } catch (error) {
    console.error("Failed to save annotator to local storage", error);
  }
}

function setActiveAnnotator(annotatorId) {
  annotatorState.activeId = annotatorId;
  storeAnnotatorId(annotatorId);
  setUrlAnnotatorId(annotatorId);
}

function patchsetUrl(slideIdx) {
  const params = new URLSearchParams();
  if (annotatorState.activeId) {
    params.set("annotator_id", annotatorState.activeId);
  }
  const query = params.toString();
  return `/dataset/${slideIdx}/patchset${query ? `?${query}` : ""}`;
}

function populateAnnotatorList() {
  const { annotatorList } = getElements();
  annotatorList.innerHTML = "";

  annotatorState.ids.forEach((annotatorId) => {
    const option = document.createElement("option");
    option.value = annotatorId;
    option.textContent = annotatorId;
    annotatorList.appendChild(option);
  });

  if (annotatorState.activeId) {
    annotatorList.value = annotatorState.activeId;
  }
}

function initializeAnnotators(payload) {
  const configuredIds = Array.isArray(payload.annotators) ? payload.annotators : [];
  annotatorState.ids = configuredIds;
  const requestedId = getUrlAnnotatorId() || getStoredAnnotatorId();
  const defaultId = payload.default_annotator_id || configuredIds[0] || null;
  const activeId = configuredIds.includes(requestedId) ? requestedId : defaultId;

  if (!activeId) {
    throw new Error("No annotators are configured.");
  }

  setActiveAnnotator(activeId);
  populateAnnotatorList();
}

function setStatus(message, kind = "info") {
  const { statusMessage } = getElements();
  statusMessage.textContent = message;
  statusMessage.dataset.kind = kind;
}

function countAnnotatedPatches() {
  return patchState.annotatedPatchCount;
}

function recalculateAnnotatedPatchCount() {
  patchState.annotatedPatchCount = patchState.patches.reduce(
    (count, patch) => count + Number(Boolean(patch.human_tissue_annotation)),
    0,
  );
}

function incrementManualEditCount() {
  patchState.manualEditCount += 1;
}

function updateHeaderSummary() {
  const {
    slidesList,
    slideTitle,
    annotationCount,
    annotatorList,
    prevSlideButton,
    nextSlideButton,
    thresholdSlider,
    thresholdInput,
    outlineOnlyToggle,
    saveButton,
  } = getElements();

  const displayedSlideIdx = patchState.pendingSlideIdx ?? patchState.slideIdx;
  const isPendingTarget =
    patchState.pendingSlideIdx !== null &&
    patchState.pendingSlideIdx !== patchState.slideIdx;

  if (displayedSlideIdx === null) {
    slideTitle.textContent = "Loading slide…";
    annotationCount.textContent = "0 tissue / 0 patches";
  } else if (isPendingTarget || patchState.slideIdx === null) {
    slideTitle.textContent = currentSlideLabel(displayedSlideIdx);
    annotationCount.textContent = "Loading patches…";
  } else {
    slideTitle.textContent =
      patchState.slideId || dataset[patchState.slideIdx]?.slide_id || "Current slide";
    annotationCount.textContent =
      `${countAnnotatedPatches()} tissue / ${patchState.patches.length} patches`;
  }

  slideTitle.dataset.dirty = patchState.isDirty ? "true" : "false";

  const hasSlide = patchState.slideIdx !== null;
  const busy = patchState.isLoading || patchState.isSaving;

  annotatorList.disabled = busy || !annotatorState.ids.length;
  slidesList.disabled = busy || !dataset.length;
  if (displayedSlideIdx !== null) {
    slidesList.value = String(displayedSlideIdx);
  }

  prevSlideButton.disabled = busy || !hasSlide || patchState.slideIdx <= 0;
  nextSlideButton.disabled = busy || !hasSlide || patchState.slideIdx >= dataset.length - 1;

  thresholdSlider.disabled = busy || !hasSlide;
  thresholdInput.disabled = busy || !hasSlide;
  outlineOnlyToggle.disabled = patchState.isLoading;

  saveButton.disabled = busy || !hasSlide;
  saveButton.dataset.state = patchState.isDirty ? "dirty" : "clean";
  saveButton.textContent = patchState.isSaving ? "Saving…" : "Save";
}

function updateThresholdLabel() {
  const { thresholdInput } = getElements();
  const threshold = getCurrentThreshold();
  thresholdInput.value = threshold.toFixed(3);
}

function clampThreshold(value) {
  if (!Number.isFinite(value)) {
    return defaultThreshold;
  }

  return Math.min(1, Math.max(0, value));
}

function sliderValueToThreshold(sliderValue) {
  const normalized = clampThreshold(sliderValue / SLIDER_MAX);
  return normalized * normalized;
}

function thresholdToSliderValue(threshold) {
  return Math.round(Math.sqrt(clampThreshold(threshold)) * SLIDER_MAX);
}

function getCurrentThreshold() {
  const { thresholdSlider } = getElements();
  return sliderValueToThreshold(Number.parseFloat(thresholdSlider.value));
}

function setThreshold(threshold) {
  const { thresholdSlider } = getElements();
  thresholdSlider.value = String(thresholdToSliderValue(threshold));
  updateThresholdLabel();
}

function syncPatchLookup() {
  patchState.patchesByKey = new Map(
    patchState.patches.map((patch) => [patchKey(patch.row, patch.column), patch]),
  );
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (error) {
      console.error("Failed to read error response", error);
    }
    throw new Error(detail);
  }
  return response.json();
}

function populateSlidesList() {
  const { slidesList } = getElements();
  slidesList.innerHTML = "";

  dataset.forEach((row, index) => {
    const option = document.createElement("option");
    option.value = index;
    option.textContent = `${index + 1}. ${row.slide_id}`;
    slidesList.appendChild(option);
  });
}

function clearOverlayCanvas() {
  const { overlay } = getElements();
  const ctx = overlay.getContext("2d");
  if (!ctx) {
    return;
  }

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function clearPatchPreview() {
  const { patch } = getElements();
  patch.removeAttribute("src");
}

function setScrollableThumbnailViewport(enabled) {
  const { appFrame } = getElements();
  appFrame.dataset.scrollableThumbnail = enabled ? "true" : "false";
}

function resizeOverlayToThumbnail() {
  const { thumbnailStage, thumbnail, overlay } = getElements();

  if (thumbnail.naturalWidth && thumbnail.naturalHeight) {
    thumbnailStage.style.width = `${thumbnail.naturalWidth}px`;
    thumbnail.style.width = `${thumbnail.naturalWidth}px`;
    thumbnail.style.height = `${thumbnail.naturalHeight}px`;
  }

  const rect = thumbnail.getBoundingClientRect();
  if (!rect.width || !rect.height) {
    return;
  }

  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.round(rect.width * dpr);
  overlay.height = Math.round(rect.height * dpr);
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
}

function patchDisplayBounds(row, column, displayWidth, displayHeight) {
  if (!patchState.levelWidth || !patchState.levelHeight || !patchState.patchSize) {
    const cellWidth = displayWidth / patchState.gridColumns;
    const cellHeight = displayHeight / patchState.gridRows;
    return {
      left: column * cellWidth,
      top: row * cellHeight,
      width: cellWidth,
      height: cellHeight,
    };
  }

  const sourceLeft = Math.min(column * patchState.patchSize, patchState.levelWidth);
  const sourceRight = Math.min((column + 1) * patchState.patchSize, patchState.levelWidth);
  const sourceTop = Math.min(row * patchState.patchSize, patchState.levelHeight);
  const sourceBottom = Math.min((row + 1) * patchState.patchSize, patchState.levelHeight);

  return {
    left: (sourceLeft / patchState.levelWidth) * displayWidth,
    top: (sourceTop / patchState.levelHeight) * displayHeight,
    width: ((sourceRight - sourceLeft) / patchState.levelWidth) * displayWidth,
    height: ((sourceBottom - sourceTop) / patchState.levelHeight) * displayHeight,
  };
}

function drawOverlay() {
  const { overlay, outlineOnlyToggle, thumbnail } = getElements();
  const ctx = overlay.getContext("2d");
  if (!ctx) {
    return;
  }

  const dpr = window.devicePixelRatio || 1;
  const rect = thumbnail.getBoundingClientRect();
  const displayWidth = rect.width;
  const displayHeight = rect.height;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  if (!patchState.gridRows || !patchState.gridColumns) {
    return;
  }

  const showOutlinesOnly = outlineOnlyToggle.checked;

  patchState.patches.forEach((patch) => {
    const bounds = patchDisplayBounds(patch.row, patch.column, displayWidth, displayHeight);
    const isTissue = Boolean(patch.human_tissue_annotation);

    if (showOutlinesOnly) {
      if (!isTissue) {
        return;
      }

      ctx.strokeStyle = "rgba(12, 115, 98, 0.7)";
      ctx.lineWidth = 1.25;
      ctx.strokeRect(
        bounds.left + 0.5,
        bounds.top + 0.5,
        Math.max(bounds.width - 1, 0),
        Math.max(bounds.height - 1, 0),
      );
      return;
    }

    ctx.fillStyle = isTissue
      ? "rgba(12, 115, 98, 0.18)"
      : "rgba(159, 73, 47, 0.12)";
    ctx.strokeStyle = isTissue
      ? "rgba(12, 115, 98, 0.55)"
      : "rgba(159, 73, 47, 0.35)";
    ctx.lineWidth = 1;
    ctx.fillRect(bounds.left, bounds.top, bounds.width, bounds.height);
    ctx.strokeRect(
      bounds.left + 0.5,
      bounds.top + 0.5,
      Math.max(bounds.width - 1, 0),
      Math.max(bounds.height - 1, 0),
    );
  });

  if (!patchState.hoveredKey) {
    return;
  }

  const hoveredPatch = patchState.patchesByKey.get(patchState.hoveredKey);
  if (!hoveredPatch) {
    return;
  }

  const bounds = patchDisplayBounds(
    hoveredPatch.row,
    hoveredPatch.column,
    displayWidth,
    displayHeight,
  );
  ctx.strokeStyle = "rgba(21, 24, 33, 0.9)";
  ctx.lineWidth = 2;
  ctx.strokeRect(
    bounds.left + 1,
    bounds.top + 1,
    Math.max(bounds.width - 2, 0),
    Math.max(bounds.height - 2, 0),
  );
}

function patchCoordinatesFromPointerEvent(event) {
  const { overlay } = getElements();
  if (
    patchState.isLoading ||
    patchState.slideIdx === null ||
    !patchState.gridRows ||
    !patchState.gridColumns
  ) {
    return null;
  }

  const rect = overlay.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
    return null;
  }

  let column;
  let row;

  if (patchState.levelWidth && patchState.levelHeight && patchState.patchSize) {
    const sourceX = Math.min(
      patchState.levelWidth - 1e-6,
      Math.max(0, (x / rect.width) * patchState.levelWidth),
    );
    const sourceY = Math.min(
      patchState.levelHeight - 1e-6,
      Math.max(0, (y / rect.height) * patchState.levelHeight),
    );
    column = Math.min(
      patchState.gridColumns - 1,
      Math.floor(sourceX / patchState.patchSize),
    );
    row = Math.min(
      patchState.gridRows - 1,
      Math.floor(sourceY / patchState.patchSize),
    );
  } else {
    column = Math.min(
      patchState.gridColumns - 1,
      Math.floor((x / rect.width) * patchState.gridColumns),
    );
    row = Math.min(
      patchState.gridRows - 1,
      Math.floor((y / rect.height) * patchState.gridRows),
    );
  }

  return { x, y, row, column };
}

function updatePatchPreview(row, column) {
  if (patchState.slideIdx === null) {
    return;
  }

  const { patch } = getElements();
  patch.src = `/dataset/${patchState.slideIdx}/patch?x=${column}&y=${row}`;
}

function createInitialAnnotations(patches, threshold) {
  return patches.map((patch) => ({
    ...patch,
    human_tissue_annotation:
      patch.human_tissue_annotation !== null &&
      patch.human_tissue_annotation !== undefined
        ? Boolean(patch.human_tissue_annotation)
      : patch.tissue_score >= threshold,
  }));
}

function applyThresholdToAnnotations(threshold = getCurrentThreshold()) {
  if (!patchState.patches.length) {
    return;
  }

  patchState.patches.forEach((patch) => {
    patch.human_tissue_annotation = patch.tissue_score >= threshold;
  });
  recalculateAnnotatedPatchCount();
  patchState.isDirty = true;
  updateHeaderSummary();
  drawOverlay();
}

function setPatchAnnotationValue(patch, nextValue) {
  const currentValue = Boolean(patch.human_tissue_annotation);
  if (currentValue === nextValue) {
    return false;
  }

  patch.human_tissue_annotation = nextValue;
  patchState.annotatedPatchCount += nextValue ? 1 : -1;
  incrementManualEditCount();
  patchState.isDirty = true;
  return true;
}

function resetPaintStroke() {
  patchState.isPainting = false;
  patchState.paintValue = null;
  patchState.paintedKeys.clear();
  patchState.lastPaintedCoords = null;
}

function paintPatchAtGridPosition(row, column) {
  if (patchState.paintValue === null) {
    return false;
  }

  const key = patchKey(row, column);
  if (patchState.paintedKeys.has(key)) {
    return false;
  }

  const patch = patchState.patchesByKey.get(key);
  if (!patch) {
    return false;
  }

  patchState.paintedKeys.add(key);
  return setPatchAnnotationValue(patch, patchState.paintValue);
}

function paintStrokePath(startCoords, endCoords) {
  const path = [];
  let row = startCoords.row;
  let column = startCoords.column;
  const rowDelta = Math.abs(endCoords.row - startCoords.row);
  const columnDelta = Math.abs(endCoords.column - startCoords.column);
  const rowStep = row < endCoords.row ? 1 : -1;
  const columnStep = column < endCoords.column ? 1 : -1;
  let error = columnDelta - rowDelta;

  while (true) {
    path.push({ row, column });
    if (row === endCoords.row && column === endCoords.column) {
      return path;
    }

    const doubledError = error * 2;
    if (doubledError > -rowDelta) {
      error -= rowDelta;
      column += columnStep;
    }
    if (doubledError < columnDelta) {
      error += columnDelta;
      row += rowStep;
    }
  }
}

function paintPatchAtCoordinates(coords) {
  if (!coords || patchState.paintValue === null) {
    return false;
  }

  const startCoords = patchState.lastPaintedCoords ?? coords;
  let changed = false;

  paintStrokePath(startCoords, coords).forEach((point) => {
    changed = paintPatchAtGridPosition(point.row, point.column) || changed;
  });

  patchState.lastPaintedCoords = {
    row: coords.row,
    column: coords.column,
  };

  return changed;
}

function startPaintStroke(event) {
  if (event.button !== 0 || patchState.isLoading || patchState.isSaving) {
    return;
  }

  const coords = patchCoordinatesFromPointerEvent(event);
  if (!coords) {
    return;
  }

  const key = patchKey(coords.row, coords.column);
  const patch = patchState.patchesByKey.get(key);
  if (!patch) {
    return;
  }

  event.preventDefault();
  patchState.isPainting = true;
  patchState.paintValue = !Boolean(patch.human_tissue_annotation);
  patchState.paintedKeys.clear();
  patchState.lastPaintedCoords = null;
  patchState.hoveredKey = key;
  updatePatchPreview(coords.row, coords.column);
  paintPatchAtCoordinates(coords);
  updateHeaderSummary();
  drawOverlay();
  setStatus(
    `Painting ${patchState.paintValue ? "tissue" : "background"} on ${patchState.slideId}.`,
  );
}

function handlePointerMove(event) {
  if (patchState.isLoading) {
    const { mouseCoords } = getElements();
    mouseCoords.textContent = "";
    patchState.hoveredKey = null;
    return;
  }

  if (patchState.isPainting && event.buttons === 0) {
    resetPaintStroke();
  }

  const coords = patchCoordinatesFromPointerEvent(event);
  const { mouseCoords } = getElements();

  if (!coords) {
    mouseCoords.textContent = "";
    patchState.hoveredKey = null;
    drawOverlay();
    return;
  }

  patchState.hoveredKey = patchKey(coords.row, coords.column);
  if (!patchState.patchesByKey.has(patchState.hoveredKey)) {
    mouseCoords.textContent =
      `Mouse coordinates: (${Math.round(coords.x)}, ${Math.round(coords.y)})`;
    drawOverlay();
    return;
  }

  mouseCoords.textContent =
    `Mouse coordinates: (${Math.round(coords.x)}, ${Math.round(coords.y)}), ` +
    `Patch coordinates: (${coords.column}, ${coords.row})`;
  updatePatchPreview(coords.row, coords.column);
  if (patchState.isPainting && paintPatchAtCoordinates(coords)) {
    updateHeaderSummary();
  }
  drawOverlay();
}

function handlePointerLeave() {
  const { mouseCoords } = getElements();
  mouseCoords.textContent = "";
  patchState.hoveredKey = null;
  if (patchState.isLoading) {
    return;
  }
  drawOverlay();
}

function handlePointerUp() {
  resetPaintStroke();
}

function currentSlideLabel(slideIdx) {
  return dataset[slideIdx]?.slide_id || `slide ${slideIdx + 1}`;
}

function getAdjacentSlideIndex(direction) {
  if (patchState.slideIdx === null) {
    return null;
  }

  const targetIdx = patchState.slideIdx + direction;
  if (targetIdx < 0 || targetIdx >= dataset.length) {
    return null;
  }

  return targetIdx;
}

function confirmDiscardChanges() {
  if (!patchState.isDirty) {
    return true;
  }

  return window.confirm(
    `Discard unsaved changes for ${patchState.slideId || "the current slide"}?`,
  );
}

async function loadSlide(slideIdx) {
  const targetSlideIdx = Number(slideIdx);
  const { slidesList, thumbnail, mouseCoords } = getElements();
  const requestId = ++patchState.activeLoadId;

  patchState.isLoading = true;
  patchState.pendingSlideIdx = targetSlideIdx;
  resetPaintStroke();
  patchState.hoveredKey = null;
  mouseCoords.textContent = "";
  slidesList.value = String(targetSlideIdx);
  updateHeaderSummary();
  setStatus(`Loading ${currentSlideLabel(targetSlideIdx)}…`);
  clearOverlayCanvas();
  clearPatchPreview();

  try {
    const patchset = await fetchJson(patchsetUrl(targetSlideIdx));
    if (requestId !== patchState.activeLoadId) {
      return;
    }

    defaultThreshold = Number.parseFloat(
      patchset.annotation_threshold ??
        patchset.default_tissue_threshold ??
        defaultThreshold,
    );
    patchState.slideIdx = targetSlideIdx;
    patchState.slideId = patchset.slide_id;
    patchState.gridRows = patchset.grid_rows;
    patchState.gridColumns = patchset.grid_columns;
    patchState.levelWidth = patchset.level_width;
    patchState.levelHeight = patchset.level_height;
    patchState.patchSize = patchset.patch_size;
    patchState.patches = createInitialAnnotations(patchset.patches, defaultThreshold);
    recalculateAnnotatedPatchCount();
    patchState.manualEditCount = Number.parseInt(patchset.manual_edit_count ?? 0, 10);
    patchState.isDirty = false;
    resetPaintStroke();
    syncPatchLookup();
    setScrollableThumbnailViewport(Boolean(patchset.scrollable_thumbnail_viewport));
    setThreshold(defaultThreshold);
    updateHeaderSummary();

    thumbnail.src = `/dataset/${targetSlideIdx}/thumbnail`;
    await thumbnail.decode();
    if (requestId !== patchState.activeLoadId) {
      return;
    }

    resizeOverlayToThumbnail();
    drawOverlay();

    const firstPatch =
      patchState.patches.find((patch) => Boolean(patch.human_tissue_annotation)) ??
      patchState.patches[0];
    if (firstPatch) {
      updatePatchPreview(firstPatch.row, firstPatch.column);
    }

    setStatus(
      `Loaded ${patchState.slideId} for ${patchset.annotator_id} with ${patchState.patches.length} patches.`,
      "success",
    );
  } catch (error) {
    if (requestId === patchState.activeLoadId) {
      if (patchState.slideIdx !== null) {
        slidesList.value = String(patchState.slideIdx);
        drawOverlay();
      }
      throw error;
    }
  } finally {
    if (requestId === patchState.activeLoadId) {
      patchState.isLoading = false;
      patchState.pendingSlideIdx = null;
      updateHeaderSummary();
    }
  }
}

async function navigateToSlide(slideIdx) {
  const targetSlideIdx = Number(slideIdx);
  if (!Number.isInteger(targetSlideIdx)) {
    return;
  }

  if (targetSlideIdx < 0 || targetSlideIdx >= dataset.length) {
    return;
  }

  if (targetSlideIdx === patchState.slideIdx && patchState.patches.length) {
    return;
  }

  const wasDirty = patchState.isDirty;
  if (!confirmDiscardChanges()) {
    updateHeaderSummary();
    return;
  }

  if (wasDirty) {
    patchState.isDirty = false;
    updateHeaderSummary();
  }

  try {
    await loadSlide(targetSlideIdx);
  } catch (error) {
    console.error("Error loading slide", error);
    setStatus(error.message, "error");
  }
}

async function saveCurrentSlide() {
  if (patchState.slideIdx === null || patchState.isSaving) {
    return false;
  }

  patchState.isSaving = true;
  updateHeaderSummary();
  setStatus(`Saving ${patchState.slideId}…`);

  let wasSaved = false;
  try {
    const result = await fetchJson(patchsetUrl(patchState.slideIdx), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        annotation_threshold: getCurrentThreshold(),
        manual_edit_count: patchState.manualEditCount,
        annotations: patchState.patches.map((patch) => ({
          row: patch.row,
          column: patch.column,
          human_tissue_annotation: Boolean(patch.human_tissue_annotation),
        })),
      }),
    });

    patchState.isDirty = false;
    wasSaved = true;
    updateHeaderSummary();
    setStatus(result.detail, "success");
  } catch (error) {
    console.error("Error saving patchset", error);
    setStatus(error.message, "error");
  } finally {
    patchState.isSaving = false;
    updateHeaderSummary();
  }

  return wasSaved;
}

function bindEvents() {
  const {
    annotatorList,
    slidesList,
    prevSlideButton,
    nextSlideButton,
    thresholdSlider,
    thresholdInput,
    outlineOnlyToggle,
    saveButton,
    overlay,
    thumbnail,
  } = getElements();

  annotatorList.addEventListener("change", async (event) => {
    const nextAnnotatorId = event.target.value;
    if (!annotatorState.ids.includes(nextAnnotatorId)) {
      return;
    }

    if (nextAnnotatorId === annotatorState.activeId) {
      return;
    }

    if (!confirmDiscardChanges()) {
      annotatorList.value = annotatorState.activeId;
      updateHeaderSummary();
      return;
    }

    patchState.isDirty = false;
    setActiveAnnotator(nextAnnotatorId);
    updateHeaderSummary();

    if (patchState.slideIdx !== null) {
      try {
        await loadSlide(patchState.slideIdx);
      } catch (error) {
        console.error("Error loading annotator patchset", error);
        setStatus(error.message, "error");
      }
    }
  });

  slidesList.addEventListener("change", (event) => {
    navigateToSlide(event.target.value);
  });

  prevSlideButton.addEventListener("click", () => {
    const previousSlideIdx = getAdjacentSlideIndex(-1);
    if (previousSlideIdx !== null) {
      navigateToSlide(previousSlideIdx);
    }
  });

  nextSlideButton.addEventListener("click", () => {
    const nextSlideIdx = getAdjacentSlideIndex(1);
    if (nextSlideIdx !== null) {
      navigateToSlide(nextSlideIdx);
    }
  });

  thresholdSlider.addEventListener("input", () => {
    updateThresholdLabel();
    applyThresholdToAnnotations();
    setStatus(
      `Reinitialized ${patchState.slideId || "current slide"} from threshold ${getCurrentThreshold().toFixed(3)}.`,
    );
  });

  thresholdInput.addEventListener("change", () => {
    const threshold = clampThreshold(Number.parseFloat(thresholdInput.value));
    setThreshold(threshold);
    applyThresholdToAnnotations(threshold);
    setStatus(
      `Reinitialized ${patchState.slideId || "current slide"} from threshold ${threshold.toFixed(3)}.`,
    );
  });

  outlineOnlyToggle.addEventListener("change", () => {
    drawOverlay();
  });

  saveButton.addEventListener("click", () => {
    saveCurrentSlide();
  });

  overlay.addEventListener("mousedown", startPaintStroke);
  overlay.addEventListener("mousemove", handlePointerMove);
  overlay.addEventListener("mouseleave", handlePointerLeave);
  window.addEventListener("mouseup", handlePointerUp);

  thumbnail.addEventListener("load", () => {
    resizeOverlayToThumbnail();
    drawOverlay();
  });

  window.addEventListener("resize", () => {
    resizeOverlayToThumbnail();
    drawOverlay();
  });

  window.addEventListener("beforeunload", (event) => {
    if (!patchState.isDirty) {
      return;
    }

    event.preventDefault();
    event.returnValue = "";
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  setThreshold(defaultThreshold);
  updateHeaderSummary();
  bindEvents();

  try {
    const annotators = await fetchJson("/annotators");
    initializeAnnotators(annotators);
    updateHeaderSummary();

    const data = await fetchJson("/dataset");
    dataset.length = 0;
    dataset.push(...data);
    populateSlidesList();
    updateHeaderSummary();

    if (!dataset.length) {
      setStatus("No slides found in the dataset.", "error");
      return;
    }

    await loadSlide(0);
  } catch (error) {
    console.error("Error loading dataset index", error);
    setStatus(error.message, "error");
  }
});
