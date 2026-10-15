"use strict";
// Similar photos: a SigLIP 2 vector for each photo (vision service /embed), cached in IndexedDB.
// Groups the grid by similarity, sorts it by similarity to the open photo or to a description
// (like:"..."), and gives search a "similar" field. Loads after batch-view.js and before batch.js;
// uses both at runtime.

const EMBED_MODEL = "siglip2-base";
const EMBED_EDGE = 512; // SigLIP sees 224 px, so a small upload is enough
const SIMILAR_DEFAULT = 0.8;

// How alike photos must be to share a group, set with the Alike slider or typed in the box beside
// it. The slider covers only the useful part of 0 to 1; the box takes any value.
let similarValue = SIMILAR_DEFAULT;
const similarThreshold = () => similarValue;

function setSimilarThreshold(v) {
  similarValue = Math.round(v * 1000) / 1000; // 0.825, not 0.8250000001
  sim.value.value = String(similarValue);
  sim.range.value = similarValue;
}

const embedding = { active: false, stop: false, done: 0, total: 0 };
let similarCache = null; // {key, groups}

const sim = {
  wrap: $("batchSimilarWrap"),
  range: $("batchSimilar"),
  value: $("batchSimilarValue"),
  embedBtn: $("batchEmbed"),
  findBtn: $("focusSimilar"),
};

const dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

// Cosine similarity to the photo Find similar was pressed on (vectors are unit length)
function similarity(item) {
  const ref = view.vectors.get(view.similarTo);
  const v = view.vectors.get(item.fp);
  return ref && v ? dot(ref, v) : undefined;
}

// How well a photo matches the description in like:"..." (text and photo vectors share a space)
function likeMatch(item) {
  const v = view.vectors.get(item.fp);
  return view.like && v ? dot(view.like.vector, v) : undefined;
}

const similarInUse = () =>
  f.group.value === "similar" || f.sort.value === "similar" || /(^|\s)-?(similar|like)[:=<>]/i.test(f.search.value);

// ---------- vectors ----------
async function embedItem(item) {
  let dataUrl;
  try {
    dataUrl = (await prepareImage(item.file, EMBED_EDGE)).dataUrl;
  } catch {
    markUnsupported(item);
    return;
  }
  const r = await fetch("/api/vision/embed", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: EMBED_MODEL, image_data_url: dataUrl }),
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof d.detail === "string" ? d.detail : `HTTP ${r.status}`);
  const vector = Float32Array.from(d.vector);
  view.vectors.set(item.fp, vector);
  await resultStore.putVector(item.fp, vector);
}

// ---------- descriptions ----------
const textVectors = new Map(); // description -> vector, least recently used first
const textPending = new Set();

// The vector for a like:"..." description, or null while it is fetched; applyView runs again when it arrives
function textVector(text) {
  const vector = textVectors.get(text);
  if (vector) {
    textVectors.delete(text);
    textVectors.set(text, vector);
    return vector;
  }
  if (!textPending.has(text)) fetchTextVector(text);
  return null;
}

async function fetchTextVector(text) {
  textPending.add(text);
  try {
    const r = await fetch("/api/vision/embed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: EMBED_MODEL, text }),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(typeof d.detail === "string" ? d.detail : `HTTP ${r.status}`);
    textVectors.set(text, Float32Array.from(d.vector));
    if (textVectors.size > 50) textVectors.delete(textVectors.keys().next().value);
    applyView(); // reads the search box again, so a description typed over since is left alone
  } catch (e) {
    setStatus(`Could not search by description: ${e.message}`, "error");
  } finally {
    textPending.delete(text);
  }
}

// Two at a time, like reading EXIF: the vision service runs one at a time, and the other request
// overlaps it with decoding and uploading the next photo
async function embedMissing() {
  if (embedding.active) return;
  const items = batch.items;
  const todo = items.filter((item) => !item.unsupported && !view.vectors.has(item.fp));
  if (!todo.length) return;
  Object.assign(embedding, { active: true, stop: false, done: 0, total: todo.length });
  updateSimilarUi();
  let error = null;
  const worker = async () => {
    while (todo.length && !embedding.stop && !error && items === batch.items) {
      try {
        await embedItem(todo.shift());
      } catch (e) {
        error = e; // the vision service is down or refused: stop rather than fail every photo
      }
      embedding.done++;
      updateSimilarUi();
      if (embedding.done % 25 === 0) applyView();
    }
  };
  try {
    await Promise.all([worker(), worker()]);
  } finally {
    embedding.active = false;
  }
  if (error) setStatus(`Could not compare photos: ${error.message}`, "error");
  applyView();
}

// Called by loadViewData() once a folder's cached vectors are in
function similarLoaded() {
  if (similarInUse()) embedMissing();
  updateSimilarUi();
}

// ---------- grouping ----------
// Average linkage: a photo joins the group it is most alike on average (its mean similarity to the
// members), if that is at least threshold, else starts its own. Unlike comparing with a group's
// average photo, this doesn't let a big group draw in everything. Pass 1 goes in folder order, so
// bursts build a group together; pass 2 moves each photo to the group it is now most alike, so an
// early photo isn't stuck in a group that later grew away from it.
function similarGroups(indices) {
  const threshold = similarThreshold();
  const unsupported = batch.items.filter((item) => item.unsupported).length;
  const key = `${threshold}|${view.vectors.size}|${unsupported}|${indices.join(",")}`;
  if (similarCache?.key === key) return similarCache.groups;

  const ordered = indices.slice().sort((a, b) => a - b);
  const members = ordered.filter((i) => view.vectors.has(batch.items[i].fp));
  const vecs = members.map((i) => view.vectors.get(batch.items[i].fp));
  const sums = [];   // per group, the sum of its members' vectors
  const counts = [];
  // own: the group v is in now, which leaves v itself out of the mean
  const best = (v, own = -1) => {
    let c = -1;
    let top = threshold;
    for (let k = 0; k < sums.length; k++) {
      const n = counts[k] - (k === own);
      if (!n) continue;
      const s = (dot(v, sums[k]) - (k === own)) / n;
      if (s >= top) [c, top] = [k, s];
    }
    return c;
  };
  const assign = vecs.map((v) => {
    let c = best(v);
    if (c < 0) {
      sums.push(new Float32Array(v.length));
      counts.push(0);
      c = sums.length - 1;
    }
    const sum = sums[c];
    for (let d = 0; d < v.length; d++) sum[d] += v[d];
    counts[c]++;
    return c;
  });
  vecs.forEach((v, n) => {
    const c = best(v, assign[n]);
    if (c >= 0) assign[n] = c;
  });

  const cluster = new Map(members.map((i, n) => [i, assign[n]]));
  const byCluster = new Map();
  for (const i of indices) { // in display order, so a sort still applies inside each group
    if (!cluster.has(i)) continue;
    const c = cluster.get(i);
    if (!byCluster.has(c)) byCluster.set(c, []);
    byCluster.get(c).push(i);
  }
  const lists = [...byCluster.values()].sort((a, b) => b.length - a.length);
  const groups = lists.filter((g) => g.length > 1).map((g, n) => ({ label: groupLabel(n + 1, g), indices: g }));
  const alone = lists.filter((g) => g.length === 1).flat();
  if (alone.length) groups.push({ label: "(no similar photo)", indices: alone });
  const missing = indices.filter((i) => !cluster.has(i));
  const unreadable = missing.filter((i) => batch.items[i].unsupported); // e.g. TIFF or HEIC outside Safari
  const pending = missing.filter((i) => !batch.items[i].unsupported);
  if (pending.length) groups.push({ label: "(not compared yet)", indices: pending });
  if (unreadable.length) groups.push({ label: "(browser can't open)", indices: unreadable });
  similarCache = { key, groups };
  return groups;
}

// "Similar 3 · mostly beach" when at least half the group shares a category
function groupLabel(n, indices) {
  const counts = new Map();
  for (const i of indices) {
    const c = fieldValues(batch.items[i], "category")?.[0];
    if (c !== undefined) counts.set(String(c), (counts.get(String(c)) || 0) + 1);
  }
  const [top, count] = [...counts].sort((a, b) => b[1] - a[1])[0] || [];
  return top && count * 2 >= indices.length ? `Similar ${n} · mostly ${top}` : `Similar ${n}`;
}

// ---------- UI ----------
function updateSimilarUi() {
  sim.wrap.hidden = f.group.value !== "similar";
  const missing = batch.items.filter((item) => !item.unsupported && !view.vectors.has(item.fp)).length;
  sim.embedBtn.hidden = !embedding.active && !(similarInUse() && missing);
  sim.embedBtn.textContent = embedding.active
    ? `comparing ${embedding.done} of ${embedding.total}… stop`
    : `compare photos (${missing})`;
  sim.embedBtn.title = embedding.active
    ? "Stop comparing; photos done so far are kept"
    : "Send each photo once to SigLIP 2 so similar photos can be found. Cached in this browser.";
}

sim.embedBtn.addEventListener("click", () => {
  if (embedding.active) embedding.stop = true;
  else embedMissing();
});

for (const [x, event] of [[f.group, "change"], [f.sort, "change"], [f.search, "input"]]) {
  x.addEventListener(event, () => {
    if (similarInUse()) embedMissing();
  });
}

let similarTimer = null;
sim.range.addEventListener("input", () => {
  setSimilarThreshold(Number(sim.range.value));
  clearTimeout(similarTimer);
  similarTimer = setTimeout(applyView, 150);
});
// Typed values apply as you type, once they are a number from 0 to 1
sim.value.addEventListener("input", () => {
  const v = Number(sim.value.value);
  if (sim.value.value === "" || !(v >= 0 && v <= 1)) return;
  similarValue = v;
  sim.range.value = v;
  clearTimeout(similarTimer);
  similarTimer = setTimeout(applyView, 300);
});
// Leaving the box tidies it: an empty or out-of-range entry goes back to the value in use
sim.value.addEventListener("change", () => setSimilarThreshold(similarThreshold()));

// Sort by similarity to the open photo; ← and → then step through the closest ones
sim.findBtn.addEventListener("click", async () => {
  const item = batch.items[batch.focus];
  if (!item) return;
  if (!view.vectors.has(item.fp)) {
    setStatus(`Comparing ${item.file.name}…`);
    try {
      await embedItem(item);
    } catch (e) {
      setStatus(`Could not compare photos: ${e.message}`, "error");
      return;
    }
    if (!view.vectors.has(item.fp) || batch.items[batch.focus] !== item) return;
  }
  view.similarTo = item.fp;
  f.sort.value = "similar";
  setSortDir("desc");
  applyView();
  updateFocusNav();
  setStatus(`Sorted by similarity to ${item.file.name}. ← and → step through the closest photos.`);
  embedMissing();
});

(() => {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(VIEW_UI_KEY)) || {};
  } catch { /* storage unavailable */ }
  const v = Number(saved.similar);
  setSimilarThreshold(v > 0 && v <= 1 ? v : SIMILAR_DEFAULT); // the rest waits for a folder (batch.js loads later)
})();
