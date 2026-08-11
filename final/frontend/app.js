const api = "http://localhost:8000";

// ─── Authentication Check ──────────────────────────────────────

function initAuth() {
  const isGuest = localStorage.getItem('isGuest') === 'true';
  const userId = localStorage.getItem('userId');

  if (!userId && !isGuest) {
    window.location.href = 'login.html';
    return false;
  }
  return true;
}

if (!initAuth()) {
  throw new Error('Not authenticated');
}

let userId = "";

let queue = [];
let currentIndex = 0;
let startedAt = Date.now();
let attentionScore = 0.5;
let latestAttention = {
  brightness: 50, eye_openness: 70, movement: 20,
  energy_score: 63.5, gaze_score: 0.58, attention_score: 0.0,
  normalized_attention: 0.5, eye_aspect_ratio: 0.25,
  eye_movement: 0.0, face_distance: 0.5, distance_ok: true,
  head_movement: 0.0, face_detected: false, state: "no_face",
  source: "fallback", status: "idle",
};
let attentionInterval = null;   // /attention JSON polling (silent — drives scoring, never rendered)
let pollTimer = null;
let scrollDepth = 0;
let userMood = "";
let userLocation = "";
let userInterestText = "";        // used to pre-check the settings panel's interest boxes
let userExplorationSignal = 0.25; // used to pre-fill the settings panel's slider
let currentPollVal = 0.5;         // updated when user answers poll
let currentLongTermHistory = 0.2; // updated from user profile on each load

const $ = (id) => document.getElementById(id);

// ─── Toasts ─────────────────────────────────────────────────

function showToast(message, kind = "info", duration = 3200) {
  const stack = $("toastStack");
  if (!stack) return;
  const el = document.createElement("div");
  el.className = `toast toast-${kind}`;
  el.textContent = message;
  stack.appendChild(el);
  setTimeout(() => {
    el.classList.add("toast-out");
    setTimeout(() => el.remove(), 200);
  }, duration);
}

// ─── Loading bar ────────────────────────────────────────────

let loadingDepth = 0;

function beginLoading() {
  loadingDepth += 1;
  const bar = $("loadingBar");
  if (bar) { bar.classList.remove("done"); bar.classList.add("active"); }
}

function endLoading() {
  loadingDepth = Math.max(0, loadingDepth - 1);
  if (loadingDepth > 0) return;
  const bar = $("loadingBar");
  if (!bar) return;
  bar.classList.remove("active");
  bar.classList.add("done");
  setTimeout(() => bar.classList.remove("done"), 250);
}

// ─── Network helpers ──────────────────────────────────────────

async function request(method, path, body) {
  const headers = { "Content-Type": "application/json" };
  const token = localStorage.getItem('token');
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${api}${path}`, {
    method,
    headers,
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    if (res.status === 401) {
      localStorage.clear();
      window.location.href = 'login.html';
    }
    throw new Error(text);
  }
  return res.json();
}

const post = (path, body) => request("POST", path, body);
const patch = (path, body) => request("PATCH", path, body);

// GET with the bearer token attached, same as post() does for writes —
// a couple of reads used to skip this (harmless for guests, since the
// backend lets a tokenless request through, but inconsistent for a
// logged-in user who has a token to send).
async function authGet(path) {
  const headers = {};
  const token = localStorage.getItem('token');
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(`${api}${path}`, { headers });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ─── Data loading ─────────────────────────────────────────────

async function loadCategories() {
  try {
    const res = await fetch(`${api}/categories`);
    const data = await res.json();
    const container = $("interestOptions");
    container.innerHTML = "";
    (data.categories || []).forEach((cat) => {
      const label = document.createElement("label");
      label.className = "checkbox-item";
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.name = "interest"; cb.value = cat;
      if (["technology", "health", "sports"].includes(cat)) cb.checked = true;
      label.append(cb, document.createTextNode(cat));
      container.appendChild(label);
    });
  } catch (e) { console.warn("loadCategories:", e); }
}

async function loadUserIds() {
  try {
    const res = await fetch(`${api}/users`);
    const data = await res.json();
    const dl = $("userOptions");
    if (dl) {
      dl.innerHTML = "";
      (data.users || []).forEach((id) => {
        const opt = document.createElement("option"); opt.value = id;
        dl.appendChild(opt);
      });
    }
  } catch (e) { console.warn("loadUserIds:", e); }
}

// Loads the user's profile (mood / location / interaction history) so the
// recommender and live scoring keep adapting — mostly not rendered, except
// where the settings panel reuses it to pre-fill its fields.
async function loadUserInfo(uid) {
  if (!uid) return null;
  try {
    const u = await authGet(`/user/${encodeURIComponent(uid)}`);
    userMood = u.mood || userMood;
    userLocation = u.current_location || u.location || userLocation;
    userInterestText = u.user_interest_text || userInterestText;
    userExplorationSignal = Number(u.exploration_signal ?? userExplorationSignal);
    currentLongTermHistory = Number(u.interaction_score || 0.2);
    return u;
  } catch (e) { console.warn("loadUserInfo:", e); return null; }
}

function selectedInterests() {
  return Array.from(document.querySelectorAll("#interestOptions input:checked")).map(i => i.value);
}

// ─── Trending Now ───────────────────────────────────────────────
// Site-wide, not personalized — a separate signal from the adaptive feed
// below it, surfaced so there's always something to look at even before
// the recommender has any read on this particular user.

async function loadTrending() {
  const section = $("trendingSection");
  const row = $("trendingRow");
  if (!row) return;
  try {
    const res = await fetch(`${api}/trending?limit_categories=6&articles_per_category=1`);
    if (!res.ok) return;
    const data = await res.json();
    const sections = data.trending || [];
    if (!sections.length) { section?.classList.add("hidden"); return; }

    row.innerHTML = "";
    sections.forEach(({ category, articles }) => {
      const article = articles[0];
      if (!article) return;
      const card = document.createElement("button");
      card.type = "button";
      card.className = "trending-card";
      const catEl = document.createElement("span");
      catEl.className = "trending-card-cat";
      catEl.textContent = category;
      const headEl = document.createElement("span");
      headEl.className = "trending-card-headline";
      headEl.textContent = article.headline;
      card.append(catEl, headEl);
      card.addEventListener("click", () => openTrendingArticle(article));
      row.appendChild(card);
    });
    section?.classList.remove("hidden");
  } catch (e) { console.warn("loadTrending:", e); }
}

// Jumps straight to a trending article by splicing it into the queue right
// after the current position, rather than requiring a full /recommend
// round-trip — it's not personalized, but reading it still goes through
// the normal feedback loop like anything else in the feed.
function openTrendingArticle(article) {
  const alreadyQueued = queue.findIndex((q) => q.news_id === article.news_id);
  if (alreadyQueued !== -1) {
    currentIndex = alreadyQueued;
  } else {
    const insertAt = currentIndex + 1;
    queue.splice(insertAt, 0, { ...article, source_mix: "trending_dashboard", score: 0, similarity: 0, trending: 1 });
    currentIndex = insertAt;
  }
  paintCard();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ─── Attention (silent) ─────────────────────────────────────────
// The webcam is started automatically in the background the moment the feed
// opens — no button, no preview, no on-screen metrics. The resulting
// attention score still drives dwell-time pacing, auto-skip and feedback.
// The settings panel exposes a toggle to pause/resume it after the fact.

let attentionCaptureActive = false;

async function startAttentionCapture() {
  try {
    await post("/attention/start", {});
    attentionCaptureActive = true;
  } catch (e) { attentionCaptureActive = false; /* no camera available — fall back silently */ }
}

async function stopAttentionCapture() {
  try { await post("/attention/stop", {}); } catch (e) { /* already stopped, or no camera to begin with */ }
  attentionCaptureActive = false;
}

async function fetchAttention() {
  try {
    const res = await fetch(`${api}/attention`);
    if (!res.ok) return;
    const data = await res.json();
    latestAttention = { ...latestAttention, ...data };

    let norm = Number(data.normalized_attention);
    if (!Number.isFinite(norm)) {
      const raw = Number(data.attention_score || 0);
      norm = raw > 1 ? raw / 100 : raw;
    }
    attentionScore = Math.max(0, Math.min(1, norm));
  } catch (e) { /* attention endpoint not ready yet */ }
}

function startAttentionPolling() {
  stopAttentionPolling();
  attentionInterval = setInterval(fetchAttention, 1000);
}

function stopAttentionPolling() {
  clearInterval(attentionInterval);
  attentionInterval = null;
}

// ─── Article rendering ────────────────────────────────────────

function currentItem() { return queue[currentIndex]; }

// Deterministic, tasteful gradient per category — stands in for a photo
// since the dataset has no article images.
const HERO_PALETTES = [
  ["#1f2933", "#3e4c59"], ["#7a1f2b", "#b8121a"], ["#20344a", "#3a5a80"],
  ["#2c2a1f", "#5c5433"], ["#1f3a2e", "#2f6b4f"], ["#33202f", "#6b3a5a"],
  ["#2a2a2a", "#525252"], ["#1f2d3d", "#4a6a8a"],
];

function heroPalette(category) {
  const str = category || "news";
  let hash = 0;
  for (let i = 0; i < str.length; i++) hash = (hash * 31 + str.charCodeAt(i)) >>> 0;
  return HERO_PALETTES[hash % HERO_PALETTES.length];
}

function resetTimer() {
  startedAt = Date.now();
  currentPollVal = 0.5;
}

// Slides the previous card out, then paints the new one and slides it in.
// Skipped on the very first paint (nothing on screen yet to slide away).
function paintCard() {
  const item = currentItem();
  if (!item) return;
  const card = $("card");

  if (card && card.dataset.painted === "1") {
    card.classList.add("card-slide-out");
    setTimeout(() => {
      card.classList.remove("card-slide-out");
      paintCardContent(item);
      slideCardIn(card);
    }, 180);
  } else {
    paintCardContent(item);
    if (card) card.dataset.painted = "1";
  }
}

function slideCardIn(card) {
  card.classList.add("card-slide-in");
  // Two rAFs: let the browser paint the offset starting position (with
  // transition disabled by the .card-slide-in rule) before removing the
  // class, or the transition-back-to-normal has nothing to animate from.
  requestAnimationFrame(() => {
    requestAnimationFrame(() => card.classList.remove("card-slide-in"));
  });
}

function paintCardContent(item) {
  hidePoll();
  resetScrollTracking();
  scheduleInterestPoll();

  const category = item.category || "news";
  $("category").textContent = category;

  const [c1, c2] = heroPalette(category);
  const visual = $("heroVisual");
  if (visual) { visual.style.setProperty("--hc1", c1); visual.style.setProperty("--hc2", c2); }
  $("heroWatermark").textContent = category;

  $("headline").textContent = item.full_article
    ? item.full_article.split(".")[0]
    : item.abstract || item.news_id;
  $("abstract").textContent = item.abstract || item.full_article || "No summary available.";

  const urlEl = $("url");
  if (item.url) { urlEl.classList.remove("hidden"); urlEl.href = item.url; }
  else          { urlEl.classList.add("hidden"); }

  renderQueue();
  resetTimer();

  // Refresh the (invisible) attention reading for this new card.
  fetchAttention();

  if (item.hitl_decision === "auto_skip") {
    const nid = item.news_id;
    const delay = attentionScore < 0.35 || Number(item.score || 0) < 0.1 ? 9000 : 20000;
    setTimeout(() => { if (currentItem()?.news_id === nid) sendFeedback(0, 1, true); }, delay);
  }
}

function renderQueue() {
  const list = $("queueList");
  list.innerHTML = "";
  queue.forEach((item, i) => {
    const li = document.createElement("li");
    li.className = i === currentIndex ? "active" : "";

    const cat = document.createElement("span");
    cat.className = "update-cat";
    cat.textContent = item.category || "";

    const headline = document.createElement("span");
    headline.className = "update-headline";
    const raw = item.full_article ? item.full_article.split(".")[0] : (item.abstract || item.news_id || "");
    headline.textContent = raw.slice(0, 110);

    li.append(cat, headline);
    li.addEventListener("click", () => { currentIndex = i; paintCard(); });
    list.appendChild(li);
  });
}

// ─── Queue manipulation helpers ───────────────────────────────

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function reorderQueueAfterLike(likedCategory) {
  const before    = queue.slice(0, currentIndex);
  const remaining = queue.slice(currentIndex);

  const samecat = remaining.filter(item => item.category === likedCategory);
  const others  = remaining.filter(item => item.category !== likedCategory);

  const targetCount = Math.min(samecat.length, Math.floor(Math.random() * 2) + 2); // 2 or 3
  const promoted    = samecat.slice(0, targetCount);
  const restSamecat = samecat.slice(targetCount);

  const fillCount    = Math.max(0, 6 - promoted.length);
  const windowOthers = others.slice(0, fillCount);
  const afterWindow  = [...restSamecat, ...others.slice(fillCount)];

  const window6 = shuffle([...promoted, ...windowOthers]);

  queue = [...before, ...window6, ...shuffle(afterWindow)];
}

function suppressCategory(category) {
  const before    = queue.slice(0, currentIndex);
  const remaining = queue.slice(currentIndex);

  const window5   = remaining.slice(0, 5);
  const afterWindow = remaining.slice(5);

  const samecatInWindow = window5.filter(item => item.category === category);
  const othersInWindow  = window5.filter(item => item.category !== category);

  const keepOne = samecatInWindow.slice(0, 1);
  const extras  = samecatInWindow.slice(1);

  const newWindow5 = shuffle([...othersInWindow, ...keepOne]);

  queue = [...before, ...newWindow5, ...afterWindow, ...extras];
}

// A background-fetched batch, ready before the user actually runs out of
// articles — refreshRecommendations() reaches for this first so the "next
// batch" transition has no network gap most of the time. Only covers the
// plain (mode=null) refresh path; a forced/trending refresh always fetches
// fresh, which is fine — it's the uncommon path.
let prefetchedBatch = null;
let prefetchInFlight = false;

async function prefetchNextBatch() {
  if (prefetchInFlight || prefetchedBatch) return;
  prefetchInFlight = true;
  try {
    const data = await post("/recommend", {
      user_id: userId, k: 8, mode: null, location: userLocation, mood: userMood,
    });
    prefetchedBatch = data.recommendations || [];
  } catch (e) {
    // Silent — refreshRecommendations() will just fetch fresh when needed.
  } finally {
    prefetchInFlight = false;
  }
}

async function refreshRecommendations(mode = null) {
  let recommendations;
  if (!mode && prefetchedBatch) {
    recommendations = prefetchedBatch;
    prefetchedBatch = null;
  } else {
    beginLoading();
    try {
      const data = await post("/recommend", {
        user_id: userId, k: 8, mode, location: userLocation, mood: userMood,
      });
      recommendations = data.recommendations || [];
    } catch (e) {
      showToast("Couldn't load new recommendations — retrying next action.", "error");
      throw e;
    } finally {
      endLoading();
    }
  }
  queue = recommendations;
  currentIndex = 0;
  paintCard();
  await Promise.all([loadUserInfo(userId), loadUserIds()]);
}

// ─── Feedback ─────────────────────────────────────────────────

async function sendFeedback(liked, skipped = 0, automatic = false, forceTrending = false) {
  const item = currentItem();
  if (!item) return;

  const timeSpent = (Date.now() - startedAt) / 1000;

  await post("/feedback", {
    user_id:          userId,
    news_id:          item.news_id,
    time_spent:       timeSpent,
    liked,
    skipped,
    scroll_depth:     scrollDepth || (liked ? 0.9 : 0.15),
    click_val:        liked ? 1 : 0,
    attention_score:  attentionScore,
    normalized_attention: attentionScore,
    brightness:       latestAttention.brightness,
    eye_openness:     latestAttention.eye_openness,
    movement:         latestAttention.movement,
    energy_score:     latestAttention.energy_score,
    gaze_score:       latestAttention.gaze_score,
    face_detected:    latestAttention.face_detected ? 1 : 0,
    attention_source: latestAttention.source || "frontend",
    final_score:      item.score,
    similarity:       item.similarity,
    trending:         item.trending,
    poll_val:         currentPollVal,
  });

  currentIndex += 1;
  hidePoll();

  if (liked === 1) {
    reorderQueueAfterLike(item.category);
    if (!automatic) showToast("Noted — more like this coming up.", "success", 1800);
  } else if (!automatic) {
    suppressCategory(item.category);
    showToast("Got it — showing less of this.", "info", 1800);
  }

  await Promise.all([fetchAttention(), loadUserInfo(userId)]);

  // Kick off the next batch in the background once only a couple of
  // articles remain, so the eventual refresh below (or the next one)
  // usually finds it already there instead of waiting on the network.
  if (!forceTrending && queue.length - currentIndex <= 2) {
    prefetchNextBatch();
  }

  if (currentIndex >= queue.length || forceTrending || currentIndex % 3 === 0) {
    await refreshRecommendations(forceTrending ? "trending" : null);
  } else {
    paintCard();
  }
}

// ─── Poll ─────────────────────────────────────────────────────

function resetScrollTracking() {
  scrollDepth = 0;
  window.scrollTo({ top: 0, behavior: 'instant' });
}

function scheduleInterestPoll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(() => {
    if ($("pollPrompt")?.classList.contains("hidden") && (attentionScore < 0.55 || scrollDepth < 0.25)) {
      showPoll();
    }
  }, 20000);
}

function showPoll() {
  const item = currentItem();
  const nameEl = $("pollArticleName");
  if (nameEl && item) {
    const raw = item.full_article
      ? item.full_article.split(".")[0]
      : (item.abstract || item.news_id || "this article");
    const title = raw.slice(0, 65);
    nameEl.textContent = title + (raw.length > 65 ? "…" : "");
  }
  $("pollPrompt")?.classList.remove("hidden");
}
function hidePoll() { $("pollPrompt")?.classList.add("hidden"); clearTimeout(pollTimer); }

window.addEventListener("scroll", () => {
  const maxScroll = document.body.scrollHeight - window.innerHeight;
  if (maxScroll <= 0) return;
  scrollDepth = Math.max(scrollDepth, Math.min(1, window.scrollY / maxScroll));
});

async function sendPollFeedback(liked) {
  const item = currentItem();
  if (!item) return;
  currentPollVal = liked ? 1.0 : 0.0;
  hidePoll();
  try {
    await post("/poll_feedback", { user_id: userId, news_id: item.news_id, liked });
  } catch (e) { console.warn("Poll feedback failed:", e); }
  if (!liked) await sendFeedback(0, 1, true);
}

$("pollYes").addEventListener("click", () => sendPollFeedback(1));
$("pollNo").addEventListener("click",  () => sendPollFeedback(0));

// ─── Onboard ──────────────────────────────────────────────────

$("onboardForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData(e.currentTarget);
  const btn  = e.currentTarget.querySelector('button[type="submit"]');
  const orig = btn?.textContent;
  if (btn) { btn.disabled = true; btn.textContent = "Opening…"; }

  const interests = selectedInterests();
  const entered = String(form.get("user_id") || "").trim();
  userId = entered || `user_${Math.floor(Math.random() * 90000 + 10000)}`;

  try {
    await post("/onboard", {
      user_id:               userId,
      interests:             interests.length ? interests : [String(form.get("sample_click") || "general")],
      mood:                  form.get("mood"),
      time_available:        Number(form.get("time_available")),
      time_of_day:           form.get("time_of_day"),
      location:              form.get("location"),
      exploration_preference: Number(form.get("exploration_preference")),
      sample_click:          form.get("sample_click"),
    });

    $("onboarding").classList.add("hidden");
    $("feed").classList.remove("hidden");

    await loadUserInfo(userId);
    await refreshRecommendations();
    loadTrending();

    // Kick off silent background signals: camera-driven attention + always-on polling.
    startAttentionCapture();
    startAttentionPolling();
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = orig; }
  }
});

// ─── Button handlers ──────────────────────────────────────────

$("likeBtn").addEventListener("click", () => sendFeedback(1, 0));
$("skipBtn").addEventListener("click", () => sendFeedback(0, 1));
$("nextBtn").addEventListener("click", () => sendFeedback(0, 1, true));

// ─── Reading timer (updates the small byline label only) ──────

setInterval(() => {
  const elapsed = (Date.now() - startedAt) / 1000;
  const t = $("readTimer");
  if (t) t.textContent = elapsed < 2 ? "just now" : `reading · ${elapsed.toFixed(0)}s`;
  if (elapsed > 20 && currentItem()) showPoll();
}, 250);

// ─── User Menu ──────────────────────────────────────────────

const userMenuBtn = $("userMenuBtn");
const menuDropdown = $("menuDropdown");
const userEmail = $("userEmail");
const logoutLink = $("logoutLink");

const username = localStorage.getItem('username');
if (username && userEmail) userEmail.textContent = username;

if (userMenuBtn) {
  userMenuBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    menuDropdown.style.display = menuDropdown.style.display === "none" ? "block" : "none";
  });
}

document.addEventListener("click", () => {
  if (menuDropdown && menuDropdown.style.display !== "none") menuDropdown.style.display = "none";
});

if (logoutLink) {
  logoutLink.addEventListener("click", (e) => {
    e.preventDefault();
    // Best-effort server-side revoke so the token can't keep working if
    // it leaks after this — don't block navigation on it either way.
    const token = localStorage.getItem('token');
    if (token) {
      fetch(`${api}/auth/logout`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` },
      }).catch(() => {});
    }
    localStorage.clear();
    window.location.href = "login.html";
  });
}

// ─── Settings panel ─────────────────────────────────────────

async function openSettings() {
  const overlay = $("settingsOverlay");
  const panel = $("settingsPanel");
  if (!overlay || !panel) return;

  try {
    const res = await fetch(`${api}/categories`);
    const data = await res.json();
    const container = $("settingsInterestOptions");
    if (container) {
      container.innerHTML = "";
      const currentWords = userInterestText.toLowerCase().split(/\s+/);
      (data.categories || []).forEach((cat) => {
        const label = document.createElement("label");
        label.className = "checkbox-item";
        const cb = document.createElement("input");
        cb.type = "checkbox"; cb.name = "settings-interest"; cb.value = cat;
        cb.checked = currentWords.includes(cat.toLowerCase());
        label.append(cb, document.createTextNode(cat));
        container.appendChild(label);
      });
    }
  } catch (e) { console.warn("openSettings categories:", e); }

  const moodEl = $("settingsMood"); if (moodEl) moodEl.value = userMood || "";
  const locEl = $("settingsLocation"); if (locEl) locEl.value = userLocation || "";
  const expEl = $("settingsExploration"); if (expEl) expEl.value = String(userExplorationSignal);
  const toggle = $("settingsAttentionToggle"); if (toggle) toggle.checked = attentionCaptureActive;

  overlay.classList.remove("hidden");
  panel.classList.remove("hidden");
}

function closeSettings() {
  $("settingsOverlay")?.classList.add("hidden");
  $("settingsPanel")?.classList.add("hidden");
}

async function saveSettings() {
  const interests = Array.from(
    document.querySelectorAll("#settingsInterestOptions input:checked")
  ).map((i) => i.value);
  const mood = $("settingsMood")?.value.trim() || "";
  const location = $("settingsLocation")?.value.trim() || "";
  const exploration = Number($("settingsExploration")?.value ?? userExplorationSignal);
  const wantsAttention = !!$("settingsAttentionToggle")?.checked;

  const btn = $("settingsSave");
  const orig = btn?.textContent;
  if (btn) { btn.disabled = true; btn.textContent = "Saving…"; }

  try {
    await patch(`/user/${encodeURIComponent(userId)}/interests`, {
      interests,
      mood,
      location,
      exploration_preference: exploration,
    });

    if (wantsAttention && !attentionCaptureActive) {
      await startAttentionCapture();
    } else if (!wantsAttention && attentionCaptureActive) {
      await stopAttentionCapture();
    }

    await loadUserInfo(userId);
    closeSettings();
    showToast("Settings saved.", "success");
  } catch (e) {
    console.warn("saveSettings:", e);
    showToast("Couldn't save settings — please try again.", "error");
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = orig; }
  }
}

const settingsLink = $("settingsLink");
if (settingsLink) {
  settingsLink.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (menuDropdown) menuDropdown.style.display = "none";
    openSettings();
  });
}

$("settingsClose")?.addEventListener("click", closeSettings);
$("settingsOverlay")?.addEventListener("click", closeSettings);
$("settingsSave")?.addEventListener("click", saveSettings);

// ─── Init ─────────────────────────────────────────────────────

Promise.all([loadCategories(), loadUserIds()]).catch(console.error);
