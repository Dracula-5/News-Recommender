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
let currentPollVal = 0.5;         // updated when user answers poll
let currentLongTermHistory = 0.2; // updated from user profile on each load

const $ = (id) => document.getElementById(id);

// ─── Network helpers ──────────────────────────────────────────

async function post(path, body) {
  const headers = { "Content-Type": "application/json" };
  const token = localStorage.getItem('token');
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${api}${path}`, {
    method: "POST",
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
// recommender and live scoring keep adapting — none of this is rendered.
async function loadUserInfo(uid) {
  if (!uid) return;
  try {
    const res = await fetch(`${api}/user/${encodeURIComponent(uid)}`);
    if (!res.ok) return;
    const u = await res.json();
    userMood = u.mood || userMood;
    userLocation = u.current_location || u.location || userLocation;
    currentLongTermHistory = Number(u.interaction_score || 0.2);
  } catch (e) { console.warn("loadUserInfo:", e); }
}

function selectedInterests() {
  return Array.from(document.querySelectorAll("#interestOptions input:checked")).map(i => i.value);
}

// ─── Attention (silent) ─────────────────────────────────────────
// The webcam is started automatically in the background the moment the feed
// opens — no button, no preview, no on-screen metrics. The resulting
// attention score still drives dwell-time pacing, auto-skip and feedback.

async function startAttentionCapture() {
  try { await post("/attention/start", {}); } catch (e) { /* no camera available — fall back silently */ }
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

function paintCard() {
  const item = currentItem();
  if (!item) return;
  hidePoll();
  resetScrollTracking();
  scheduleInterestPoll();

  const card = $("card");
  if (card) { card.style.animation = "none"; void card.offsetWidth; card.style.animation = ""; }

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

async function refreshRecommendations(mode = null) {
  const data = await post("/recommend", {
    user_id: userId,
    k: 8,
    mode,
    location: userLocation,
    mood: userMood,
  });
  queue = data.recommendations || [];
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
  } else if (!automatic) {
    suppressCategory(item.category);
  }

  await Promise.all([fetchAttention(), loadUserInfo(userId)]);

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

// ─── Init ─────────────────────────────────────────────────────

Promise.all([loadCategories(), loadUserIds()]).catch(console.error);
