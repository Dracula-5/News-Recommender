// Single place the frontend learns where the backend API lives.
//
// Local dev: auto-detects localhost, no changes needed.
// Production (frontend on Netlify, backend on Render): after deploying
// the backend, replace the URL below with your Render service URL
// (e.g. "https://neurofeed-api.onrender.com" — no trailing slash), then
// redeploy the frontend. Every other file reads window.NEUROFEED_API_BASE
// instead of hardcoding a host, so this is the only line that changes.
(function () {
  const isLocal = ["localhost", "127.0.0.1"].includes(window.location.hostname);
  window.NEUROFEED_API_BASE = isLocal
    ? "http://localhost:8000"
    : "https://news-recommender-5wk4.onrender.com";
})();
