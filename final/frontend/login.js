// Login page handler — pulled out of login.html's inline <script> so the
// backend can serve a script-src CSP without 'unsafe-inline'. Depends on
// config.js (window.NEUROFEED_API_BASE) and auth.js (isValidEmail /
// setButtonLoading) loaded first.

const API_BASE = window.NEUROFEED_API_BASE || '';

document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    await handleLogin();
});

document.getElementById('guestBtn').addEventListener('click', async () => {
    const guestId = 'guest_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('userId', guestId);
    localStorage.setItem('isGuest', 'true');
    window.location.href = 'index.html';
});

async function handleLogin() {
    const email = document.getElementById('email').value.trim();
    const password = document.getElementById('password').value;
    const loginBtn = document.getElementById('loginBtn');
    const generalError = document.getElementById('generalError');

    clearAllErrors();

    if (!email) {
        document.getElementById('emailError').textContent = 'Email is required';
        return;
    }
    if (!isValidEmail(email)) {
        document.getElementById('emailError').textContent = 'Invalid email format';
        return;
    }
    if (!password || password.length < 6) {
        document.getElementById('passwordError').textContent = 'Password must be at least 6 characters';
        return;
    }

    setButtonLoading(loginBtn, true);

    try {
        const response = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        });
        const data = await response.json();

        if (response.ok) {
            localStorage.setItem('userId', data.user_id);
            localStorage.setItem('username', data.username);
            localStorage.setItem('userEmail', data.email);
            localStorage.setItem('token', data.token);
            localStorage.removeItem('isGuest');
            window.location.href = 'index.html';
        } else if (response.status === 429) {
            generalError.textContent = data.detail || 'Too many attempts — please wait a moment and try again.';
        } else {
            generalError.textContent = data.detail || 'Login failed. Please try again.';
        }
    } catch (error) {
        generalError.textContent = 'Connection error. Please try again.';
        console.error('Login error:', error);
    } finally {
        setButtonLoading(loginBtn, false);
    }
}
