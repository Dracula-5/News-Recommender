// Signup page handler — pulled out of signup.html's inline <script> so the
// backend can serve a script-src CSP without 'unsafe-inline'. Depends on
// config.js (window.NEUROFEED_API_BASE) and auth.js (isValidEmail /
// setButtonLoading) loaded first.

const API_BASE = window.NEUROFEED_API_BASE || '';

document.getElementById('signupForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    await handleSignup();
});

document.getElementById('guestBtn').addEventListener('click', async () => {
    const guestId = 'guest_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('userId', guestId);
    localStorage.setItem('isGuest', 'true');
    window.location.href = 'index.html';
});

async function handleSignup() {
    const username = document.getElementById('username').value.trim();
    const email = document.getElementById('email').value.trim();
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    const signupBtn = document.getElementById('signupBtn');

    clearAllErrors();

    if (!username || username.length < 3) {
        document.getElementById('usernameError').textContent = 'Username must be at least 3 characters';
        return;
    }
    if (!/^[A-Za-z0-9_]+$/.test(username)) {
        document.getElementById('usernameError').textContent = 'Username may only contain letters, numbers, and underscores';
        return;
    }
    if (!email) {
        document.getElementById('emailError').textContent = 'Email is required';
        return;
    }
    if (!isValidEmail(email)) {
        document.getElementById('emailError').textContent = 'Invalid email format';
        return;
    }
    if (!isValidPassword(password)) {
        document.getElementById('passwordError').textContent = 'Password must be at least 8 characters and contain letters and numbers';
        return;
    }
    if (!passwordsMatch(password, confirmPassword)) {
        document.getElementById('confirmPasswordError').textContent = 'Passwords do not match';
        return;
    }

    setButtonLoading(signupBtn, true);

    try {
        const response = await fetch(`${API_BASE}/auth/signup`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, username, password }),
        });
        const data = await response.json();

        if (response.ok) {
            localStorage.setItem('userId', data.user_id);
            localStorage.setItem('username', data.username);
            localStorage.setItem('userEmail', data.email);
            localStorage.setItem('token', data.token);
            localStorage.removeItem('isGuest');
            window.location.href = 'index.html';
        } else {
            document.getElementById('generalError').textContent = data.detail || 'Signup failed. Please try again.';
        }
    } catch (error) {
        document.getElementById('generalError').textContent = 'Connection error. Please try again.';
        console.error('Signup error:', error);
    } finally {
        setButtonLoading(signupBtn, false);
    }
}
