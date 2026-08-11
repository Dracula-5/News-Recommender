/* ─────────────────────────────────────────────────────────────────── */
/* AUTHENTICATION UTILITY FUNCTIONS                                     */
/* ─────────────────────────────────────────────────────────────────── */

/**
 * Validate email format
 */
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

/**
 * Validate password strength
 */
function isValidPassword(password) {
    return password && password.length >= 8 && /[a-zA-Z]/.test(password) && /[0-9]/.test(password);
}

/**
 * Check if passwords match
 */
function passwordsMatch(password, confirmPassword) {
    return password === confirmPassword;
}

/**
 * Set button loading state
 */
function setButtonLoading(btn, isLoading) {
    if (isLoading) {
        btn.disabled = true;
        btn.classList.add('loading');
    } else {
        btn.disabled = false;
        btn.classList.remove('loading');
    }
}

/**
 * Clear all error messages
 */
function clearAllErrors() {
    const errorElements = document.querySelectorAll('.error-message');
    errorElements.forEach(el => el.textContent = '');
}

/**
 * Get auth token from storage
 */
function getAuthToken() {
    return localStorage.getItem('token');
}

/**
 * Get user ID from storage
 */
function getUserId() {
    return localStorage.getItem('userId');
}

/**
 * Check if user is authenticated
 */
function isAuthenticated() {
    return !!localStorage.getItem('userId') && !localStorage.getItem('isGuest');
}

/**
 * Check if user is guest
 */
function isGuest() {
    return localStorage.getItem('isGuest') === 'true';
}

/**
 * Logout and clear storage
 */
function logout() {
    localStorage.removeItem('userId');
    localStorage.removeItem('userEmail');
    localStorage.removeItem('token');
    localStorage.removeItem('isGuest');
    window.location.href = 'login.html';
}

/**
 * Make authenticated fetch request
 */
async function authFetch(url, options = {}) {
    const token = getAuthToken();
    const headers = {
        'Content-Type': 'application/json',
        ...options.headers
    };

    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    return fetch(url, {
        ...options,
        headers
    });
}

/**
 * Redirect to login if not authenticated
 */
function redirectIfNotAuthenticated() {
    if (!getUserId()) {
        window.location.href = 'login.html';
    }
}

/**
 * Display notification/toast
 */
function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 8px;
        z-index: 9999;
        animation: slideInRight 0.3s ease-out;
        font-size: 14px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    `;

    if (type === 'success') {
        notification.style.backgroundColor = '#10b981';
        notification.style.color = 'white';
    } else if (type === 'error') {
        notification.style.backgroundColor = '#ef4444';
        notification.style.color = 'white';
    } else {
        notification.style.backgroundColor = '#3b82f6';
        notification.style.color = 'white';
    }

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

// Add animation styles
if (!document.getElementById('auth-animations-style')) {
    const style = document.createElement('style');
    style.id = 'auth-animations-style';
    style.textContent = `
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes slideOutRight {
            from {
                opacity: 1;
                transform: translateX(0);
            }
            to {
                opacity: 0;
                transform: translateX(30px);
            }
        }
    `;
    document.head.appendChild(style);
}
