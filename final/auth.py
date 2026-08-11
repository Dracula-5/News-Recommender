"""
Authentication module for NeuroFeed
- User database schema
- Password hashing and verification
- JWT token generation
"""

import sqlite3
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

from config import get_settings

# Previously a hardcoded constant with no way to override it — every test
# that touched signup/login was silently reading and writing the real dev
# auth.sqlite (the isolated-test-DB setup in tests/conftest.py only ever
# covered the main news_recommender.sqlite via NEUROFEED_DB_PATH). Deriving
# this from settings means NEUROFEED_AUTH_DB_PATH actually isolates it.
AUTH_DB_PATH: Path = get_settings().auth_db_path


def _utcnow() -> datetime:
    """
    Naive UTC datetime (matches what's already stored/compared here).
    `datetime.utcnow()` is deprecated as of 3.12; this is its non-deprecated
    equivalent, not a behavior change — the whole module compares these as
    naive datetimes, so switching to timezone-aware ones everywhere would
    be a larger, riskier change than fixing the deprecation warrants.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def init_auth_db():
    """Initialize authentication database"""
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS auth_users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        username TEXT UNIQUE,
        password_hash TEXT NOT NULL,
        user_id TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active BOOLEAN DEFAULT 1
    )
    """)

    # Add username column to older tables if missing
    cursor.execute("PRAGMA table_info(auth_users)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'username' not in columns:
        cursor.execute('ALTER TABLE auth_users ADD COLUMN username TEXT')
        cursor.execute("UPDATE auth_users SET username = user_id WHERE username IS NULL OR username = ''")

    # Ensure existing rows have a username set
    cursor.execute("UPDATE auth_users SET username = user_id WHERE username IS NULL OR username = ''")
    
    # Auth tokens table (for future JWT management)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS auth_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        token TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP NOT NULL,
        is_valid BOOLEAN DEFAULT 1,
        FOREIGN KEY (user_id) REFERENCES auth_users(user_id)
    )
    """)
    
    # Create indexes for faster lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_email ON auth_users(email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON auth_users(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_username ON auth_users(username)")
    
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    """Hash password with salt"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    return f"{salt}${password_hash.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    try:
        salt, stored_hash = password_hash.split('$')
        password_check = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        # Constant-time comparison — a plain `==` on strings short-circuits
        # on the first mismatched byte, which leaks a timing signal an
        # attacker can use to recover the hash (and thus brute-force the
        # password) byte-by-byte instead of needing the whole keyspace.
        return hmac.compare_digest(password_check.hex(), stored_hash)
    except Exception:
        return False


def generate_token() -> str:
    """Generate secure auth token"""
    return secrets.token_urlsafe(32)


def generate_user_id(username: str = None) -> str:
    """Generate unique user ID from username or random"""
    if username:
        return username
    return 'U' + secrets.token_hex(8).upper()


def user_exists(email: str) -> bool:
    """Check if user exists by email"""
    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM auth_users WHERE email = ?", (email,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def username_exists(username: str) -> bool:
    """Check if username already exists"""
    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM auth_users WHERE username = ?", (username,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def signup_user(email: str, password: str, username: str = None) -> dict:
    """
    Create new user account
    
    Args:
        email: User email
        password: User password
        username: Display username (optional, generates if not provided)
    
    Returns:
        {
            'success': bool,
            'message': str,
            'user_id': str (if success),
            'username': str (if success),
            'token': str (if success)
        }
    """
    if user_exists(email):
        return {'success': False, 'message': 'Email already registered'}

    if username and username_exists(username):
        return {'success': False, 'message': 'Username already taken'}
    
    try:
        # Use provided username or generate one
        if not username:
            username = 'User' + secrets.token_hex(4).upper()
        
        # Use username as user_id
        user_id = generate_user_id(username)
        password_hash = hash_password(password)
        token = generate_token()
        
        conn = sqlite3.connect(AUTH_DB_PATH)
        cursor = conn.cursor()
        
        # Insert user
        cursor.execute("""
        INSERT INTO auth_users (email, username, password_hash, user_id)
        VALUES (?, ?, ?, ?)
        """, (email, username, password_hash, user_id))
        
        # Insert token
        expires_at = _utcnow() + timedelta(days=30)
        cursor.execute("""
        INSERT INTO auth_tokens (user_id, token, expires_at)
        VALUES (?, ?, ?)
        """, (user_id, token, expires_at.isoformat(sep=" ")))

        conn.commit()
        conn.close()
        
        return {
            'success': True,
            'message': 'User created successfully',
            'user_id': user_id,
            'username': username,
            'email': email,
            'token': token
        }
    except Exception as e:
        return {'success': False, 'message': f'Signup failed: {str(e)}'}


def login_user(email: str, password: str) -> dict:
    """
    Authenticate user
    
    Returns:
        {
            'success': bool,
            'message': str,
            'user_id': str (if success),
            'token': str (if success),
            'email': str (if success)
        }
    """
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT id, email, username, password_hash, user_id, is_active
    FROM auth_users
    WHERE email = ?
    """, (email,))
    
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return {'success': False, 'message': 'Invalid email or password'}
    
    if not user['is_active']:
        return {'success': False, 'message': 'Account is inactive'}
    
    if not verify_password(password, user['password_hash']):
        return {'success': False, 'message': 'Invalid email or password'}
    
    try:
        token = generate_token()
        
        conn = sqlite3.connect(AUTH_DB_PATH)
        cursor = conn.cursor()
        
        # Insert new token
        expires_at = _utcnow() + timedelta(days=30)
        cursor.execute("""
        INSERT INTO auth_tokens (user_id, token, expires_at)
        VALUES (?, ?, ?)
        """, (user['user_id'], token, expires_at.isoformat(sep=" ")))
        
        conn.commit()
        conn.close()
        
        return {
            'success': True,
            'message': 'Login successful',
            'user_id': user['user_id'],
            'username': user['username'],
            'email': user['email'],
            'token': token
        }
    except Exception as e:
        return {'success': False, 'message': f'Login failed: {str(e)}'}


def verify_token(token: str) -> dict:
    """Verify auth token"""
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT user_id, expires_at, is_valid
    FROM auth_tokens
    WHERE token = ?
    """, (token,))
    
    token_row = cursor.fetchone()
    conn.close()
    
    if not token_row or not token_row['is_valid']:
        return {'valid': False}
    
    expires_at = datetime.fromisoformat(token_row['expires_at'])
    if _utcnow() > expires_at:
        return {'valid': False, 'message': 'Token expired'}
    
    return {'valid': True, 'user_id': token_row['user_id']}


def revoke_token(token: str) -> bool:
    """
    Invalidate a single token (server-side logout).

    Before this, "logging out" was purely a client-side
    `localStorage.clear()` — the token itself stayed marked valid in
    auth_tokens until its 30-day expiry, so a token that had leaked (XSS,
    a shared/public machine, a synced browser profile) kept working long
    after the user thought they'd logged out, with no way to cut it off.
    """
    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE auth_tokens SET is_valid = 0 WHERE token = ?", (token,))
    revoked = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return revoked


def revoke_all_tokens_for_user(user_id: str) -> int:
    """Invalidate every token for a user — e.g. after a password change."""
    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE auth_tokens SET is_valid = 0 WHERE user_id = ? AND is_valid = 1", (user_id,))
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count


def get_user_by_id(user_id: str) -> dict:
    """Get user info by user_id"""
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT id, email, username, user_id, created_at
    FROM auth_users
    WHERE user_id = ?
    """, (user_id,))
    
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return None
    
    return dict(user)
