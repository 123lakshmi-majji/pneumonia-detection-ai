import sqlite3
import os
import hashlib
import secrets

DB_PATH = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Users table (with all columns)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT DEFAULT '',
            bio TEXT DEFAULT '',
            profile_pic TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Migration: add missing columns to existing users table
    cursor.execute("PRAGMA table_info(users)")
    existing_columns = [col[1] for col in cursor.fetchall()]
    if 'full_name' not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN full_name TEXT DEFAULT ''")
    if 'bio' not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN bio TEXT DEFAULT ''")
    if 'profile_pic' not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN profile_pic TEXT DEFAULT ''")
    if 'created_at' not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP")

    # History table with user_id
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            gradcam_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Add user_id to history if missing
    cursor.execute("PRAGMA table_info(history)")
    hist_columns = [col[1] for col in cursor.fetchall()]
    if 'user_id' not in hist_columns:
        cursor.execute("ALTER TABLE history ADD COLUMN user_id INTEGER REFERENCES users(id)")

    # Contact messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contact_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            subject TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

def log_prediction(user_id, filename, predicted_class, confidence, gradcam_path=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO history (user_id, filename, predicted_class, confidence, gradcam_path)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, filename, predicted_class, confidence, gradcam_path))
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id

def get_history(user_id=None, limit=50):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if user_id is not None:
        cursor.execute("""
            SELECT id, filename, predicted_class, confidence, gradcam_path, timestamp
            FROM history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))
    else:
        cursor.execute("""
            SELECT id, filename, predicted_class, confidence, gradcam_path, timestamp
            FROM history
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    history_list = []
    for r in rows:
        history_list.append({
            "id": r[0], "filename": r[1], "predicted_class": r[2],
            "confidence": r[3], "gradcam_path": r[4], "timestamp": r[5]
        })
    return history_list

def get_record(record_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, predicted_class, confidence, gradcam_path, timestamp FROM history WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0], "filename": row[1], "predicted_class": row[2],
            "confidence": row[3], "gradcam_path": row[4], "timestamp": row[5]
        }
    return None

def get_user_stats(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM history WHERE user_id = ?", (user_id,))
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM history WHERE user_id = ? AND predicted_class='Normal'", (user_id,))
    normal = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM history WHERE user_id = ? AND predicted_class='Bacterial Pneumonia'", (user_id,))
    bacterial = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM history WHERE user_id = ? AND predicted_class='Viral Pneumonia'", (user_id,))
    viral = cursor.fetchone()[0]
    conn.close()
    return {"total": total, "normal": normal, "bacterial": bacterial, "viral": viral}

# ---------- User management ----------
def hash_password(password):
    salt = secrets.token_hex(16)
    hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}${hash_obj.hex()}"

def verify_password(stored_password, provided_password):
    salt, hash_str = stored_password.split('$')
    hash_obj = hashlib.pbkdf2_hmac('sha256', provided_password.encode(), salt.encode(), 100000)
    return hash_obj.hex() == hash_str

def create_user(username, email, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    hashed = hash_password(password)
    try:
        cursor.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                       (username, email, hashed))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None

def get_user_by_username(username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Select all columns safely (they exist now after migration)
    cursor.execute("SELECT id, username, email, password_hash, full_name, bio, profile_pic FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0], "username": row[1], "email": row[2], "password_hash": row[3],
            "full_name": row[4], "bio": row[5], "profile_pic": row[6]
        }
    return None

def get_user_by_id(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, full_name, bio, profile_pic FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0], "username": row[1], "email": row[2],
            "full_name": row[3], "bio": row[4], "profile_pic": row[5]
        }
    return None

def update_user_profile(user_id, full_name, bio, profile_pic_path=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if profile_pic_path:
        cursor.execute("UPDATE users SET full_name = ?, bio = ?, profile_pic = ? WHERE id = ?",
                       (full_name, bio, profile_pic_path, user_id))
    else:
        cursor.execute("UPDATE users SET full_name = ?, bio = ? WHERE id = ?",
                       (full_name, bio, user_id))
    conn.commit()
    conn.close()

# ---------- Contact messages ----------
def save_contact_message(name, email, subject, message):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO contact_messages (name, email, subject, message) VALUES (?, ?, ?, ?)",
        (name, email, subject, message)
    )
    conn.commit()
    conn.close()

# Initialize database (runs on import)
init_db()