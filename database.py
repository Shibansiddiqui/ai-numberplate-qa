import os
import sqlite3

DB_PATH = os.getenv("PARKVISION_DB_PATH", "parkvision.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            filename TEXT NOT NULL,
            image_sha256 TEXT NOT NULL,
            duplicate_in_db INTEGER NOT NULL,
            blur_score REAL NOT NULL,
            brightness REAL NOT NULL,
            contrast REAL NOT NULL,
            quality_status TEXT NOT NULL,
            ocr_text TEXT,
            ocr_confidence REAL,
            ocr_status TEXT NOT NULL,
            operator_correction TEXT,
            notes TEXT
        )
        """)
        conn.commit()

def event_exists(image_hash):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM events WHERE image_sha256 = ? LIMIT 1", (image_hash,)
        ).fetchone()
    return row is not None

def insert_event(event):
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO events (
                created_at, filename, image_sha256, duplicate_in_db,
                blur_score, brightness, contrast, quality_status,
                ocr_text, ocr_confidence, ocr_status, operator_correction, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event["created_at"], event["filename"], event["image_sha256"],
            int(event["duplicate_in_db"]), event["blur_score"], event["brightness"],
            event["contrast"], event["quality_status"], event["ocr_text"],
            event["ocr_confidence"], event["ocr_status"],
            event["operator_correction"], event["notes"]
        ))
        conn.commit()
        return cur.lastrowid

def fetch_events(limit=200):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(row) for row in rows]

def update_correction(event_id, correction, notes):
    with get_conn() as conn:
        conn.execute("""
            UPDATE events SET operator_correction = ?, notes = ?
            WHERE id = ?
        """, (correction.strip(), notes.strip(), int(event_id)))
        conn.commit()

# Initialize database on module import
init_db()