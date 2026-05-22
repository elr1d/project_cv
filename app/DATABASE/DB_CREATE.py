import sqlite3
import os
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
DB_NAME = os.path.join(_PROJECT_ROOT, 'CV_project.db')

def init_db():
    with sqlite3.connect(DB_NAME) as connection:
        cursor = connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT UNIQUE NOT NULL,
        is_used BOOLEAN DEFAULT FALSE,
        class_label TEXT NOT NULL,
        added_at DATE DEFAULT (datetime('now', 'localtime')),
        file_hash TEXT UNIQUE NOT NULL
    )
                    """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_path TEXT UNIQUE NOT NULL,
        added_at DATE DEFAULT (datetime('now', 'localtime')),
        val_accuracy FLOAT NOT NULL,
        test_accuracy FLOAT NOT NULL
    )
                    """)
