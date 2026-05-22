from app.DATABASE.DB_CREATE import DB_NAME
import sqlite3
import os
def get_connection(conn=None):
    if conn is None:
        conn = sqlite3.connect(DB_NAME)
        close_after = True
    else:
        close_after = False
    return conn, close_after
def add_image(image_path, class_label, file_hash, conn=None):
    conn, close_after = get_connection(conn)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO image_data (image_path, class_label, file_hash) VALUES (?, ?, ?)",
            (str(image_path), class_label, file_hash)
        )
        if close_after:
            conn.commit() 
    finally:
        if close_after:
            conn.close()

def update_used(image_path, used, conn=None):
    conn, close_after = get_connection(conn)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE image_data SET is_used = ? WHERE image_path = ?",
            (used, str(image_path))
        )
        if close_after:
            conn.commit()
    finally:
        if close_after:
            conn.close()

def check_duplicate(file_hash, conn=None):
    conn, close_after = get_connection(conn)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM image_data WHERE file_hash = ?", (str(file_hash),))
        row = cursor.fetchone()
        return row is not None
    finally:
        if close_after:
            conn.close()
            
def save_files_to_db_and_folder_transactioned(to_save):
    with sqlite3.connect(DB_NAME) as connection:
        connection.execute("BEGIN")
        saved = 0
        class_counts = {}
        try:
            for file_data, file_path_str, class_name, file_hash in to_save:
                
                os.makedirs(os.path.dirname(file_path_str), exist_ok=True)
                with open(file_path_str, 'wb') as f:
                    f.write(file_data)

                add_image(file_path_str, class_name, file_hash, conn=connection)
                saved += 1
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            connection.commit()
            return saved, class_counts

        except Exception as e:
            connection.rollback()
            for _, file_path_str, _, _ in to_save:
                if os.path.exists(file_path_str):
                    try:
                        os.remove(file_path_str)
                    except OSError:
                        pass
            return {'error': f'Ошибка при сохранении: {str(e)}'}, 500
        
def is_file_used(path,conn = None):
    conn, close_after = get_connection(conn)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM image_data WHERE is_used = 0 AND image_path = ?",(str(path),))
        used = cursor.fetchone()
        return used is None
    finally:
        if close_after:
            conn.close()
            
def add_model(model_path,val_accuracy,test_acc, conn = None):
    conn, close_after = get_connection(conn)
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO model_data (model_path, val_accuracy, test_accuracy) VALUES (?, ?, ?)", (str(model_path), val_accuracy, test_acc))
        if close_after:
            conn.commit()
    finally:
        if close_after:
            conn.close()
            
def get_newest_model_path(conn = None):
    conn, close_after = get_connection(conn)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT model_path FROM model_data ORDER BY added_at DESC LIMIT 1")
        newest_model_path = cursor.fetchone()
        return newest_model_path[0] if newest_model_path else None
    finally:
        if close_after:
            conn.close()
            
def get_unused_count(conn = None):
    conn, close_after = get_connection(conn)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM image_data WHERE is_used = 0")
        unused_count = cursor.fetchone()[0]
        return unused_count
    finally:
        if close_after:
            conn.close()

def get_model_date(model_path,conn = None):
    conn, close_after = get_connection(conn)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT added_at FROM model_data WHERE model_path = ?",(str(model_path),))
        newest_model_date = cursor.fetchone()
        return newest_model_date[0] if newest_model_date else None
    finally:
        if close_after:
            conn.close()