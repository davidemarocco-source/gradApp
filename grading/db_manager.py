import sqlite3
import json
import os

DB_FILE = "omr_database.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Classes Table
    c.execute('''CREATE TABLE IF NOT EXISTS classes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE
                )''')
    
    # Students Table
    # educational_id: The official school ID (e.g. D33000123)
    # omr_id: A short simplified ID (e.g. 1, 2, 3) unique PRO CLASS for bubbling
    c.execute('''CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    educational_id TEXT, 
                    omr_id INTEGER,
                    class_id INTEGER,
                    FOREIGN KEY (class_id) REFERENCES classes(id),
                    UNIQUE(class_id, omr_id), -- OMR ID must be unique within the class
                    UNIQUE(class_id, educational_id) -- Edu ID should be unique within class
                )''')
    
    # Exams Table
    c.execute('''CREATE TABLE IF NOT EXISTS exams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    class_id INTEGER,
                    date TEXT,
                    answer_key TEXT, -- JSON string of correct answers
                    FOREIGN KEY (class_id) REFERENCES classes(id)
                )''')
    
    # Results Table
    c.execute('''CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exam_id INTEGER,
                    student_id INTEGER,
                    score REAL,
                    answers TEXT, -- JSON string of student's answers
                    image_path TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (exam_id) REFERENCES exams(id),
                    FOREIGN KEY (student_id) REFERENCES students(id)
                )''')
    
    conn.commit()
    conn.close()

def get_connection():
    return sqlite3.connect(DB_FILE)

# --- Classes ---
def add_class(name):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO classes (name) VALUES (?)", (name,))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_all_classes():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name FROM classes")
    data = c.fetchall()
    conn.close()
    return data

def get_class_name(class_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT name FROM classes WHERE id=?", (class_id,))
    data = c.fetchone()
    conn.close()
    return data[0] if data else None

# --- Students ---
def add_student(name, educational_id, class_id):
    """
    Adds a student and auto-assigns the next available OMR ID for that class.
    """
    conn = get_connection()
    c = conn.cursor()
    try:
        # specific logic to find next omr_id
        c.execute("SELECT MAX(omr_id) FROM students WHERE class_id=?", (class_id,))
        max_id = c.fetchone()[0]
        next_omr_id = 1 if max_id is None else max_id + 1
        
        c.execute("INSERT INTO students (name, educational_id, omr_id, class_id) VALUES (?, ?, ?, ?)", 
                  (name, educational_id, next_omr_id, class_id))
        conn.commit()
        return next_omr_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_students_by_class(class_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, educational_id, omr_id FROM students WHERE class_id=?", (class_id,))
    data = c.fetchall()
    conn.close()
    return data

def get_student_by_omr(class_id, omr_id):
    """
    Find student by their short OMR ID within a specific class.
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, educational_id, omr_id FROM students WHERE class_id=? AND omr_id=?", (class_id, omr_id))
    data = c.fetchone()
    conn.close()
    return data

# --- Exams ---
def create_exam(name, class_id, date, answer_key):
    """
    answer_key: dict {question_idx: answer_idx}
    """
    conn = get_connection()
    c = conn.cursor()
    key_json = json.dumps(answer_key)
    c.execute("INSERT INTO exams (name, class_id, date, answer_key) VALUES (?, ?, ?, ?)",
              (name, class_id, date, key_json))
    exam_id = c.lastrowid
    conn.commit()
    conn.close()
    return exam_id

def get_exams_by_class(class_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, date FROM exams WHERE class_id=?", (class_id,))
    data = c.fetchall()
    conn.close()
    return data

def get_exam_details(exam_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, class_id, date, answer_key FROM exams WHERE id=?", (exam_id,))
    data = c.fetchone()
    conn.close()
    return data

# --- Results ---
def save_result(exam_id, student_id, score, answers, image_path):
    conn = get_connection()
    c = conn.cursor()
    answers_json = json.dumps(answers)
    c.execute("INSERT INTO results (exam_id, student_id, score, answers, image_path) VALUES (?, ?, ?, ?, ?)",
              (exam_id, student_id, score, answers_json, image_path))
    conn.commit()
    conn.close()

def get_results_by_exam(exam_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''SELECT r.id, r.student_id, s.name, s.educational_id, s.omr_id, r.score 
                 FROM results r 
                 JOIN students s ON r.student_id = s.id 
                 WHERE r.exam_id=?''', (exam_id,))
    data = c.fetchall()
    conn.close()
    return data

def delete_class(class_id):
    conn = get_connection()
    c = conn.cursor()
    # Manual cascade (assuming PRAGMA foreign_keys = OFF by default)
    c.execute("DELETE FROM results WHERE student_id IN (SELECT id FROM students WHERE class_id=?)", (class_id,))
    c.execute("DELETE FROM results WHERE exam_id IN (SELECT id FROM exams WHERE class_id=?)", (class_id,))
    c.execute("DELETE FROM students WHERE class_id=?", (class_id,))
    c.execute("DELETE FROM exams WHERE class_id=?", (class_id,))
    c.execute("DELETE FROM classes WHERE id=?", (class_id,))
    conn.commit()
    conn.close()

def delete_exam(exam_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM results WHERE exam_id=?", (exam_id,))
    c.execute("DELETE FROM exams WHERE id=?", (exam_id,))
    conn.commit()
    conn.close()

def delete_result(result_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM results WHERE id=?", (result_id,))
    conn.commit()
    conn.close()

# Initialize on import
if __name__ == "__main__":
    init_db()
