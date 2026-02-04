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
    c.execute('''CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    student_roll_id TEXT UNIQUE,
                    class_id INTEGER,
                    FOREIGN KEY (class_id) REFERENCES classes(id)
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
def add_student(name, roll_id, class_id):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO students (name, student_roll_id, class_id) VALUES (?, ?, ?)", 
                  (name, roll_id, class_id))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_students_by_class(class_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, student_roll_id FROM students WHERE class_id=?", (class_id,))
    data = c.fetchall()
    conn.close()
    return data

def get_student_by_roll(roll_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, class_id FROM students WHERE student_roll_id=?", (roll_id,))
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
    c.execute('''SELECT r.student_id, s.name, s.student_roll_id, r.score 
                 FROM results r 
                 JOIN students s ON r.student_id = s.id 
                 WHERE r.exam_id=?''', (exam_id,))
    data = c.fetchall()
    conn.close()
    return data

# Initialize on import
if __name__ == "__main__":
    init_db()
