import os
import sqlite3
import base64
import io
from datetime import datetime, date
from functools import wraps

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, session
)
from werkzeug.security import generate_password_hash, check_password_hash
from openpyxl import Workbook

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    TwilioClient = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "attendance.db"))
YUNET_MODEL = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(BASE_DIR, "models", "face_recognition_sface_2021dec.onnx")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "kucet-attendance-secret-key-2026")


def db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('teacher', 'parent')),
            full_name TEXT NOT NULL,
            linked_student_id TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            section TEXT NOT NULL DEFAULT 'A',
            parent_name TEXT DEFAULT '',
            parent_mobile TEXT DEFAULT '',
            face_image BLOB
        );

        CREATE TABLE IF NOT EXISTS class_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            department TEXT NOT NULL DEFAULT 'ECE',
            section TEXT NOT NULL DEFAULT 'A',
            teacher_name TEXT DEFAULT '',
            session_date TEXT NOT NULL,
            session_time TEXT NOT NULL,
            is_finalized INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS attendance_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            student_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Present',
            marked_at TEXT NOT NULL,
            UNIQUE(session_id, student_id),
            FOREIGN KEY(session_id) REFERENCES class_sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY(student_id) REFERENCES students(student_id) ON DELETE CASCADE
        );
        """)

        # Lightweight schema migration if upgrading an existing attendance.db
        student_cols = [r["name"] for r in conn.execute("PRAGMA table_info(students)").fetchall()]
        if "section" not in student_cols:
            conn.execute("ALTER TABLE students ADD COLUMN section TEXT NOT NULL DEFAULT 'A'")

        session_cols = [r["name"] for r in conn.execute("PRAGMA table_info(class_sessions)").fetchall()]
        if "department" not in session_cols:
            conn.execute("ALTER TABLE class_sessions ADD COLUMN department TEXT NOT NULL DEFAULT 'ECE'")
        if "section" not in session_cols:
            conn.execute("ALTER TABLE class_sessions ADD COLUMN section TEXT NOT NULL DEFAULT 'A'")
        if "teacher_name" not in session_cols:
            conn.execute("ALTER TABLE class_sessions ADD COLUMN teacher_name TEXT DEFAULT ''")
        if "is_finalized" not in session_cols:
            conn.execute("ALTER TABLE class_sessions ADD COLUMN is_finalized INTEGER NOT NULL DEFAULT 0")

        # Seed default teacher account if none exists
        teacher_exists = conn.execute("SELECT 1 FROM users WHERE username='teacher'").fetchone()
        if not teacher_exists:
            conn.execute(
                "INSERT INTO users (username, password_hash, role, full_name) VALUES (?,?,?,?)",
                ("teacher", generate_password_hash("teacher123"), "teacher", "Subject Teacher")
            )


def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in to access this section.", "warning")
                if role == "parent":
                    return redirect(url_for("parent_login"))
                return redirect(url_for("teacher_login"))
            if role and session.get("role") != role:
                flash(f"Access restricted to {role} accounts.", "error")
                return redirect(url_for("home"))
            return f(*args, **kwargs)
        return wrapped
    return decorator


def load_models():
    if not (os.path.exists(YUNET_MODEL) and os.path.exists(SFACE_MODEL)):
        return None, None
    try:
        detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL, "", (320, 320), 0.75, 0.3, 5000
        )
        recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
        return detector, recognizer
    except Exception as exc:
        app.logger.exception("Could not load face models: %s", exc)
        return None, None


detector, recognizer = load_models()


def decode_data_url(data_url):
    if not data_url or "," not in data_url:
        raise ValueError("No camera image was received.")
    raw = base64.b64decode(data_url.split(",", 1)[1])
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not read the camera image.")
    return frame


def extract_all_face_features(frame):
    """Detects ALL faces in a frame and returns a list of (bbox, feature_vector)."""
    global detector, recognizer
    if detector is None or recognizer is None:
        detector, recognizer = load_models()
    if detector is None or recognizer is None:
        raise RuntimeError("Face models are missing. Run: python download_models.py")

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return []

    results = []
    for face in faces:
        aligned = recognizer.alignCrop(frame, face)
        feature = recognizer.feature(aligned)
        bbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
        results.append((bbox, feature))
    return results


def single_face_feature(frame):
    faces = extract_all_face_features(frame)
    if not faces:
        raise ValueError("No face detected. Face the camera clearly and try again.")
    largest = max(faces, key=lambda item: item[0][2] * item[0][3])
    return largest[1]


def compare_features(feature_a, feature_b):
    return float(recognizer.match(feature_a, feature_b, cv2.FaceRecognizerSF_FR_COSINE))


def format_whatsapp_number(mobile):
    digits = "".join(ch for ch in str(mobile) if ch.isdigit())
    if not digits:
        return None
    if len(digits) == 10:
        digits = "91" + digits  # Default India country code (+91)
    return f"whatsapp:+{digits}"


def send_whatsapp_absent_alert(parent_mobile, parent_name, student_name, student_id, subject, section, session_date):
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_whatsapp = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

    to_whatsapp = format_whatsapp_number(parent_mobile)
    if not to_whatsapp:
        return False, "Missing/invalid parent mobile"

    body = (
        f"*Attendance Alert - Absent Notice*\n"
        f"Dear {parent_name or 'Parent'},\n"
        f"Your ward *{student_name}* (ID: {student_id}, Section: {section}) "
        f"was marked *ABSENT* for *{subject}* on {session_date}.\n"
        f"- Department Attendance System"
    )

    if not (TwilioClient and sid and token):
        app.logger.warning("[WHATSAPP DRY-RUN] To: %s | Msg: %s", to_whatsapp, body)
        return False, "Twilio credentials not configured (logged locally)"

    try:
        client = TwilioClient(sid, token)
        client.messages.create(from_=from_whatsapp, to=to_whatsapp, body=body)
        return True, "Sent"
    except Exception as exc:
        app.logger.error("Twilio WhatsApp error for %s: %s", to_whatsapp, exc)
        return False, str(exc)


# -------------------- PUBLIC ROUTES --------------------

@app.route("/")
def home():
    with db() as conn:
        student_count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        session_count = conn.execute("SELECT COUNT(*) FROM class_sessions").fetchone()[0]
        today_count = conn.execute(
            "SELECT COUNT(*) FROM attendance_records ar JOIN class_sessions cs "
            "ON ar.session_id=cs.session_id WHERE cs.session_date=? AND ar.status='Present'",
            (date.today().isoformat(),)
        ).fetchone()[0]
    return render_template(
        "home.html",
        student_count=student_count,
        session_count=session_count,
        today_count=today_count,
        models_ready=detector is not None and recognizer is not None
    )


@app.route("/student")
def student():
    """Public Student Portal: No login credentials required."""
    sid = request.args.get("student_id", "").strip()
    section_filter = request.args.get("section", "").strip()
    with db() as conn:
        if section_filter:
            students = conn.execute(
                "SELECT student_id, name, section FROM students WHERE section=? ORDER BY name",
                (section_filter,)
            ).fetchall()
        else:
            students = conn.execute(
                "SELECT student_id, name, section FROM students ORDER BY section, name"
            ).fetchall()

        selected = None
        summary = None
        subjects = []
        if sid:
            selected = conn.execute("SELECT * FROM students WHERE student_id=?", (sid,)).fetchone()
            if selected:
                total = conn.execute(
                    "SELECT COUNT(*) FROM class_sessions WHERE section=?",
                    (selected["section"],)
                ).fetchone()[0]
                present = conn.execute(
                    """SELECT COUNT(*) FROM attendance_records ar
                       JOIN class_sessions cs ON ar.session_id = cs.session_id
                       WHERE ar.student_id=? AND cs.section=? AND ar.status='Present'""",
                    (sid, selected["section"])
                ).fetchone()[0]
                percent = (present / total * 100) if total else 0
                summary = {
                    "total": total,
                    "present": present,
                    "absent": max(total - present, 0),
                    "percentage": round(percent, 1),
                    "risk": total > 0 and percent < 75
                }
                subjects = conn.execute("""
                    SELECT cs.subject, COUNT(*) AS total,
                    SUM(CASE WHEN ar.status='Present' THEN 1 ELSE 0 END) AS present
                    FROM class_sessions cs
                    LEFT JOIN attendance_records ar
                      ON ar.session_id=cs.session_id AND ar.student_id=?
                    WHERE cs.section=?
                    GROUP BY cs.subject ORDER BY cs.subject
                """, (sid, selected["section"])).fetchall()

    return render_template(
        "student.html",
        students=students,
        selected=selected,
        summary=summary,
        subjects=subjects,
        section_filter=section_filter
    )


# -------------------- AUTHENTICATION ROUTES --------------------

@app.route("/teacher/login", methods=["GET", "POST"])
def teacher_login():
    if session.get("role") == "teacher":
        return redirect(url_for("teacher"))
    if request.method == "POST":
        action = request.form.get("action", "login")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        with db() as conn:
            if action == "register":
                full_name = request.form.get("full_name", "").strip() or username
                access_code = request.form.get("access_code", "").strip()
                expected_code = os.environ.get("TEACHER_SIGNUP_CODE", "KUCET2026")
                if access_code != expected_code:
                    flash("Invalid Staff Access Code.", "error")
                    return redirect(url_for("teacher_login"))
                if not username or not password:
                    flash("Username and password are required.", "error")
                    return redirect(url_for("teacher_login"))
                try:
                    conn.execute(
                        "INSERT INTO users (username, password_hash, role, full_name) VALUES (?,?,?,?)",
                        (username, generate_password_hash(password), "teacher", full_name)
                    )
                    flash("Teacher account created. Please sign in.", "success")
                except sqlite3.IntegrityError:
                    flash("Username already exists.", "error")
                return redirect(url_for("teacher_login"))

            user = conn.execute(
                "SELECT * FROM users WHERE username=? AND role='teacher'", (username,)
            ).fetchone()
            if user and check_password_hash(user["password_hash"], password):
                session.clear()
                session["user_id"] = user["user_id"]
                session["role"] = "teacher"
                session["full_name"] = user["full_name"]
                flash(f"Welcome, {user['full_name']}!", "success")
                return redirect(url_for("teacher"))
            flash("Invalid teacher credentials.", "error")
    return render_template("teacher_login.html")


@app.route("/parent/login", methods=["GET", "POST"])
def parent_login():
    if session.get("role") == "parent":
        return redirect(url_for("parent"))
    if request.method == "POST":
        action = request.form.get("action", "login")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        with db() as conn:
            if action == "register":
                student_id = request.form.get("student_id", "").strip()
                parent_mobile = request.form.get("parent_mobile", "").strip()
                student_row = conn.execute(
                    "SELECT * FROM students WHERE student_id=?", (student_id,)
                ).fetchone()
                if not student_row:
                    flash("Student ID not found. Ask the teacher to register the student first.", "error")
                    return redirect(url_for("parent_login"))
                if student_row["parent_mobile"] and parent_mobile not in student_row["parent_mobile"]:
                    flash("Parent mobile does not match registered student record.", "error")
                    return redirect(url_for("parent_login"))
                if not username or not password:
                    flash("Username and password are required.", "error")
                    return redirect(url_for("parent_login"))
                try:
                    conn.execute(
                        """INSERT INTO users (username, password_hash, role, full_name, linked_student_id)
                           VALUES (?,?,?,?,?)""",
                        (username, generate_password_hash(password), "parent",
                         student_row["parent_name"] or f"Parent of {student_row['name']}", student_id)
                    )
                    flash("Parent account activated! You can now log in.", "success")
                except sqlite3.IntegrityError:
                    flash("Username already taken. Choose another.", "error")
                return redirect(url_for("parent_login"))

            user = conn.execute(
                "SELECT * FROM users WHERE username=? AND role='parent'", (username,)
            ).fetchone()
            if user and check_password_hash(user["password_hash"], password):
                session.clear()
                session["user_id"] = user["user_id"]
                session["role"] = "parent"
                session["full_name"] = user["full_name"]
                session["linked_student_id"] = user["linked_student_id"]
                flash(f"Welcome, {user['full_name']}!", "success")
                return redirect(url_for("parent"))
            flash("Invalid parent username or password.", "error")
    return render_template("parent_login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("home"))


# -------------------- PARENT PORTAL (PROTECTED) --------------------

@app.route("/parent")
@login_required(role="parent")
def parent():
    sid = session.get("linked_student_id", "")
    with db() as conn:
        selected = conn.execute("SELECT * FROM students WHERE student_id=?", (sid,)).fetchone()
        if not selected:
            flash("Linked student record not found.", "error")
            return redirect(url_for("logout"))

        total = conn.execute(
            "SELECT COUNT(*) FROM class_sessions WHERE section=?", (selected["section"],)
        ).fetchone()[0]
        present = conn.execute(
            """SELECT COUNT(*) FROM attendance_records ar
               JOIN class_sessions cs ON ar.session_id = cs.session_id
               WHERE ar.student_id=? AND cs.section=? AND ar.status='Present'""",
            (sid, selected["section"])
        ).fetchone()[0]
        percent = (present / total * 100) if total else 0
        summary = {
            "total": total,
            "present": present,
            "absent": max(total - present, 0),
            "percentage": round(percent, 1),
            "risk": total > 0 and percent < 75
        }
        subjects = conn.execute("""
            SELECT cs.subject, COUNT(*) AS total,
            SUM(CASE WHEN ar.status='Present' THEN 1 ELSE 0 END) AS present
            FROM class_sessions cs
            LEFT JOIN attendance_records ar ON ar.session_id=cs.session_id AND ar.student_id=?
            WHERE cs.section=?
            GROUP BY cs.subject ORDER BY cs.subject
        """, (sid, selected["section"])).fetchall()

        recent_logs = conn.execute("""
            SELECT cs.session_date, cs.session_time, cs.subject,
                   COALESCE(ar.status, 'Absent') AS status
            FROM class_sessions cs
            LEFT JOIN attendance_records ar ON ar.session_id=cs.session_id AND ar.student_id=?
            WHERE cs.section=?
            ORDER BY cs.session_id DESC LIMIT 15
        """, (sid, selected["section"])).fetchall()

    return render_template(
        "parent.html",
        selected=selected,
        summary=summary,
        subjects=subjects,
        recent_logs=recent_logs
    )


# -------------------- TEACHER PORTAL (PROTECTED) --------------------

@app.route("/teacher", methods=["GET", "POST"])
@login_required(role="teacher")
def teacher():
    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        department = request.form.get("department", "ECE").strip().upper()
        section = request.form.get("section", "A").strip().upper()
        if subject:
            now = datetime.now()
            with db() as conn:
                conn.execute(
                    """INSERT INTO class_sessions
                       (subject, department, section, teacher_name, session_date, session_time)
                       VALUES (?,?,?,?,?,?)""",
                    (subject, department, section, session.get("full_name", ""),
                     now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"))
                )
            flash(f"Class session created for {department} - Section {section}.", "success")
        return redirect(url_for("teacher"))

    section_filter = request.args.get("section", "").strip().upper()
    with db() as conn:
        if section_filter:
            sessions = conn.execute(
                "SELECT * FROM class_sessions WHERE section=? ORDER BY session_id DESC",
                (section_filter,)
            ).fetchall()
            students = conn.execute(
                "SELECT student_id, name, department, section, parent_name, parent_mobile "
                "FROM students WHERE section=? ORDER BY student_id",
                (section_filter,)
            ).fetchall()
        else:
            sessions = conn.execute(
                "SELECT * FROM class_sessions ORDER BY session_id DESC"
            ).fetchall()
            students = conn.execute(
                "SELECT student_id, name, department, section, parent_name, parent_mobile "
                "FROM students ORDER BY section, student_id"
            ).fetchall()

        sections = [r["section"] for r in conn.execute(
            "SELECT DISTINCT section FROM students ORDER BY section"
        ).fetchall()]

    return render_template(
        "teacher.html",
        sessions=sessions,
        students=students,
        sections=sections,
        section_filter=section_filter
    )


@app.route("/register", methods=["GET", "POST"])
@login_required(role="teacher")
def register():
    """Add or update a student (Teacher restricted)."""
    if request.method == "POST":
        sid = request.form.get("student_id", "").strip()
        name = request.form.get("name", "").strip()
        department = request.form.get("department", "").strip().upper()
        section = request.form.get("section", "A").strip().upper()
        parent_name = request.form.get("parent_name", "").strip()
        parent_mobile = request.form.get("parent_mobile", "").strip()
        parent_username = request.form.get("parent_username", "").strip()
        parent_password = request.form.get("parent_password", "").strip()
        image_data = request.form.get("face_image", "")

        if not all([sid, name, department, section, image_data]):
            flash("Please complete student ID, name, department, section, and capture a face image.", "error")
            return redirect(url_for("register"))
        try:
            frame = decode_data_url(image_data)
            single_face_feature(frame)  # Validate face exists
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError("Could not save the captured image.")
            with db() as conn:
                conn.execute(
                    """INSERT INTO students
                    (student_id, name, department, section, parent_name, parent_mobile, face_image)
                    VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(student_id) DO UPDATE SET
                    name=excluded.name, department=excluded.department, section=excluded.section,
                    parent_name=excluded.parent_name, parent_mobile=excluded.parent_mobile,
                    face_image=excluded.face_image""",
                    (sid, name, department, section, parent_name, parent_mobile, encoded.tobytes())
                )
                if parent_username and parent_password:
                    conn.execute(
                        """INSERT INTO users (username, password_hash, role, full_name, linked_student_id)
                           VALUES (?,?,?,?,?)
                           ON CONFLICT(username) DO UPDATE SET
                           password_hash=excluded.password_hash, linked_student_id=excluded.linked_student_id""",
                        (parent_username, generate_password_hash(parent_password),
                         "parent", parent_name or f"Parent of {name}", sid)
                    )
            flash(f"Student {name} ({sid}) added to Section {section}.", "success")
            return redirect(url_for("teacher"))
        except Exception as exc:
            flash(str(exc), "error")
            return redirect(url_for("register"))
    return render_template("register.html")


@app.route("/teacher/student/delete/<student_id>", methods=["POST"])
@login_required(role="teacher")
def delete_student(student_id):
    with db() as conn:
        conn.execute("DELETE FROM attendance_records WHERE student_id=?", (student_id,))
        conn.execute("DELETE FROM users WHERE role='parent' AND linked_student_id=?", (student_id,))
        conn.execute("DELETE FROM students WHERE student_id=?", (student_id,))
    flash(f"Student {student_id} and associated records removed.", "success")
    return redirect(url_for("teacher"))


@app.route("/start/<int:session_id>")
@login_required(role="teacher")
def start_session(session_id):
    with db() as conn:
        class_session = conn.execute(
            "SELECT * FROM class_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if class_session is None:
            flash("Class session not found.", "error")
            return redirect(url_for("teacher"))

        roster = conn.execute("""
            SELECT s.student_id, s.name, s.section,
                   COALESCE(ar.status, 'Absent') AS status, ar.marked_at
            FROM students s
            LEFT JOIN attendance_records ar
              ON s.student_id = ar.student_id AND ar.session_id = ?
            WHERE s.section = ?
            ORDER BY s.student_id
        """, (session_id, class_session["section"])).fetchall()

    return render_template("recognize.html", session=class_session, roster=roster)


@app.route("/api/recognize/<int:session_id>", methods=["POST"])
@login_required(role="teacher")
def api_recognize(session_id):
    """Multi-face detection & recognition in a single frame for the target section."""
    payload = request.get_json(silent=True) or {}
    try:
        frame = decode_data_url(payload.get("image", ""))
        detected_faces = extract_all_face_features(frame)
        if not detected_faces:
            return jsonify(ok=False, message="No faces detected in frame.", matches=[], boxes=[])

        with db() as conn:
            class_session = conn.execute(
                "SELECT * FROM class_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if not class_session:
                return jsonify(ok=False, message="Class session not found."), 404
            if class_session["is_finalized"]:
                return jsonify(ok=False, message="This session is already finalized."), 400

            students = conn.execute(
                "SELECT student_id, name, face_image FROM students WHERE section=? AND face_image IS NOT NULL",
                (class_session["section"],)
            ).fetchall()

            if not students:
                return jsonify(
                    ok=False,
                    message=f"No registered students with face data in Section {class_session['section']}."
                )

            gallery = []
            for st in students:
                image_arr = np.frombuffer(st["face_image"], dtype=np.uint8)
                saved = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
                if saved is None:
                    continue
                try:
                    saved_feat = single_face_feature(saved)
                    gallery.append((st["student_id"], st["name"], saved_feat))
                except Exception:
                    continue

            matched_students = []
            boxes = []
            already_matched_ids = set()

            for bbox, query_feat in detected_faces:
                best_id = None
                best_name = "Unknown"
                best_score = -1.0

                for st_id, st_name, gal_feat in gallery:
                    score = compare_features(query_feat, gal_feat)
                    if score > best_score:
                        best_score = score
                        best_id = st_id
                        best_name = st_name

                if best_id and best_score >= 0.363 and best_id not in already_matched_ids:
                    already_matched_ids.add(best_id)
                    conn.execute(
                        """INSERT INTO attendance_records (session_id, student_id, status, marked_at)
                           VALUES (?,?,?,?)
                           ON CONFLICT(session_id, student_id) DO UPDATE SET
                           status='Present', marked_at=excluded.marked_at""",
                        (session_id, best_id, "Present", datetime.now().isoformat(timespec="seconds"))
                    )
                    matched_students.append({
                        "student_id": best_id,
                        "name": best_name,
                        "score": round(best_score, 3)
                    })
                    boxes.append({"bbox": bbox, "label": f"{best_name} ({best_id})", "matched": True})
                else:
                    boxes.append({"bbox": bbox, "label": "Unknown", "matched": False})

        if not matched_students:
            return jsonify(
                ok=False,
                message=f"Detected {len(detected_faces)} face(s), but none matched Section {class_session['section']}.",
                matches=[],
                boxes=boxes
            )

        names_str = ", ".join(f"{m['name']} ({m['student_id']})" for m in matched_students)
        return jsonify(
            ok=True,
            message=f"Marked Present ({len(matched_students)}/{len(detected_faces)} faces): {names_str}",
            matches=matched_students,
            boxes=boxes
        )
    except Exception as exc:
        return jsonify(ok=False, message=str(exc)), 400


@app.route("/session/<int:session_id>/toggle/<student_id>", methods=["POST"])
@login_required(role="teacher")
def toggle_attendance(session_id, student_id):
    """Allows teacher to manually toggle Present/Absent before finalizing."""
    with db() as conn:
        class_session = conn.execute(
            "SELECT * FROM class_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not class_session or class_session["is_finalized"]:
            flash("Cannot edit a finalized or missing session.", "error")
            return redirect(url_for("teacher"))

        existing = conn.execute(
            "SELECT status FROM attendance_records WHERE session_id=? AND student_id=?",
            (session_id, student_id)
        ).fetchone()
        new_status = "Absent" if (existing and existing["status"] == "Present") else "Present"
        conn.execute(
            """INSERT INTO attendance_records (session_id, student_id, status, marked_at)
               VALUES (?,?,?,?)
               ON CONFLICT(session_id, student_id) DO UPDATE SET
               status=excluded.status, marked_at=excluded.marked_at""",
            (session_id, student_id, new_status, datetime.now().isoformat(timespec="seconds"))
        )
    return redirect(url_for("start_session", session_id=session_id))


@app.route("/session/<int:session_id>/finalize", methods=["POST"])
@login_required(role="teacher")
def finalize_session(session_id):
    """Finalizes attendance, marks unmarked section students Absent, and triggers WhatsApp alerts."""
    with db() as conn:
        class_session = conn.execute(
            "SELECT * FROM class_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not class_session:
            flash("Class session not found.", "error")
            return redirect(url_for("teacher"))
        if class_session["is_finalized"]:
            flash("This session has already been finalized.", "warning")
            return redirect(url_for("start_session", session_id=session_id))

        section_students = conn.execute(
            "SELECT * FROM students WHERE section=?", (class_session["section"],)
        ).fetchall()

        now_iso = datetime.now().isoformat(timespec="seconds")
        absentees = []

        for st in section_students:
            rec = conn.execute(
                "SELECT status FROM attendance_records WHERE session_id=? AND student_id=?",
                (session_id, st["student_id"])
            ).fetchone()
            if not rec:
                conn.execute(
                    "INSERT INTO attendance_records (session_id, student_id, status, marked_at) VALUES (?,?,?,?)",
                    (session_id, st["student_id"], "Absent", now_iso)
                )
                absentees.append(st)
            elif rec["status"] == "Absent":
                absentees.append(st)

        conn.execute("UPDATE class_sessions SET is_finalized=1 WHERE session_id=?", (session_id,))

    sent_count = 0
    logged_count = 0
    for st in absentees:
        ok, _ = send_whatsapp_absent_alert(
            parent_mobile=st["parent_mobile"],
            parent_name=st["parent_name"],
            student_name=st["name"],
            student_id=st["student_id"],
            subject=class_session["subject"],
            section=class_session["section"],
            session_date=class_session["session_date"]
        )
        if ok:
            sent_count += 1
        else:
            logged_count += 1

    flash(
        f"Attendance finalized! Total Absentees: {len(absentees)}. "
        f"WhatsApp alerts sent: {sent_count} (Queued/Dry-run: {logged_count}).",
        "success"
    )
    return redirect(url_for("start_session", session_id=session_id))


@app.route("/export/<int:session_id>")
@login_required(role="teacher")
def export_session(session_id):
    with db() as conn:
        class_session = conn.execute(
            "SELECT * FROM class_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not class_session:
            flash("Session not found.", "error")
            return redirect(url_for("teacher"))
        rows = conn.execute("""
            SELECT s.student_id, s.name, s.department, s.section,
                   COALESCE(ar.status, 'Absent') AS status, ar.marked_at
            FROM students s
            LEFT JOIN attendance_records ar
              ON s.student_id=ar.student_id AND ar.session_id=?
            WHERE s.section=?
            ORDER BY s.student_id
        """, (session_id, class_session["section"])).fetchall()

    wb = Workbook()
    ws = wb.active
    ws.title = f"Sec_{class_session['section']}_Attendance"
    ws.append(["Student ID", "Name", "Department", "Section", "Status", "Marked At"])
    for row in rows:
        ws.append([
            row["student_id"], row["name"], row["department"],
            row["section"], row["status"], row["marked_at"] or ""
        ])
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    safe_subject = "".join(c if c.isalnum() else "_" for c in class_session["subject"])
    return send_file(
        output,
        as_attachment=True,
        download_name=f"{safe_subject}_Sec{class_session['section']}_{class_session['session_date']}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.route("/health")
def health():
    return jsonify(status="ok")


init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
