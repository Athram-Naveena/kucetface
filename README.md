# Student Attendance & Risk Analysis System (Multi-Face Recognition + WhatsApp Alerts)

A cloud-ready Flask web application using OpenCV YuNet & SFace ONNX models for multi-face classroom attendance, section-wise tracking, role-based Teacher/Parent logins, open Student self-service lookup, and automated Twilio WhatsApp notifications to parents upon finalizing attendance.

## Features
- **Role-Based Authentication**:
  - **Teacher Portal** (`/teacher/login`): Dedicated login & staff signup (Default demo: `teacher` / `teacher123`, Staff Code: `KUCET2026`).
  - **Parent Portal** (`/parent/login`): Dedicated login & self-activation linked to their ward's Student ID and registered mobile.
  - **Student Portal** (`/student`): Public access—no login required.
- **Class Sections**: Filter students, sessions, attendance rosters, and Excel reports by section (`A`, `B`, `C`, `D`).
- **Multi-Face Classroom Scan**: Detects and marks multiple students simultaneously in a single camera snapshot with live green/amber bounding-box overlays.
- **Student Management & Manual Overrides**: Add or remove students directly from the Teacher Dashboard and manually toggle Present/Absent status before finalizing.
- **Finalize & Auto WhatsApp Alerts**: Clicking **Finalize Attendance & Send WhatsApp Alerts** locks the session, marks remaining section students as Absent, and sends automated WhatsApp notifications via Twilio to parents.

## Local Setup
```bash
pip install -r requirements.txt
python download_models.py
python app.py
```

## Render Deployment
1. Push this repository to GitHub.
2. Create a new **Web Service** on Render connected to your GitHub repo.
3. **Build Command**: `pip install -r requirements.txt && python download_models.py`
4. **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Add Environment Variables on Render:
   - `TWILIO_ACCOUNT_SID`
   - `TWILIO_AUTH_TOKEN`
   - `TWILIO_WHATSAPP_FROM` (e.g. `whatsapp:+14155238886`)
   - `TEACHER_SIGNUP_CODE` (default `KUCET2026`)
"# kucetface" 
"# kucetface" 
