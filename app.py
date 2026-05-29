# ========================= IMPORTS =========================
import os
import sys
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
import cv2

from PIL import Image

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    send_file,
    flash,
    session
)

from werkzeug.utils import secure_filename
from functools import wraps

from database import (
    create_user,
    get_user_by_username,
    get_user_by_id,
    verify_password,
    save_contact_message
)


# ========================= LOCAL MODULES =========================
import database as db
from explainable_ai import get_gradcam_heatmap, save_and_display_gradcam

# ========================= FLASK APP =========================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_only_key_change_me")

# ========================= CONFIGURATION =========================
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ========================= GLOBAL MODEL VARIABLES =========================
model = None
label_map = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========================= COMPATIBILITY PATCH FOR OLDER TENSORFLOW =========================
# This fixes the "InputLayer" error when loading a model saved with TF 2.16+
class CompatibleInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, batch_shape=None, optional=None, **kwargs):
        # Remove the problematic arguments before passing to parent
        if 'batch_shape' in kwargs:
            del kwargs['batch_shape']
        if 'optional' in kwargs:
            del kwargs['optional']
        # If batch_shape was passed as positional, handle it
        if batch_shape is not None:
            kwargs['batch_shape'] = batch_shape
        super().__init__(*args, **kwargs)

# Register the custom object globally so that load_model uses it
tf.keras.utils.get_custom_objects()['InputLayer'] = CompatibleInputLayer

# ========================= DEBUG: LIST FILES AT STARTUP =========================
print("=== Starting up ===")
print(f"BASE_DIR = {BASE_DIR}")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in BASE_DIR: {os.listdir(BASE_DIR)}")

model_dir = os.path.join(BASE_DIR, "model")
if os.path.exists(model_dir):
    print(f"✅ model folder exists. Contents: {os.listdir(model_dir)}")
else:
    print(f"❌ model folder NOT found at {model_dir}")

# Also check if the model file exists directly (maybe in root)
model_in_root = os.path.join(BASE_DIR, "pneumonia_model.h5")
print(f"Model file in root? {os.path.exists(model_in_root)}")

# ========================= LOAD AI MODEL =========================
def load_dl_model():
    global model, label_map

    model_path = os.path.join(BASE_DIR, "model", "pneumonia_model.h5")
    encoder_path = os.path.join(BASE_DIR, "model", "label_encoder.json")

    print(f"🔍 Looking for model at: {model_path}")
    print(f"📁 Model file exists? {os.path.exists(model_path)}")
    print(f"📁 Label encoder exists? {os.path.exists(encoder_path)}")

    # If not found, try looking in current directory root
    if not os.path.exists(model_path):
        alt_path = os.path.join(BASE_DIR, "pneumonia_model.h5")
        print(f"Trying alternative path: {alt_path} exists? {os.path.exists(alt_path)}")
        if os.path.exists(alt_path):
            model_path = alt_path
            print(f"✅ Using alternative model path: {model_path}")

    if os.path.exists(model_path):
        try:
            print("⏳ Attempting to load model with compatibility patch...")
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model load error: {e}")
            import traceback
            traceback.print_exc()
            model = None
    else:
        print("⚠️ Model file not found – check your deployment")
        model = None

    if os.path.exists(encoder_path):
        with open(encoder_path, "r") as f:
            label_map = json.load(f)
        print("✅ Label encoder loaded")
    else:
        print("⚠️ label_encoder.json not found – using fallback mapping")
        label_map = {
            "0": "Normal",
            "1": "Bacterial Pneumonia",
            "2": "Viral Pneumonia"
        }


def get_model():
    global model
    if model is None:
        load_dl_model()
    return model


# ========================= FILE VALIDATION =========================
def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'})


# ========================= LOGIN REQUIRED DECORATOR =========================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ========================= CONTEXT PROCESSOR =========================
@app.context_processor
def inject_user():
    if 'user_id' in session:
        user = db.get_user_by_id(session['user_id'])
        return dict(current_user=user)
    return dict(current_user=None)


# ========================= AUTH ROUTES =========================
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template('signup.html')

        user_id = create_user(username, email, password)

        if user_id:
            flash("Account created successfully! Please login.", "success")
            return redirect(url_for('login'))
        else:
            flash("Username or email already exists.", "danger")

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = get_user_by_username(username)

        if user and verify_password(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.", "danger")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


# ========================= PUBLIC PAGES =========================
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/model')
def model_page():
    has_metrics = os.path.exists(os.path.join(BASE_DIR, "static/images/accuracy_history.png"))
    return render_template('model.html', has_metrics=has_metrics)


@app.route('/about')
def about_organisation():
    return render_template('about.html')


@app.route('/technology')
def technology_page():
    return render_template('technology.html')


# ========================= DASHBOARD =========================
@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    history = db.get_history(user_id=user_id, limit=10)
    stats = db.get_user_stats(user_id)
    return render_template('dashboard.html', history=history, stats=stats)


# ========================= UPLOAD PAGE =========================
@app.route('/upload')
@login_required
def upload_page():
    return render_template('upload.html')


# ========================= X-RAY VALIDATION =========================
def is_likely_chest_xray(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        img = cv2.resize(img, (512, 512))
        b, g, r = cv2.split(img)
        diff_bg = np.mean(np.abs(b.astype(np.int16) - g.astype(np.int16)))
        diff_br = np.mean(np.abs(b.astype(np.int16) - r.astype(np.int16)))
        diff_gr = np.mean(np.abs(g.astype(np.int16) - r.astype(np.int16)))
        color_score = (diff_bg + diff_br + diff_gr) / 3
        if color_score > 12:
            print("Rejected: Colorful image detected")
            return False
        green_pixels = np.sum((g > r + 40) & (g > b + 40))
        red_pixels = np.sum((r > g + 40) & (r > b + 40))
        blue_pixels = np.sum((b > r + 40) & (b > g + 40))
        total_pixels = img.shape[0] * img.shape[1]
        color_ratio = (green_pixels + red_pixels + blue_pixels) / total_pixels
        if color_ratio > 0.01:
            print("Rejected: Colored UI elements detected")
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        if mean_intensity < 15 or mean_intensity > 240:
            print("Rejected: Invalid brightness")
            return False
        variance = np.var(gray)
        if variance < 80:
            print("Rejected: Blank image")
            return False
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density > 0.25:
            print("Rejected: Too many edges")
            return False
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120,
                                minLineLength=120, maxLineGap=5)
        line_count = 0 if lines is None else len(lines)
        if line_count > 120:
            print("Rejected: Screenshot/UI")
            return False
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8)
        if len(faces) > 0:
            print("Rejected: Face detected")
            return False
        print("Valid Chest X-ray")
        return True
    except Exception as e:
        print("Validation Error:", e)
        return False


# ========================= PREDICT ROUTE =========================
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        flash("No file selected.", "danger")
        return redirect(url_for('upload_page'))

    file = request.files['file']
    if file.filename == '':
        flash("No file selected.", "danger")
        return redirect(url_for('upload_page'))

    if not allowed_file(file.filename):
        flash("Invalid file format. Only JPG, JPEG, PNG allowed.", "danger")
        return redirect(url_for('upload_page'))

    model = get_model()
    if model is None:
        flash("AI Model could not be loaded. Please check server logs.", "danger")
        return redirect(url_for('upload_page'))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_filename = secure_filename(f"temp_{timestamp}_{file.filename}")
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    file.save(temp_path)

    try:
        if not is_likely_chest_xray(temp_path):
            os.remove(temp_path)
            flash("❌ Invalid Medical Scan Detected. Please upload a valid Chest X-Ray image only.", "danger")
            return redirect(url_for('upload_page'))
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        flash(f"Validation Error: {str(e)}", "danger")
        return redirect(url_for('upload_page'))

    try:
        img = Image.open(temp_path).convert('RGB')
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array)
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        preds = model.predict(img_batch)
        pred_index = np.argmax(preds[0])
        pred_label = label_map.get(str(pred_index), "Unknown")
        confidence = float(preds[0][pred_index])

        if confidence < 0.60:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            flash("❌ Unable to confidently analyze this image. Please upload a clearer Chest X-Ray scan.", "danger")
            return redirect(url_for('upload_page'))

        gradcam_filename = f"gradcam_{timestamp}_{file.filename}"
        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
        gradcam_rel_path = None
        try:
            heatmap = get_gradcam_heatmap(img_batch, model, 'conv5_block3_out', pred_index=pred_index)
            save_and_display_gradcam(temp_path, heatmap, cam_path=gradcam_path)
            gradcam_rel_path = f"/static/uploads/{gradcam_filename}"
        except Exception as e:
            print(f"Grad-CAM Error: {e}")

        final_filename = secure_filename(f"scan_{timestamp}_{file.filename}")
        final_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
        os.rename(temp_path, final_path)
        original_rel_path = f"/static/uploads/{final_filename}"

        record_id = db.log_prediction(
            user_id=session['user_id'],
            filename=original_rel_path,
            predicted_class=pred_label,
            confidence=confidence,
            gradcam_path=gradcam_rel_path
        )

        return redirect(url_for('result_page', record_id=record_id))

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        flash(f"Error processing image: {str(e)}", "danger")
        return redirect(url_for('upload_page'))


# ========================= RESULT PAGE =========================
@app.route('/result/<int:record_id>')
@login_required
def result_page(record_id):
    record = db.get_record(record_id)
    if not record:
        flash("Record not found.", "danger")
        return redirect(url_for('dashboard'))

    record["confidence_pct"] = round(record["confidence"] * 100, 2)

    explanations = {
        "Normal": "The scan exhibits clear lungs with no visual indications of infection.",
        "Bacterial Pneumonia": "The scan shows focal consolidations consistent with bacterial pneumonia.",
        "Viral Pneumonia": "The scan shows diffuse infiltrates matching viral pneumonia patterns."
    }
    explanation = explanations.get(record["predicted_class"], "Clinical correlation required.")

    return render_template('result.html', record=record, explanation=explanation)


# ========================= DOWNLOAD REPORT =========================
@app.route('/download_report/<int:record_id>')
@login_required
def download_report(record_id):
    record = db.get_record(record_id)
    if not record:
        return "Record not found", 404

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.platypus.flowables import HRFlowable

        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"report_{record_id}.pdf")

        doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                                rightMargin=40, leftMargin=40,
                                topMargin=40, bottomMargin=30)
        styles = getSampleStyleSheet()
        story = []

        title = Paragraph("<font size=22 color='#0d9488'><b>PNEUMONIA DIAGNOSTIC SCAN REPORT</b></font>", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 15))
        story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#0d9488")))
        story.append(Spacer(1, 20))

        analysis_date = datetime.now().strftime("%B %d, %Y - %I:%M %p")
        confidence_pct = f"{record['confidence'] * 100:.2f}%"
        details_data = [
            ["Scan Reference ID:", f"PND-{record_id:05d}"],
            ["Diagnostic System:", "ResNet50 Classifier"],
            ["Analysis Date:", analysis_date],
            ["Model Version:", "v2.1.0"]
        ]
        details_table = Table(details_data, colWidths=[180, 300])
        details_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#e6fffb")),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(details_table)
        story.append(Spacer(1, 25))

        summary_title = Paragraph("<font size=16 color='#0d9488'><b>Diagnostic Summary</b></font>", styles['Heading2'])
        story.append(summary_title)
        story.append(Spacer(1, 10))
        summary_data = [
            ["Predicted Classification:", record["predicted_class"].upper()],
            ["Model Confidence:", confidence_pct]
        ]
        summary_table = Table(summary_data, colWidths=[220, 260])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f0fdfa")),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 0.6, colors.grey)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 30))

        image_title = Paragraph("<font size=16 color='#0d9488'><b>Radiograph & Grad-CAM Heatmap</b></font>", styles['Heading2'])
        story.append(image_title)
        story.append(Spacer(1, 15))

        original_img_path = os.path.join(BASE_DIR, record["filename"].lstrip('/'))
        gradcam_rel = record.get("gradcam_path", "")
        gradcam_img_path = os.path.join(BASE_DIR, gradcam_rel.lstrip('/')) if gradcam_rel else None

        image_row = []
        if os.path.exists(original_img_path):
            image_row.append(RLImage(original_img_path, width=220, height=220))
        if gradcam_img_path and os.path.exists(gradcam_img_path):
            image_row.append(RLImage(gradcam_img_path, width=220, height=220))

        if image_row:
            img_table = Table([image_row])
            img_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
            story.append(img_table)

        story.append(Spacer(1, 30))

        disclaimer_title = Paragraph("<font size=14 color='red'><b>CLINICAL DISCLAIMER</b></font>", styles['Heading3'])
        story.append(disclaimer_title)
        story.append(Spacer(1, 10))
        disclaimer_text = Paragraph(
            """This AI-generated report is for educational and supportive clinical workflow purposes only. 
            All findings must be reviewed and confirmed by a certified medical professional or radiologist.
            PneuVision AI should not be used as a standalone diagnostic tool.""",
            styles['BodyText']
        )
        story.append(disclaimer_text)
        story.append(Spacer(1, 25))

        footer = Paragraph("<font size=9 color='grey'>© 2026 PneuVision AI • Deep Learning Powered Diagnostic Support</font>", styles['BodyText'])
        story.append(footer)

        doc.build(story)
        return send_file(pdf_path, as_attachment=True, download_name=f"pneumonia_report_{record_id}.pdf")

    except Exception as e:
        return f"PDF generation failed: {str(e)}", 500


# ========================= CONTACT PAGE =========================
@app.route('/contact', methods=['GET', 'POST'])
@login_required
def contact_page():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        subject = request.form.get('subject', '').strip()
        message = request.form.get('message', '').strip()

        if not all([name, email, subject, message]):
            flash("Please fill all fields.", "danger")
            return redirect(url_for('contact_page'))

        save_contact_message(name, email, subject, message)
        flash("Thank you! Your message has been sent.", "success")
        return redirect(url_for('contact_page'))

    return render_template('contact.html')


# ------------------- PROFILE ROUTE -------------------
PROFILE_UPLOAD_FOLDER = os.path.join('static', 'profile_pics')
os.makedirs(PROFILE_UPLOAD_FOLDER, exist_ok=True)


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_id = session['user_id']
    user = db.get_user_by_id(user_id)

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        bio = request.form.get('bio', '').strip()
        profile_pic_path = user.get('profile_pic', '')

        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file and allowed_file(file.filename):
                if profile_pic_path:
                    old_path = os.path.join(BASE_DIR, profile_pic_path.lstrip('/'))
                    if os.path.exists(old_path):
                        os.remove(old_path)

                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                filename = secure_filename(f"profile_{user_id}_{timestamp}_{file.filename}")
                filepath = os.path.join(PROFILE_UPLOAD_FOLDER, filename)
                file.save(filepath)
                profile_pic_path = f"/static/profile_pics/{filename}"

        db.update_user_profile(user_id, full_name, bio, profile_pic_path)
        flash("Profile updated successfully!", "success")
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user)


# ========================= RUN APP =========================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)