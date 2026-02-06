from flask import Flask, render_template, request, jsonify, Response, flash
import re
import cv2
import pandas as pd
import numpy as np
import os
import tempfile
import pytesseract
from PIL import Image
import fitz
from io import BytesIO
from werkzeug.datastructures import FileStorage
import base64
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy.dialects.mysql import LONGBLOB, JSON
from sqlalchemy import LargeBinary, func
import math

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@localhost/sonalika'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class ClientKYC(db.Model):
    __tablename__ = "client_kyc"

    id = db.Column(db.Integer, primary_key=True)

    # =====================
    # PERSONAL INFORMATION
    # =====================
    full_name = db.Column(db.String(150))
    company_name = db.Column(db.String(200))
    mobile = db.Column(db.String(20))
    alt_phone = db.Column(db.String(20))
    office_phone = db.Column(db.String(20))
    landline = db.Column(db.String(20))
    email = db.Column(db.String(150))
    address = db.Column(db.Text)

    # =====================
    # BUSINESS INFORMATION
    # =====================
    gst_number = db.Column(db.String(30))
    company_pan = db.Column(db.String(20))
    pan_number = db.Column(db.String(20))
    aadhar_number = db.Column(db.String(20))
    msme_registration = db.Column(db.String(50))
    diamond_weight_lazer_marking = db.Column(db.String(50))
    iec_code = db.Column(db.String(30))

    # =====================
    # DOCUMENTS (LONGBLOB)
    # =====================
    aadhar_doc = db.Column(LONGBLOB)
    gst_doc = db.Column(LONGBLOB)
    pan_doc = db.Column(LONGBLOB)
    msme_doc = db.Column(LONGBLOB)
    iec_doc = db.Column(LONGBLOB)
    visiting_card_doc = db.Column(LONGBLOB)

class StyleImage(db.Model):
    __tablename__ = "style_images"

    id = db.Column(db.Integer, primary_key=True)

    style_no = db.Column(db.String(100), nullable=False, index=True)

    image = db.Column(LONGBLOB, nullable=False)

    # 🔹 NEW FIELDS
    gold_wt = db.Column(db.Float, nullable=True)          # 1.800
    total_gem_count = db.Column(db.Integer, nullable=True)  # 14
    round_details = db.Column(JSON, nullable=True)        # list of rounds

class DiamondSieveMaster(db.Model):
    __tablename__ = "diamond_sieve_master"

    id = db.Column(db.Integer, primary_key=True)

    sieve_range = db.Column(db.String(50), nullable=False)
    sieve_size = db.Column(db.String(20), nullable=False)
    mm_size = db.Column(db.Float, nullable=False)
    no_of_stones = db.Column(db.Integer, nullable=False)
    ct_weight_per_piece = db.Column(db.Float, nullable=False)




# -----------------------------
# MAIN ROUTES
# -----------------------------

@app.route("/upload-docs", methods=["POST"])
def upload_docs():

    def file_to_bytes(file):
        if file and file.filename != "":
            return file.read()
        return None

    kyc = ClientKYC(

        # =====================
        # PERSONAL INFORMATION
        # =====================
        full_name=request.form.get("full_name"),
        company_name=request.form.get("company_name"),
        mobile=request.form.get("mobile"),
        alt_phone=request.form.get("alt_phone"),
        office_phone=request.form.get("office_phone"),
        landline=request.form.get("landline"),
        email=request.form.get("email"),
        address=request.form.get("address"),

        # =====================
        # BUSINESS INFORMATION
        # =====================
        gst_number=request.form.get("gst_number"),
        company_pan=request.form.get("company_pan"),
        pan_number=request.form.get("pan_number"),
        aadhar_number=request.form.get("aadhar_number"),
        msme_registration=request.form.get("msme_registration"),
        diamond_weight_lazer_marking=request.form.get("diamond_weight_lazer_marking"),
        iec_code=request.form.get("iec_code"),

        # =====================
        # DOCUMENTS (BLOB)
        # =====================
        aadhar_doc=file_to_bytes(request.files.get("aadhar_doc")),
        gst_doc=file_to_bytes(request.files.get("gst_doc")),
        pan_doc=file_to_bytes(request.files.get("pan_doc")),
        msme_doc=file_to_bytes(request.files.get("msme_doc")),
        iec_doc=file_to_bytes(request.files.get("iec_doc")),
        visiting_card_doc=file_to_bytes(request.files.get("visiting_card_doc")),
    )

    db.session.add(kyc)
    db.session.commit()

    return jsonify({
        "status": "success",
        "message": "Client KYC submitted successfully",
        "id": kyc.id
    })


@app.route("/")
def login():
    return render_template("signin.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/signup")
def signup():
    return render_template("signup.html")

# -----------------------------
# PAGES ROUTES (ERP MODULES)
# -----------------------------

@app.route("/table")
def table():
    return render_template("table.html")

@app.route("/form")
def form():
    return render_template("form.html")

@app.route("/chart")
def chart():
    return render_template("chart.html")

@app.route("/button")
def button():
    return render_template("button.html")

@app.route("/widget")
def widget():
    return render_template("widget.html")

@app.route("/element")
def element():
    return render_template("element.html")

@app.route("/typography")
def typography():
    return render_template("typography.html")

@app.route("/blank")
def blank():
    return render_template("blank.html")

@app.route("/error")
def error():
    return render_template("404.html")

@app.route("/create_order")
def create_order():
    clients = ClientKYC.query.order_by(ClientKYC.id.desc()).all()
    return render_template("create_order.html", clients=clients)

@app.route("/Client_kyc")
def Client_kyc():
    return render_template("Client_kyc.html")

@app.route("/upload-doc", methods=["POST"])
def upload_doc():
    file = request.files.get("file")
    doc_type = request.form.get("doc_type")

    if not file:
        return jsonify({})

    result = process_image_ocr(file, doc_type)
    return jsonify(result)

    # ---------- Upload Pdf doc to DB ----------

@app.route("/upload-pdf-doc", methods=["POST"])
def upload_pdf_doc():
    file = request.files.get("file")
    doc_type = request.form.get("doc_type")

    if not file:
        return jsonify({})

    pdf_bytes = file.read()
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    if pdf.page_count == 0:
        return jsonify({})

    page = pdf.load_page(0)
    pix = page.get_pixmap(dpi=300)

    img_bytes = pix.tobytes("png")
    img_io = BytesIO(img_bytes)

    image_file = FileStorage(
        stream=img_io,
        filename="page1.png",
        content_type="image/png"
    )

    # ✅ ONLY place jsonify here
    result = process_image_ocr(image_file, doc_type)
    return jsonify(result)


# ---------- Total ocr Functions ----------

def process_image_ocr(file, doc_type):

    if not file:
        return {}

    # ---------- IMAGE READ ----------
    image = Image.open(file).convert("RGB")
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # ---------- OCR ----------
    text = pytesseract.image_to_string(
        thresh,
        config="--oem 3 --psm 6"
    )

    print("===== OCR TEXT =====")
    print(text)
    print("====================")

    clean_text = text.upper()

    # ---------- FIX COMMON OCR ERRORS ----------
    clean_text = (
        clean_text
        .replace("GMAILL", "GMAIL")
        .replace("GMAILLC", "GMAIL")
        .replace("GMAIL.C0M", "GMAIL.COM")
    )

    response = {}

    # ================= AADHAR =================
    if doc_type == "aadhar":
        match = re.search(r"\d[\d\s]{11,14}", clean_text)
        if match:
            aadhar = re.sub(r"\s+", "", match.group())
            if len(aadhar) == 12:
                response["aadhar_number"] = aadhar

    # ================= PAN =================
    elif doc_type == "pan":

        lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
        pan_text = re.sub(r"[^A-Z0-9]", "", clean_text)

        for i in range(len(pan_text) - 9):
            chunk = pan_text[i:i+10]
            fixed = ""

            for idx, ch in enumerate(chunk):
                if idx < 5:
                    ch = ch.replace("0", "O").replace("1", "I")
                    if not ch.isalpha():
                        break
                elif idx < 9:
                    ch = ch.replace("O", "0").replace("I", "1")
                    if not ch.isdigit():
                        break
                else:
                    ch = ch.replace("0", "O").replace("1", "I")
                    if not ch.isalpha():
                        break

                fixed += ch
            else:
                response["pan_number"] = fixed
                break

        for i, line in enumerate(lines):
            if "NAME" in line.upper():
                if i + 1 < len(lines):
                    name = lines[i + 1]
                    name = re.sub(r"[^A-Z ]", "", name)
                    if len(name.split()) >= 2:
                        response["full_name"] = name.strip()
                break

    # ================= VISITING CARD =================
    elif doc_type == "visiting_card":

        mobile_match = re.search(r"(?:\+91[:\-\s]?)?[6-9]\d{9}", clean_text)
        if mobile_match:
            response["mobile"] = mobile_match.group().replace(":", "").replace(" ", "")

        email_text = clean_text.replace(" ", "")
        email_text = email_text.replace("A@", "@").replace(",COM", ".COM").replace(";COM", ".COM")
        email_match = re.search(
            r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
            email_text,
            re.IGNORECASE
        )
        if email_match:
            response["email"] = email_match.group().lower()

        address_lines = []
        for line in clean_text.splitlines():
            line = line.strip()
            if line and not re.search(r"(?:\+91[:\-\s]?)?[6-9]\d{9}", line) \
               and not re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", line, re.IGNORECASE):
                address_lines.append(line)

        if address_lines:
            response["address"] = " ".join(address_lines[-3:]).title()

    # ================= MSME / UDYAM =================
    elif doc_type == "msme":

        udyam_match = re.search(
            r"UDYAM[-\s]*[A-Z]{2}[-\s]*\d{2}[-\s]*\d{7}",
            clean_text
        )
        if udyam_match:
            response["udyam_number"] = udyam_match.group().replace(" ", "")

        lines = clean_text.splitlines()
        for i, line in enumerate(lines):
            if "NAME OF ENTERPRISE" in line:
                same_line = line.replace("NAME OF ENTERPRISE", "").strip()

                if len(same_line) > 3:
                    name = re.sub(r"[^A-Z ]", "", same_line)
                    name = re.sub(r"\s+", " ", name).strip()
                    response["enterprise_name"] = name.title()
                elif i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    name = re.sub(r"[^A-Z ]", "", next_line)
                    name = re.sub(r"\s+", " ", name).strip()
                    response["enterprise_name"] = name.title()
                break
            # ================= IEC =================
    elif doc_type == "iec":
    
        # ---------- IEC CODE ----------
        iec_match = re.search(
            r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
            clean_text
        )
        if iec_match:
            response["iec_code"] = iec_match.group()
    
        # ---------- FIRM NAME ----------
        lines = clean_text.splitlines()
        for i, line in enumerate(lines):
    
            if "FIRM NAME" in line or "NAME OF FIRM" in line:
                same_line = (
                    line.replace("FIRM NAME", "")
                        .replace("NAME OF FIRM", "")
                        .strip()
                )
    
                # CASE 1: Name on same line
                if len(same_line) > 3:
                    name = re.sub(r"[^A-Z ]", "", same_line)
                    name = re.sub(r"\s+", " ", name).strip()
                    response["firm_name"] = name.title()
    
                # CASE 2: Name on next line
                elif i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    name = re.sub(r"[^A-Z ]", "", next_line)
                    name = re.sub(r"\s+", " ", name).strip()
                    response["firm_name"] = name.title()
    
                break
        # ================= GST =================
    elif doc_type == "gst":
    
        # ---------- CLEAN OCR TEXT ----------
        clean_text_upper = clean_text.upper()           # uppercase for consistency
        clean_text_no_space = re.sub(r"[\s]", "", clean_text_upper)  # remove spaces/newlines
    
        # ---------- GST NUMBER ----------
        gst_pattern = r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]"
        gst_match = re.search(gst_pattern, clean_text_no_space)
        if gst_match:
            response["gst_number"] = gst_match.group()
    
        # ---------- COMPANY NAME (TRADE / LEGAL) ----------
        lines = clean_text.splitlines()
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
    
            # ----- Prefer TRADE NAME -----
            if "TRADE NAME" in line_upper:
                # Remove the label text
                possible_name = line_upper.replace("TRADE NAME", "").replace("IF ANY", "").strip()
    
                if len(possible_name) > 3:
                    # Keep only letters and spaces
                    name = re.sub(r"[^A-Z ]", "", possible_name)
                    response["company_name"] = re.sub(r"\s+", " ", name).title()
                elif i + 1 < len(lines):
                    # fallback to next line
                    next_line = lines[i + 1].strip().upper()
                    name = re.sub(r"[^A-Z ]", "", next_line)
                    response["company_name"] = re.sub(r"\s+", " ", name).title()
                break
    
    return response

def extract_style_no(img):
    crop = img[0:280, 0:520]   # exact STYLE NO box
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    text = pytesseract.image_to_string(
        gray,
        config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    for line in text.splitlines():
        line = line.strip().replace(" ", "")
        match = re.search(r"SJ[A-Z0-9]{3,}", line)
        if match:
            return match.group()

    return None



# ---------------- CROP GOLD + GEM ----------------
def crop_gold_gem(img):
    h, w, _ = img.shape

    y1 = int(h * 0.28) # start from TOP
    y2 = int(h * 0.60) # go DOWN

    x1 = int(w * 0.0)  # start from LEFT
    x2 = int(w * 0.70) # go RIGHT

    return img[y1:y2, x1:x2]

# ---------------- BLOCK GEM ----------------
def blur_gems(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([90, 40, 50])
    upper = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    blurred = cv2.GaussianBlur(img, (25, 25), 0)
    img[mask > 0] = blurred[mask > 0]

    return img


# ---------------- extract_gold_wt ----------------

def extract_gold_wt(img):
    h, w, _ = img.shape

    # ✅ correct crop
    crop = img[
        int(h * 0.148):int(h * 0.176),
        int(w * 0.56):int(w * 0.98)
    ]

    # preprocess (text is small → must enlarge)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

 #   cv2.imwrite("DEBUG_GOLD_crop.jpg", crop)
 #   cv2.imwrite("DEBUG_GOLD_ready.jpg", gray)

    text = pytesseract.image_to_string(
        gray,
        config="--psm 7 -c tessedit_char_whitelist=0123456789."
    )

    print("GOLD WT OCR TEXT >>>", repr(text))

    match = re.search(r"(\d+\.\d+)", text)
    return float(match.group(1)) if match else None



def crop_cad_panel(img):
    h, w, _ = img.shape
    return img[
        int(h * 0.35):int(h * 0.56),
        int(w * 0.00):int(w * 0.46)
    ]



# ---------------- extract_total_gem_count ----------------


def extract_total_gem_count(img, debug=True):
    """
    Extracts Total Gem Count from CAD panel using HSV cyan mask.
    Returns: int or None
    Saves debug: DEBUG_GEM_crop.jpg, DEBUG_GEM_ready_FIXED.jpg (if debug=True)
    """

    h, w, _ = img.shape

    # ✅ 1) CAD panel (keep same)
    cad = img[
        int(h * 0.35):int(h * 0.56),
        int(w * 0.12):int(w * 0.38)
    ]

    ch, cw, _ = cad.shape

    # ✅ 2) GEM COUNT bar crop (keep same)
    crop = cad[
        int(ch * 0.00):int(ch * 0.24),
        int(cw * 0.22):int(cw * 1.00)
    ]

    if debug:
        cv2.imwrite("DEBUG_GEM_crop.jpg", crop)

    # ✅ 3) HSV to isolate cyan/teal digits
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # (works for your images)
    lower_cyan = np.array([80, 100, 100])
    upper_cyan = np.array([100, 255, 255])

    mask = cv2.inRange(hsv, lower_cyan, upper_cyan)

    # ✅ 4) Make OCR-ready
    ready = cv2.resize(mask, None, fx=7, fy=7, interpolation=cv2.INTER_CUBIC)
    ready = cv2.GaussianBlur(ready, (3, 3), 0)

    # Invert -> black digits on white (Tesseract friendly)
    ready = cv2.bitwise_not(ready)

    if debug:
        cv2.imwrite("DEBUG_GEM_ready_FIXED.jpg", ready)

    # ✅ 5) OCR digits
    text = pytesseract.image_to_string(
        ready,
        config="--psm 6 -c tessedit_char_whitelist=0123456789"
    )

    print("GEM COUNT OCR TEXT >>>", repr(text))

    m = re.search(r"\d+", text)
    return int(m.group()) if m else None






# ---------------- extract_round_details ----------------


def extract_round_details(img):
    """
    Extracts round stone details by using high-contrast thresholding 
    to handle dark backgrounds and cyan text.
    """
    # 1️⃣ Crop CAD panel
    cad = crop_cad_panel(img)
    ch, cw, _ = cad.shape

    # 2️⃣ Crop numeric columns 
    # Starts at 0.12 to skip the colored icons on the left
    crop = cad[
        int(ch * 0.22):int(ch * 1.00), 
        int(cw * 0.12):int(cw * 0.90)  
    ]

    # 3️⃣ High-Contrast Preprocessing
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ready = cv2.resize(gray, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    
    # Binary Threshold to isolate text
    _, ready = cv2.threshold(ready, 125, 255, cv2.THRESH_BINARY_INV)
    
    # Dilation makes thin cyan text bolder
    kernel = np.ones((2,2), np.uint8)
    ready = cv2.dilate(ready, kernel, iterations=1)

    cv2.imwrite("DEBUG_ROUND_ready.jpg", ready)

    # 4️⃣ OCR
    text = pytesseract.image_to_string(ready, config="--psm 6")
    print("RAW OCR TEXT >>>\n", repr(text))

    rounds = []
    # --- INDENTATION FIXED BELOW ---
    for line in text.splitlines():
        # 1. Clean common OCR noise
        line = line.replace('×', 'x').replace('X', 'x').replace('|', ' ').replace('=', ' ')
        line = line.lower().strip()

        # 2. Relaxed Regex: Handles symbols like @, |, or = between numbers
        # (\d+\.\d+) -> size 1
        # \D+        -> any non-digit (skip the 'x' and symbols)
        # (\d+\.\d+) -> size 2
        # \D+        -> skip symbols/spaces
        # (\d+)      -> pcs
        # \D+        -> skip symbols/spaces
        # (\d+\.\d+) -> weight
        match = re.search(r"(\d+\.\d+)\D+(\d+\.\d+)\D+(\d+)\D+(\d+\.\d+)", line)

        if match:
            rounds.append({
                "shape": "Round",
                "size": f"{match.group(1)} x {match.group(2)}",
                "pcs": int(match.group(3)),
                "ct": float(match.group(4))
            })

    return rounds






def ocr_ready_no_threshold(crop, scale=8):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # 🔥 NO THRESHOLD
    return gray


 

# ---------- Job Card Folder Uploding db ----------

@app.route("/upload-folder", methods=["POST"])
def upload_folder():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    saved = 0
    ocr_failed = 0
    debug_limit = 3          # ✅ only save debug for first 3 images
    debug_count = 0

    for file in files:
        try:
            file_bytes = file.read()
            npimg = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if img is None:
                ocr_failed += 1
                continue

            style_no = extract_style_no(img)
            if not style_no:
                ocr_failed += 1
                continue

            gold_wt = extract_gold_wt(img)
            total_gem_count = extract_total_gem_count(img)
            round_details = extract_round_details(img)
            
            print("FINAL VALUES >>>")
            print("STYLE:", style_no)
            print("GOLD WT:", gold_wt)
            print("GEM COUNT:", total_gem_count)
            print("ROUNDS:", round_details)

            # ✅ debug only first few
            if debug_count < debug_limit:
                draw_cad_debug_boxes(img, out_path=f"DEBUG_JOB_CARD_BOXES_{debug_count+1}.jpg")
                debug_count += 1

            # ✅ if gold not found, skip saving
            if gold_wt is None:
                ocr_failed += 1
                continue

            # store processed image (your old crop + blur)
            crop = crop_gold_gem(img)
            final_img = blur_gems(crop)

            success, buffer = cv2.imencode(".jpg", final_img)
            if not success:
                ocr_failed += 1
                continue

            record = StyleImage(
                style_no=style_no,
                image=buffer.tobytes(),
                gold_wt=gold_wt,
                total_gem_count=total_gem_count,
                round_details=round_details
            )

            db.session.add(record)
            saved += 1

        except Exception as e:
            print("ERROR >>>", e)
            ocr_failed += 1



    db.session.commit()

    return jsonify({
        "status": "success",
        "uploaded": len(files),
        "saved": saved,
        "ocr_failed": ocr_failed,
        "debug_saved": debug_count
    })


def draw_cad_debug_boxes(img, out_path="DEBUG_CAD_BOXES.jpg"):
    h, w, _ = img.shape
    debug = img.copy()

    # 🟣 CAD PANEL (tight)
    cx1, cy1 = int(w * 0.00), int(h * 0.35)
    cx2, cy2 = int(w * 0.46), int(h * 0.56)

    cv2.rectangle(debug, (cx1, cy1), (cx2, cy2), (255, 0, 255), 3)
    cv2.putText(debug, "CAD PANEL", (cx1 + 5, max(30, cy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # 🟢 GEM COUNT (top black bar - right side)
    gx1, gy1 = int(w * 0.10), int(h * 0.35)
    gx2, gy2 = int(w * 0.46), int(h * 0.40)

    cv2.rectangle(debug, (gx1, gy1), (gx2, gy2), (0, 255, 0), 3)
    cv2.putText(debug, "GEM COUNT", (gx1 + 5, max(30, gy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 🔵 ROUND ROWS (3 rows area)
    rx1, ry1 = int(w * 0.04), int(h * 0.40)
    rx2, ry2 = int(w * 0.46), int(h * 0.515)

    cv2.rectangle(debug, (rx1, ry1), (rx2, ry2), (255, 0, 0), 3)
    cv2.putText(debug, "ROUND ROWS", (rx1 + 5, max(30, ry1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imwrite(out_path, debug)
    print(f"✅ CAD debug saved: {out_path}")


# ---------- get style no imge ----------
@app.route("/api/style/<style_no>", methods=["GET"])
def api_get_style(style_no):
    row = StyleImage.query.filter_by(style_no=style_no).first()

    if not row:
        return jsonify({"ok": False, "message": "Style not found"}), 404

    # LONGBLOB -> base64 -> data URL
    img_b64 = base64.b64encode(row.image).decode("utf-8")
    img_url = f"data:image/jpeg;base64,{img_b64}"  # if png, change to image/png

    return jsonify({
        "ok": True,
        "style_no": row.style_no,
        "image_url": img_url,
        "gold_wt": row.gold_wt,
        "total_gem_count": row.total_gem_count,
        "round_details": row.round_details  # JSON field
    })

# ---------- get Cad card details ----------
@app.route("/api/styles", methods=["GET"])
def api_styles():
    styles = StyleImage.query.with_entities(StyleImage.style_no).order_by(StyleImage.style_no.asc()).all()
    return jsonify([s[0] for s in styles])
# ---------- Get Client Names ----------
@app.route("/api/clients")
def api_clients():
    rows = ClientKYC.query.with_entities(
        ClientKYC.id,
        ClientKYC.full_name
    ).all()

    return jsonify([
        {"id": r.id, "full_name": r.full_name}
        for r in rows
    ])


    # ---------- Sieve api ----------

#@app.route("/upload-sieve", methods=["GET", "POST"])
#def upload_sieve():
#    if request.method == "POST":
#        file = request.files.get("excel_file")
#        if not file or file.filename == "":
#            flash("❌ Please select an Excel file", "danger")
#            return redirect(url_for("upload_sieve"))
#
#        try:
#            df = pd.read_excel(file)
#
#            # ✅ 1) FIX: remove hidden \n in column names
#            df.columns = df.columns.astype(str).str.strip()
#
#            # ✅ 2) Clean string cells (remove \n, extra spaces)
#            def clean_text(x):
#                if pd.isna(x):
#                    return None
#                if isinstance(x, str):
#                    x = x.replace("\n", " ").strip()
#                    x = re.sub(r"\s+", " ", x)   # multiple spaces -> single space
#                    return x
#                return x
#
#            df = df.applymap(clean_text)
#
#            # ✅ 3) Optional: normalize sieve_size format (+ 000 - 00 -> +000-00)
#            if "sieve_size" in df.columns:
#                df["sieve_size"] = df["sieve_size"].astype(str).str.replace(" ", "", regex=False)
#
#            # ✅ 4) Validate required columns
#            required_cols = ["sieve_range", "sieve_size", "mm_size", "no_of_stones", "ct_weight_per_piece"]
#            missing = [c for c in required_cols if c not in df.columns]
#            if missing:
#                flash(f"❌ Missing columns in Excel: {missing}", "danger")
#                return redirect(url_for("upload_sieve"))
#
#            # ✅ 5) Insert into DB
#            for _, row in df.iterrows():
#                record = DiamondSieveMaster(
#                    sieve_range=str(row["sieve_range"]).strip(),
#                    sieve_size=str(row["sieve_size"]).strip(),
#                    mm_size=float(row["mm_size"]),
#                    no_of_stones=int(row["no_of_stones"]),
#                    ct_weight_per_piece=float(row["ct_weight_per_piece"])
#                )
#                db.session.add(record)
#
#            db.session.commit()
#            flash("✅ Sieve chart uploaded & saved successfully!", "success")
#            return redirect(url_for("upload_sieve"))
#
#        except Exception as e:
#            db.session.rollback()
#            flash(f"❌ Upload failed: {str(e)}", "danger")
#            return redirect(url_for("upload_sieve"))
#
#    return render_template("upload_sieve.html")
@app.route("/api/sieve-lookup-batch", methods=["POST"])
def sieve_lookup_batch():
    payload = request.get_json(silent=True) or {}
    mm_list = payload.get("mm_list", [])

    out = {}

    for mm in mm_list:
        try:
            mm = float(mm)
        except:
            out[str(mm)] = None
            continue

        mm_key = math.floor(mm * 10) / 10.0  # 1.75 -> 1.7

        row = (DiamondSieveMaster.query
               .filter(func.round(DiamondSieveMaster.mm_size, 1) == mm_key)
               .first())

        out[str(mm)] = None
        if row:
            out[str(mm)] = {
                "mm_key": mm_key,
                "sieve_range": row.sieve_range,
                "sieve_size": row.sieve_size,
                "ct_weight_per_piece": float(row.ct_weight_per_piece),
            }

    return jsonify(out)



# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)
