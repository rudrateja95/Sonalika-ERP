from flask import Flask, render_template, request, jsonify, Response, flash
import re
import cv2
import pandas as pd
import numpy as np
import os
import tempfile
import pytesseract
from pytesseract import Output
from PIL import Image
import fitz
from io import BytesIO
from werkzeug.datastructures import FileStorage
import base64
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy.dialects.mysql import LONGBLOB, JSON
from sqlalchemy import LargeBinary, func, and_
import math
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@localhost/sonalika'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"




class ClientKYC(db.Model):
    __tablename__ = "client_kyc"

    id = db.Column(db.Integer, primary_key=True)

    # =====================
    # PERSONAL INFORMATION
    # =====================
    full_name = db.Column(db.String(150), nullable=False)
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
    round_details = db.Column(JSON, nullable=True)        # list of rounds

class DiamondSieveMaster(db.Model):
    __tablename__ = "diamond_sieve_master"

    id = db.Column(db.Integer, primary_key=True)

    sieve_range = db.Column(db.String(50), nullable=False)
    sieve_size = db.Column(db.String(20), nullable=False)
    mm_size = db.Column(db.Float, nullable=False)
    no_of_stones = db.Column(db.Integer, nullable=False)
    ct_weight_per_piece = db.Column(db.Float, nullable=False)

class ProductionOrder(db.Model):
    __tablename__ = "production_orders"

    id = db.Column(db.Integer, primary_key=True)

    client_id = db.Column(db.Integer, db.ForeignKey("client_kyc.id"), nullable=False)
    client = db.relationship("ClientKYC", backref="production_orders")

    order_datetime = db.Column(db.DateTime, nullable=False)
    delivery_datetime = db.Column(db.DateTime)

    style_no = db.Column(db.String(100), nullable=False)

    diamond_clarity = db.Column(db.String(20))
    gold_color = db.Column(db.String(30))
    diamond_color = db.Column(db.String(20))

    gold_purity = db.Column(db.String(20))
    gold_purity_factor = db.Column(db.Float)

    total_amount = db.Column(db.Float, default=0)
    remark = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


    





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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    data = pytesseract.image_to_data(thr, output_type=Output.DICT, config="--psm 6")

    # group words by (block, par, line) so we can read whole line
    lines = {}
    n = len(data["text"])
    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append((data["left"][i], word))

    # search each line for GM/gm and take number before it
    for key, words in lines.items():
        words.sort(key=lambda x: x[0])
        line = " ".join(w for _, w in words)
        low = line.lower().replace(" ", "")

        if "gm" not in low:
            continue

        # extract number in that line (prefer the one closest to gm)
        # examples: "GOLD WT = 3.000 GM"
        nums = re.findall(r"\d+\.\d+", low)
        if not nums:
            continue

        val = float(nums[-1])  # usually last float on that line is weight
        # sanity: gold wt typically 1..30 gm
        if 0.5 <= val <= 30:
            return round(val, 3)

    return None




def crop_cad_panel(img):
    h, w, _ = img.shape
    return img[
        int(h * 0.35):int(h * 0.56),
        int(w * 0.00):int(w * 0.46)
    ]




def normalize_size_token(tok):
    t = str(tok).strip().replace(",", ".")
    t = re.sub(r"[^0-9.]", "", t)
    if not t: return 0.0
    if t.startswith("."): t = "0" + t
    # Handle OCR missing decimal: "90" -> 0.90
    if t.isdigit() and len(t) == 2: return float(t) / 100.0
    try: return float(t)
    except: return 0.0

def ocr_rows_with_alignment(img_bin, y_threshold=20):
    # Use image_to_data to get bounding box coordinates
    data = pytesseract.image_to_data(img_bin, output_type=Output.DICT, config="--psm 6")
    tokens = []
    
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        try:
            conf = int(float(data["conf"][i]))
        except:
            conf = -1
            
        if text and conf > 30:
            # Calculate the vertical center of the text token
            tokens.append({
                "text": text,
                "y": data["top"][i] + (data["height"][i] / 2),
                "x": data["left"][i]
            })

    if not tokens: return []

    # Sort tokens from top to bottom
    tokens.sort(key=lambda t: t['y'])
    
    merged_lines = []
    current_row = [tokens[0]]

    # Group tokens that share a similar Y-coordinate
    for i in range(1, len(tokens)):
        if abs(tokens[i]['y'] - current_row[-1]['y']) <= y_threshold:
            current_row.append(tokens[i])
        else:
            # Sort the completed row from Left to Right (X-axis)
            current_row.sort(key=lambda t: t['x'])
            merged_lines.append(" ".join([t['text'] for t in current_row]))
            current_row = [tokens[i]]
    
    current_row.sort(key=lambda t: t['x'])
    merged_lines.append(" ".join([t['text'] for t in current_row]))
    return merged_lines

def extract_round_details(img):
    # 1. Crop to the CAD panel table area
    cad = crop_cad_panel(img)
    ch, cw, _ = cad.shape
    crop = cad[int(ch * 0.05):int(ch * 0.95), int(cw * 0.16):int(cw * 0.80)]

    # 2. Pre-process for thin white/green text on dark background
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. OCR with Y-Alignment to catch the PCS counts (5, 17, etc.)
    ocr_lines = ocr_rows_with_alignment(th)
    
    # 4. Final Extraction Logic
    rounds = []
    for line in ocr_lines:
        line = line.replace('X', 'x').replace('x', ' x ')
        # Find: [Size1] x [Size2] followed by any characters then [PCS]
        m = re.search(r"(\d?\.?\d+)\s*x\s*(\d?\.?\d+)(.*)", line)
        if m:
            s1, s2, tail = m.groups()
            size_str = f"{normalize_size_token(s1):.2f} x {normalize_size_token(s2):.2f}"
            
            # Find the PCS: first integer in the tail
            pcs_match = re.search(r"(\d+)", tail)
            pcs = int(pcs_match.group(1)) if pcs_match else 0
            
            if size_str != "0.00 x 0.00":
                rounds.append({"shape": "Round", "size": size_str, "pcs": pcs})

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
            
            round_details = extract_round_details(img)
            
            print("FINAL VALUES >>>")
            print("STYLE:", style_no)
            print("GOLD WT:", gold_wt) 
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

    # --- normalize mm list (round to 1 decimal)
    norm_mm = []
    for mm in mm_list:
        try:
            mm = round(float(mm), 1)
            norm_mm.append(mm)
        except:
            pass

    if not norm_mm:
        return jsonify({})

    # --- tolerance (important for float safety)
    TOL = 0.05

    # --- fetch all possible rows in ONE query
    rows = (
        DiamondSieveMaster.query
        .filter(
            and_(
                DiamondSieveMaster.mm_size >= min(norm_mm) - TOL,
                DiamondSieveMaster.mm_size <= max(norm_mm) + TOL
            )
        )
        .all()
    )

    # --- map mm_size -> row
    sieve_map = {
        round(row.mm_size, 1): row
        for row in rows
    }

    # --- build response
    out = {}

    for raw_mm in mm_list:
        try:
            mm = round(float(raw_mm), 1)
        except:
            out[str(raw_mm)] = None
            continue

        row = sieve_map.get(mm)

        out[str(raw_mm)] = None
        if row:
            out[str(raw_mm)] = {
                "mm_key": mm,
                "sieve_range": row.sieve_range,
                "sieve_size": row.sieve_size,
                "ct_weight_per_piece": float(row.ct_weight_per_piece),
            }

    return jsonify(out)

def parse_dt(s):
    if not s:
        return None
    return datetime.fromisoformat(s)  # expects "YYYY-MM-DDTHH:MM"



@app.route("/production-board")
def production_board():
    orders = ProductionOrder.query.order_by(ProductionOrder.id.desc()).all()
    return render_template("production_board.html", orders=orders)




def parse_dt(s):
    if not s:
        return None
    # "YYYY-MM-DDTHH:MM"
    return datetime.fromisoformat(s)

@app.route("/api/create-order", methods=["POST"])
def api_create_order():
    data = request.get_json(silent=True) or {}

    client_id = int(data.get("client_id") or 0)
    style_no = (data.get("style_no") or "").strip()
    order_dt = parse_dt(data.get("order_datetime"))
    delivery_dt = parse_dt(data.get("delivery_datetime"))

    if not client_id:
        return jsonify({"ok": False, "error": "client_id required"}), 400
    if not order_dt:
        return jsonify({"ok": False, "error": "order_datetime required"}), 400
    if not style_no:
        return jsonify({"ok": False, "error": "style_no required"}), 400

    gold_purity_factor = data.get("gold_purity_factor")
    gold_purity_factor = float(gold_purity_factor) if gold_purity_factor not in (None, "", "null") else None

    row = ProductionOrder(
        client_id=client_id,
        order_datetime=order_dt,
        delivery_datetime=delivery_dt,
        style_no=style_no,

        diamond_clarity=data.get("diamond_clarity") or None,
        gold_color=data.get("gold_color") or None,
        diamond_color=data.get("diamond_color") or None,

        gold_purity=data.get("gold_purity") or None,
        gold_purity_factor=gold_purity_factor,

        total_amount=float(data.get("total_amount") or 0),
        remark=(data.get("remark") or "").strip()
    )

    db.session.add(row)
    db.session.commit()

    return jsonify({"ok": True, "order_id": row.id}), 201

@app.route("/production-board")
def production_board_page():
    rows = db.session.query(
        ProductionOrder,
        Client.full_name
    ).join(Client, Client.id == ProductionOrder.client_id)\
     .order_by(ProductionOrder.id.desc()).all()

    return render_template("production_board.html", rows=rows)



# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)
