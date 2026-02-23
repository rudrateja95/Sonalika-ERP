from flask import Flask, render_template, request, jsonify, Response, flash, session, url_for, redirect
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
from werkzeug.utils import secure_filename
from sqlalchemy.dialects.mysql import LONGBLOB, JSON
from sqlalchemy import LargeBinary, func, and_
import math
from datetime import datetime
import openpyxl

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



app = Flask(__name__)
app.secret_key='7527488f33cd3a731909c7d6e8aa8194d85e893f02a7d8548cb4b16aec47f64f'

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

class StyleNo(db.Model):
    __tablename__ = "style_no"

    id = db.Column(db.Integer, primary_key=True)
    style_no = db.Column(db.String(100), nullable=False, index=True)
    gold_wt = db.Column(db.Float, nullable=True)
    size = db.Column(db.String(50), nullable=True)
    pcs = db.Column(db.Integer, nullable=True)


class ImageCrop(db.Model):
    __tablename__ = "image_crop"

    id = db.Column(db.Integer, primary_key=True)
    style_no = db.Column(db.String(100), nullable=False, index=True)
    image = db.Column(LONGBLOB, nullable=False)


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

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False)


    

# -----------------------------
# MAIN ROUTES
# -----------------------------
@app.route("/api/styles", methods=["GET"])
def api_styles():
    rows = (
        db.session.query(StyleNo.style_no)
        .group_by(StyleNo.style_no)
        .order_by(db.func.max(StyleNo.id).desc())
        .all()
    )

    styles = [r[0] for r in rows if r[0]]
    return jsonify(styles)

@app.route("/api/style/<style_no>", methods=["GET"])
def api_style(style_no):
    # 1) rows for this style (many rows = many sizes)
    rows = (
        StyleNo.query
        .filter(StyleNo.style_no == style_no)
        .order_by(StyleNo.id.asc())
        .all()
    )

    if not rows:
        return jsonify({"ok": False, "message": "Style not found"}), 404

    # 2) gold weight (take first non-null)
    gold_wt = None
    for r in rows:
        if r.gold_wt is not None:
            gold_wt = float(r.gold_wt)
            break

    # 3) round_details from StyleNo table (size + pcs)
    # Your JS expects: [{size:"0.9", pcs:12}, ...]
    round_details = []
    for r in rows:
        if r.size and r.pcs is not None:
            round_details.append({
                "size": r.size,
                "pcs": int(r.pcs)
            })

    # 4) image URL (served by another endpoint)
    image_url = f"/api/style-image/{style_no}"

    return jsonify({
        "ok": True,
        "style_no": style_no,
        "gold_wt": gold_wt,
        "round_details": round_details,
        "image_url": image_url
    })


@app.route("/api/style-image/<style_no>", methods=["GET"])
def api_style_image(style_no):
    row = (
        ImageCrop.query
        .filter(ImageCrop.style_no == style_no)
        .order_by(ImageCrop.id.desc())
        .first()
    )

    if not row or not row.image:
        return ("", 404)

    img_bytes = row.image

    # If you always store JPEG, keep image/jpeg
    # If you store PNG, change to image/png
    return Response(img_bytes, mimetype="image/jpeg")


@app.route("/api/sieve-lookup-batch", methods=["POST"])
def sieve_lookup_batch():
    payload = request.get_json(silent=True) or {}
    mm_list = payload.get("mm_list", [])

    if not isinstance(mm_list, list) or not mm_list:
        return jsonify({})

    out = {}

    for raw_mm in mm_list:
        key = str(raw_mm)

        # ---- parse input mm
        try:
            mm = float(raw_mm)
        except Exception:
            out[key] = None
            continue

        # ---- FLOOR lookup + tie-break for duplicate mm_size
        # 1) pick closest mm_size below input
        # 2) if same mm_size exists multiple rows, pick the FIRST (lowest id)
        row = (
            DiamondSieveMaster.query
            .filter(DiamondSieveMaster.mm_size <= mm)
            .order_by(
                DiamondSieveMaster.mm_size.desc(),  # closest <= mm
                DiamondSieveMaster.id.asc()         # ✅ duplicates: first row only
            )
            .first()
        )

        if not row:
            out[key] = None
            continue

        out[key] = {
            "mm_input": mm,
            "mm_size": float(row.mm_size),
            "sieve_range": row.sieve_range,
            "sieve_size": row.sieve_size,
            "ct_weight_per_piece": float(row.ct_weight_per_piece),
            "picked_id": int(row.id),
        }

    return jsonify(out)



@app.route("/api/style-excel-upload", methods=["POST"])
def style_excel_upload():
    auth = require_admin_json()
    if auth:
        return auth

    f = request.files.get("file")
    clear_first = request.form.get("clear_first") == "1"

    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(f.filename or "")
    if not filename.lower().endswith((".xlsx", ".xls")):
        return jsonify({"error": "Upload only .xlsx or .xls"}), 400

    try:
        df = pd.read_excel(f)
    except Exception as e:
        return jsonify({"error": f"Excel read failed: {str(e)}"}), 400

    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["style_no", "gold_wt", "size", "pcs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}. Required: {required}"}), 400

    if clear_first:
        db.session.query(StyleNo).delete()
        db.session.commit()

    total_rows = len(df)
    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        style_no = str(row.get("style_no") or "").strip()
        if not style_no or style_no.lower() == "nan":
            skipped += 1
            continue

        try:
            gold_wt = float(row.get("gold_wt")) if pd.notna(row.get("gold_wt")) else None
        except:
            gold_wt = None

        try:
            pcs = int(row.get("pcs")) if pd.notna(row.get("pcs")) else None
        except:
            pcs = None

        size_val = str(row.get("size")).strip() if pd.notna(row.get("size")) else None

        db.session.add(StyleNo(style_no=style_no, gold_wt=gold_wt, size=size_val, pcs=pcs))
        inserted += 1

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"DB insert failed: {str(e)}"}), 500

    return jsonify({
        "message": "Imported successfully",
        "total_rows": total_rows,
        "inserted": inserted,
        "skipped": skipped
    }), 200


def require_admin_json():
    if "role" not in session or session["role"] != "admin":
        return jsonify({"error": "Unauthorized. Please login as admin."}), 401
    return None

@app.route("/api/image-crop-excel-upload", methods=["POST"])
def image_crop_excel_upload():
    auth = require_admin_json()
    if auth:
        return auth

    f = request.files.get("file")
    clear_first = request.form.get("clear_first") == "1"

    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(f.filename or "")
    if not filename.lower().endswith((".xlsx",)):
        return jsonify({"error": "Upload only .xlsx (images supported in .xlsx)"}), 400

    try:
        file_bytes = f.read()
        wb = openpyxl.load_workbook(BytesIO(file_bytes))
        ws = wb.active
    except Exception as e:
        return jsonify({"error": f"Excel open failed: {str(e)}"}), 400

    # Expected headers: style_no | image
    h1 = (ws.cell(row=1, column=1).value or "").strip().lower()
    h2 = (ws.cell(row=1, column=2).value or "").strip().lower()
    if h1 != "style_no" or h2 != "image":
        return jsonify({"error": "Header must be: style_no | image"}), 400

    if clear_first:
        db.session.query(ImageCrop).delete()
        db.session.commit()

    # Map: excel_row_number -> image_bytes
    # openpyxl stores images in ws._images with anchors
    row_to_img = {}
    for im in getattr(ws, "_images", []):
        try:
            # anchor._from.row is 0-based (row=1 means Excel row 2)
            excel_row = im.anchor._from.row + 1
            img_bytes = im._data()  # bytes of the image (jpg/png)
            row_to_img[excel_row] = img_bytes
        except:
            pass

    total_rows = ws.max_row - 1
    inserted = 0
    skipped = 0
    no_image = 0

    for r in range(2, ws.max_row + 1):
        style_no = str(ws.cell(row=r, column=1).value or "").strip()
        if not style_no:
            skipped += 1
            continue

        img_bytes = row_to_img.get(r)
        if not img_bytes:
            no_image += 1
            continue

        db.session.add(ImageCrop(style_no=style_no, image=img_bytes))
        inserted += 1

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"DB insert failed: {str(e)}"}), 500

    return jsonify({
        "message": "Image excel imported successfully",
        "total_rows": total_rows,
        "inserted": inserted,
        "skipped_empty_style": skipped,
        "rows_without_image": no_image
    }), 200

@app.route("/admin/upload-style-and-images")
def upload_style_and_images():
    if "role" not in session or session["role"] != "admin":
        return redirect("/signin")
    return render_template("upload_style_and_images.html")

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


@app.route("/", methods=["GET"])
def portal():
    return render_template("portal.html")
    


@app.route("/sales")
def sales_page():
    if "role" not in session or session["role"] != "sales":
        return redirect("/signin")
    return render_template("sales.html")

@app.route("/admin")
def admin():
    if "role" not in session or session["role"] != "admin":
        return redirect("/signin")
    return render_template("admin.html")

@app.route("/signup")
def signup():
    return render_template("signup.html")

# -----------------------------
# PAGES ROUTES (ERP MODULES)
# -----------------------------


@app.route("/create_order")
def create_order():
    clients = ClientKYC.query.order_by(ClientKYC.id.desc()).all()
    return render_template("create_order.html", clients=clients)

@app.route("/Client_kyc")
def Client_kyc():
    return render_template("Client_kyc.html")


@app.route("/production-board")
def production_board():
    orders = ProductionOrder.query.order_by(ProductionOrder.id.desc()).all()
    return render_template("production_board.html", orders=orders)



    # ---------- Upload Pdf doc to DB ----------

@app.route("/upload-doc", methods=["POST"])
def upload_doc():
    file = request.files.get("file")
    doc_type = request.form.get("doc_type")

    if not file:
        return jsonify({})

    result = process_image_ocr(file, doc_type)
    return jsonify(result)

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





@app.route("/signin", methods=["GET", "POST"])
def signin():
    selected_role = request.args.get("role")

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email, password=password).first()

        if user:
            session["user_id"] = user.id
            session["role"] = user.role

            if user.role == "admin":
                return redirect("/admin")
            elif user.role == "sales":
                return redirect("/sales")
            elif user.role == "production":
                return redirect("/production-board")

        return "Invalid Login"

    return render_template("signin.html", selected_role=selected_role)



@app.route("/logout")
def logout():
    session.clear()
    return redirect("/signin")




# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)
