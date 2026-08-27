from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory
)
from werkzeug.utils import secure_filename

import mysql.connector
import os
import uuid
import threading
import numpy as np

from PIL import Image


app = Flask(__name__)


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IS_VERCEL = bool(os.environ.get("VERCEL"))

# Vercel hanya cocok memakai direktori sementara untuk file upload.
# Saat lokal tetap menggunakan static/uploads seperti sebelumnya.
if IS_VERCEL:
    UPLOAD_FOLDER = "/tmp/uploads"
else:
    UPLOAD_FOLDER = os.path.join(
        BASE_DIR,
        "static",
        "uploads"
    )

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_mobilenetv2_transfer.tflite"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}


# ============================================================
# KELAS MODEL
# ============================================================

CLASS_NAMES = [
    "Chickenpox",
    "Dermatitis Eksim",
    "Measles",
    "Monkeypox"
]

CLASS_TO_DB_ID = {
    "Chickenpox": 1,
    "Dermatitis Eksim": 2,
    "Measles": 3,
    "Monkeypox": 4
}


# ============================================================
# MODEL
# ============================================================

model = None
MODEL_READY = False
MODEL_ERROR = None
MODEL_LOCK = threading.Lock()


def load_mobilenet_model():

    global model
    global MODEL_READY
    global MODEL_ERROR

    if MODEL_READY and model is not None:
        return model

    try:

        print("=" * 60)
        print("MEMUAT MODEL LITERT...")

        if not os.path.exists(MODEL_PATH):

            MODEL_ERROR = (
                "File model tidak ditemukan: "
                + MODEL_PATH
            )

            print(MODEL_ERROR)

            return None

        from ai_edge_litert.interpreter import Interpreter

        model = Interpreter(
            model_path=MODEL_PATH
        )

        model.allocate_tensors()

        MODEL_READY = True
        MODEL_ERROR = None

        print("Model LiteRT berhasil dimuat!")

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        print(
            "Input shape:",
            input_details[0]["shape"]
        )

        print(
            "Input dtype:",
            input_details[0]["dtype"]
        )

        print(
            "Output shape:",
            output_details[0]["shape"]
        )

        for i, nama in enumerate(CLASS_NAMES):
            print(i, "=", nama)

        print("=" * 60)

        return model

    except Exception as error:

        MODEL_READY = False
        MODEL_ERROR = str(error)

        print("GAGAL MEMUAT MODEL:")
        print(error)

        return None


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():

    """
    LOCAL:
    otomatis menggunakan MySQL XAMPP.

    VERCEL:
    menggunakan Environment Variables.
    """

    host = os.environ.get(
        "DB_HOST",
        "localhost"
    )

    port = int(
        os.environ.get(
            "DB_PORT",
            "3306"
        )
    )

    user = os.environ.get(
        "DB_USER",
        "root"
    )

    password = os.environ.get(
        "DB_PASSWORD",
        ""
    )

    database = os.environ.get(
        "DB_NAME",
        "db_skindetect"
    )

    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        connection_timeout=10
    )


# ============================================================
# VALIDASI FILE
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename
        .rsplit(".", 1)[1]
        .lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(filepath):

    image = Image.open(
        filepath
    ).convert(
        "RGB"
    )

    image = image.resize(
        (224, 224),
        resample=Image.Resampling.NEAREST
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Preprocessing MobileNetV2:
    # nilai pixel 0..255 menjadi -1..1
    image_array = (
        image_array / 127.5
    ) - 1.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDIKSI
# ============================================================

def predict_image(filepath):

    current_model = load_mobilenet_model()

    if current_model is None:

        raise RuntimeError(
            MODEL_ERROR
            or
            "Model belum siap digunakan."
        )

    image_array = preprocess_image(
        filepath
    )

    input_details = (
        current_model
        .get_input_details()
    )

    output_details = (
        current_model
        .get_output_details()
    )

    input_dtype = input_details[0]["dtype"]

    image_array = image_array.astype(
        input_dtype
    )

    with MODEL_LOCK:

        current_model.set_tensor(
            input_details[0]["index"],
            image_array
        )

        current_model.invoke()

        prediction = current_model.get_tensor(
            output_details[0]["index"]
        )[0]

    predicted_index = int(
        np.argmax(prediction)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        prediction[predicted_index]
        * 100
    )

    print("\nHASIL PREDIKSI")
    print("=" * 45)

    for i, value in enumerate(prediction):

        print(
            f"{CLASS_NAMES[i]} : "
            f"{value * 100:.2f}%"
        )

    print(
        "Prediksi akhir :",
        predicted_class
    )

    print(
        "Confidence     :",
        f"{confidence:.2f}%"
    )

    print("=" * 45)

    return predicted_class, confidence


# ============================================================
# FILE GAMBAR
# ============================================================

@app.route(
    "/uploads/<path:filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )

def upload_to_blob(filepath, filename):

    if not IS_VERCEL:
        return None

    try:

        from vercel.blob import BlobClient

        client = BlobClient()

        with open(filepath, "rb") as f:
            file_bytes = f.read()

        blob = client.put(
            f"uploads/{filename}",
            file_bytes,
            access="public",
            add_random_suffix=True
        )

        print(
            "BLOB UPLOAD BERHASIL:",
            blob.url
        )

        return blob.url

    except Exception as error:

        print(
            "BLOB UPLOAD ERROR:",
            error
        )

        return None

# ============================================================
# DETEKSI
# ============================================================

@app.route(
    "/deteksi",
    methods=["GET", "POST"]
)
def deteksi():

    if request.method == "GET":

        return render_template(
            "upload.html"
        )

    # ========================================================
    # VALIDASI FILE
    # ========================================================

    if "image" not in request.files:

        return render_template(
            "upload.html",
            error=(
                "Silakan pilih gambar "
                "terlebih dahulu."
            )
        )

    file = request.files[
        "image"
    ]

    if file.filename == "":

        return render_template(
            "upload.html",
            error=(
                "Silakan pilih gambar "
                "terlebih dahulu."
            )
        )

    if not allowed_file(
        file.filename
    ):

        return render_template(
            "upload.html",
            error=(
                "Format gambar harus "
                "JPG, JPEG, PNG, "
                "atau WEBP."
            )
        )

    # ========================================================
    # SIMPAN GAMBAR
    # ========================================================

    extension = (
        file.filename
        .rsplit(".", 1)[1]
        .lower()
    )

    filename = secure_filename(
        f"{uuid.uuid4().hex}.{extension}"
    )

    os.makedirs(
        app.config["UPLOAD_FOLDER"],
        exist_ok=True
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        file.save(
            filepath
        )

    except Exception as error:

        print(
            "ERROR SIMPAN GAMBAR:",
            error
        )

        return render_template(
            "upload.html",
            error=(
                "Gambar tidak dapat "
                "disimpan."
            )
        )

    # ========================================================
    # PREDIKSI
    # ========================================================

    try:

        prediction, confidence = predict_image(
            filepath,
        )
        
        image_url = upload_to_blob(
            filepath,
            filename
        )

    except Exception as error:

        print(
            "ERROR SAAT PREDIKSI:",
            error
        )

        return render_template(
            "upload.html",
            error=(
                "Gambar gagal diproses "
                "oleh model."
            )
        )

    # ========================================================
    # SIMPAN DATABASE
    # ========================================================

    try:

        id_kelas = CLASS_TO_DB_ID[
            prediction
        ]

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO prediksi
            (
                id_kelas,
                nama_gambar,
                confidence,
                image_url
            )
            VALUES
            (%s, %s, %s, %s)
            """,
            (
                id_kelas,
                filename,
                round(confidence,2),
                image_url
            )
        )

        connection.commit()

        cursor.close()
        connection.close()

        print(
            "HASIL BERHASIL "
            "DISIMPAN KE DATABASE"
        )

    except mysql.connector.Error as error:

        # Prediksi tetap ditampilkan walaupun
        # database sedang tidak tersedia.
        print(
            "DATABASE ERROR:",
            error
        )

    # ========================================================
    # HASIL
    # ========================================================

    return render_template(
    "result.html",
    filename=filename,
    image_url=image_url,
    prediction=prediction,
    confidence=round(confidence, 2),
    model_ready=True
)


# ============================================================
# RIWAYAT
# ============================================================

@app.route("/riwayat")
def riwayat():

    try:

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT
                p.id_prediksi,
                p.nama_gambar,
                p.image_url,
                p.confidence,
                p.tanggal_prediksi,
                k.nama_kelas

            FROM prediksi p

            LEFT JOIN kelas_penyakit k
                ON p.id_kelas = k.id_kelas

            ORDER BY
                p.id_prediksi DESC
            """
        )

        data = cursor.fetchall()

        cursor.close()
        connection.close()

        return render_template(
            "history.html",
            data=data,
            db_error=None
        )

    except mysql.connector.Error as error:

        print(
            "ERROR RIWAYAT:",
            error
        )

        return render_template(
            "history.html",
            data=[],
            db_error=str(error)
        )


# ============================================================
# HAPUS SATU RIWAYAT
# ============================================================

@app.route(
    "/hapus/<int:id_prediksi>",
    methods=["POST"]
)
def hapus(id_prediksi):

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT nama_gambar
            FROM prediksi
            WHERE id_prediksi = %s
            """,
            (
                id_prediksi,
            )
        )

        result = cursor.fetchone()

        if result:

            nama_gambar = result[0]

            cursor.execute(
                """
                DELETE FROM prediksi
                WHERE id_prediksi = %s
                """,
                (
                    id_prediksi,
                )
            )

            connection.commit()

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                nama_gambar
            )

            if os.path.exists(
                filepath
            ):

                try:

                    os.remove(
                        filepath
                    )

                except OSError as error:

                    print(
                        "FILE TIDAK "
                        "DAPAT DIHAPUS:",
                        error
                    )

    except mysql.connector.Error as error:

        print(
            "ERROR HAPUS:",
            error
        )

    finally:

        if cursor is not None:

            try:
                cursor.close()
            except Exception:
                pass

        if connection is not None:

            try:
                connection.close()
            except Exception:
                pass

    return redirect(
        url_for(
            "riwayat"
        )
    )


# ============================================================
# HAPUS SEMUA RIWAYAT
# ============================================================

@app.route(
    "/hapus-semua",
    methods=["POST"]
)
def hapus_semua():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT nama_gambar
            FROM prediksi
            """
        )

        images = cursor.fetchall()

        cursor.execute(
            """
            DELETE FROM prediksi
            """
        )

        connection.commit()

        for image in images:

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                image[0]
            )

            if os.path.exists(
                filepath
            ):

                try:

                    os.remove(
                        filepath
                    )

                except OSError as error:

                    print(
                        "FILE TIDAK "
                        "DAPAT DIHAPUS:",
                        error
                    )

    except mysql.connector.Error as error:

        print(
            "ERROR HAPUS SEMUA:",
            error
        )

    finally:

        if cursor is not None:

            try:
                cursor.close()
            except Exception:
                pass

        if connection is not None:

            try:
                connection.close()
            except Exception:
                pass

    return redirect(
        url_for(
            "riwayat"
        )
    )


# ============================================================
# ERROR FILE TERLALU BESAR
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "upload.html",
        error=(
            "Ukuran gambar maksimal "
            "5 MB."
        )
    ), 413


# ============================================================
# RUN LOCAL
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        app.config["UPLOAD_FOLDER"],
        exist_ok=True
    )

    app.run(
        host="0.0.0.0",
        port=5051,
        debug=False
    )
