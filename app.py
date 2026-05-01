import os
import cv2
import numpy as np
import streamlit as st
import ctypes

from deeplearning import object_detection

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Lagrange Watermark System",
    layout="wide"
)

st.title("License Plate Security System")
st.caption("Lagrange-based watermark encoding and verification")
st.divider()

# =========================
# LOAD C++ LIB (SAFE WINDOWS + LINUX)
# =========================
BASE_DIR = os.path.dirname(__file__)

DLL_PATH = os.path.join(BASE_DIR, "cpp_module", "liblagrange.dll")
SO_PATH  = os.path.join(BASE_DIR, "cpp_module", "liblagrange.so")

cpp_lib = None
use_cpp = False

try:
    if os.path.exists(SO_PATH):
        cpp_lib = ctypes.CDLL(SO_PATH)
        use_cpp = True
    elif os.path.exists(DLL_PATH):
        cpp_lib = ctypes.CDLL(DLL_PATH)
        use_cpp = True
except Exception:
    use_cpp = False

if use_cpp:
    cpp_func = cpp_lib.compute_signature
    cpp_func.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double)
    ]
    cpp_func.restype = ctypes.c_double

# =========================
# CHAR ↔ NUMBER
# =========================
def char_to_num(c):
    if c.isdigit():
        return int(c)
    if c.isalpha():
        return ord(c.upper()) - ord('A') + 10
    return None

def num_to_char(n):
    return str(n) if n < 10 else chr(n - 10 + ord('A'))

def clean_plate(text):
    text = text.upper()
    return "".join([c for c in text if char_to_num(c) is not None])

def encode_plate(text):
    text = clean_plate(text)
    return [char_to_num(c) for c in text]

# =========================
# PYTHON FALLBACK LAGRANGE
# =========================
def build_signature_python(nums):
    x = np.arange(len(nums))
    y = np.array(nums, dtype=float)

    def lagrange(t):
        total = 0
        for i in range(len(x)):
            term = y[i]
            for j in range(len(x)):
                if i != j:
                    term *= (t - x[j]) / (x[i] - x[j])
            total += term
        return total

    xs = np.linspace(0, len(nums)-1, 60)
    ys = np.array([lagrange(v) for v in xs])

    ys = ys - np.mean(ys)
    ys = ys / (np.std(ys) + 1e-9)
    ys = ys / (np.linalg.norm(ys) + 1e-9)

    return ys

# =========================
# C++ SIGNATURE WRAPPER
# =========================
def build_signature(nums):
    if not use_cpp:
        return build_signature_python(nums)

    n = len(nums)

    arr = (ctypes.c_double * n)(*nums)
    out = (ctypes.c_double * 60)()

    cpp_func(arr, n, out)

    return np.array(out)

# =========================
# WATERMARK EMBED
# =========================
def embed_watermark(img, sig):
    h, w = img.shape[:2]

    wm_w, wm_h = 220, 120
    x0 = w - wm_w - 10
    y0 = h - wm_h - 10

    cv2.rectangle(img, (x0, y0), (x0 + wm_w, y0 + wm_h), (255, 255, 255), -1)

    sig = np.interp(
        np.linspace(0, len(sig) - 1, wm_w),
        np.arange(len(sig)),
        sig
    )

    sig = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-9)

    for i in range(wm_w):
        val = int(sig[i] * wm_h)
        cv2.line(img,
                 (x0 + i, y0 + wm_h),
                 (x0 + i, y0 + wm_h - val),
                 (0, 0, 0),
                 1)

    return img

# =========================
# BLUR PLATE
# =========================
def blur_plate(image, bbox):
    x, y, w, h = bbox

    roi = image[y:y+h, x:x+w]

    if roi.size == 0:
        return image

    blurred = cv2.GaussianBlur(roi, (101, 101), 0)

    small = cv2.resize(blurred, (20, 20))
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    image[y:y+h, x:x+w] = pixelated

    return image

# =========================
# SIGNATURE EXTRACTION
# =========================
def extract_signature(img):
    h, w = img.shape[:2]

    wm_w, wm_h = 220, 120
    x0 = w - wm_w - 10
    y0 = h - wm_h - 10

    roi = img[y0:y0+wm_h, x0:x0+wm_w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    signal = 255 - np.mean(gray, axis=0)

    signal = signal - np.mean(signal)
    signal = signal / (np.std(signal) + 1e-9)

    signal = np.interp(
        np.linspace(0, len(signal) - 1, 60),
        np.arange(len(signal)),
        signal
    )

    signal = signal / (np.linalg.norm(signal) + 1e-9)

    return signal

# =========================
# COMPARISON
# =========================
def compare(a, b):
    a = np.array(a)
    b = np.array(b)

    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)

    return abs(np.dot(a, b))

# =========================
# SAVE IMAGE
# =========================
def save_image(img):
    os.makedirs("output", exist_ok=True)
    path = "output/result.jpg"
    cv2.imwrite(path, img)
    return path

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Workflow")

    st.markdown("""
### Encoding
1. Upload image   
2. Detect plate  
3. Generate watermark  
4. Download image  

### Decoding
1. Upload image  
2. Enter key  
3. Verify watermark  
""")

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["Encoding", "Decoding"])

# =========================
# ENCODING
# =========================
with tab1:
    file = st.file_uploader("Upload Car Image")

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Input Image", use_container_width=True)

        path = "temp.jpg"
        cv2.imwrite(path, img)

        result = object_detection(path, "result.jpg")
        text_list, boxes = result if isinstance(result, tuple) else (result, [])

        if not text_list or text_list[0] == "":
            st.error("No plate detected or OCR failed")
            st.write("Debug boxes:", boxes)
            st.stop()

        text = clean_plate(text_list[0])
        nums = encode_plate(text)

        sig = build_signature(nums)

        img = embed_watermark(img, sig)
        img = blur_plate(img, boxes[0] if boxes else (0,0,img.shape[1],img.shape[0]))

        with col2:
            st.image(img, caption="Processed Image", use_container_width=True)

        st.success("Detected Plate")
        st.code(text)

        st.markdown("Encoded Key")
        st.code(",".join(map(str, nums)))

        st.download_button(
            "Download Image",
            data=cv2.imencode(".jpg", img)[1].tobytes(),
            file_name="plate.jpg"
        )

# =========================
# DECODING
# =========================
with tab2:
    file = st.file_uploader("Upload Watermarked Image")
    key = st.text_input("Enter Key")

    if file and key:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

        nums = list(map(int, key.split(",")))

        sig1 = build_signature(nums)
        sig2 = extract_signature(img)

        score = compare(sig1, sig2)

        if score > 0.85:
            plate = "".join(num_to_char(n) for n in nums)
            st.success(f"VALID PLATE: {plate}")
        else:
            st.error("Invalid watermark")
