License Plate Security System

This project is a hybrid system that secures vehicle license plates using Lagrange interpolation-based watermarking.

The system detects a license plate from an image, converts it into a numeric key, generates a mathematical signature, and embeds it as a watermark. During verification, the signature is reconstructed and compared to confirm authenticity.

Technologies
Python (Streamlit, OpenCV, NumPy)
C++ (Lagrange computation)
ctypes (Python–C++ integration)

How it works
1. Detect license plate from image
2. Convert characters (A–Z, 0–9) into numbers
3. Generate Lagrange-based signature
4. Embed signature as watermark in image
5. Blur the plate for security
6. Verify by comparing signatures

Usage

Encoding: Upload image → get key → download protected image
Decoding: Upload image + enter key → system verifies plate

Note

Works best with clear images. Heavy compression may affect accuracy.
