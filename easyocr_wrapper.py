from my_utils.plate_enhancer import enhance_with_realesrgan
import cv2
import numpy as np
import easyocr
import re

reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_plate(plate_crop):
    try:
        enhanced = enhance_with_realesrgan(plate_crop)

        # CLAHE (optional)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # OCR
        results = reader.readtext(enhanced)
        for (_, text, conf) in results:
            clean = ''.join(filter(str.isalnum, text))
            if len(clean) >= 4 and conf >= 0.3:
                return clean.upper(), conf

    except Exception as e:
        print(f"[OCR ENHANCEMENT ERROR]: {e}")

    return 'unreadable', 0.0
