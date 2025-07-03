from flask import Flask, request, jsonify, render_template
import os
import cv2
import pytesseract
from PIL import Image
import numpy as np
import logging
import re

app = Flask(__name__)


logging.basicConfig(level=logging.DEBUG)

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\volka\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


Image.MAX_IMAGE_PIXELS = 400000000  


def preprocess_image(image_path):
    try:
      
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Görüntü yüklenemedi, dosya geçersiz veya bozuk.")
        
 
        max_pixels = 10000000  
        height, width = img.shape[:2]
        if width * height > max_pixels:
            scale = (max_pixels / (width * height)) ** 0.5
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Kontrast artırımı ve gürültü azaltma
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Sonucu geçici bir dosyaya kaydet
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.png')
        cv2.imwrite(temp_path, thresh)
        return temp_path
    except Exception as e:
        logging.error(f"Ön işleme hatası: {str(e)}")
        raise

# Metni analiz etme fonksiyonu
def analyze_text(text):
    # Kelimelere ayır (sadece harf içeren kelimeler)
    words = [word for word in re.findall(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ]+\b', text) if word]
    
    # En uzun ve en kısa kelime
    longest_word = max(words, key=len) if words else "Yok"
    shortest_word = min(words, key=len) if words else "Yok"
    
    # Toplam kelime sayısı
    word_count = len(words)
    
    # Toplam karakter sayısı (boşluklar hariç)
    char_count = len(re.sub(r'\s', '', text))
    
    # Türkçe özel karakter sayısı
    turkish_chars = 'çğıöşüÇĞİÖŞÜ'
    turkish_char_count = sum(1 for char in text if char in turkish_chars)
    
    return {
        'longest_word': longest_word,
        'shortest_word': shortest_word,
        'word_count': word_count,
        'char_count': char_count,
        'turkish_char_count': turkish_char_count
    }

# OCR fonksiyonu
def ocr_image(image_path, language='tur', text_type='handwritten'):
    try:
        processed_image = preprocess_image(image_path)
        psm = 6 if text_type == 'handwritten' else 3
        custom_config = f'--psm {psm}'
        text = pytesseract.image_to_string(
            Image.open(processed_image),
            lang=language,
            config=custom_config
        )
        return text.strip()
    except Exception as e:
        logging.error(f"OCR hatası: {str(e)}")
        return f"OCR hatası: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'GET':
        return jsonify({'error': 'Bu endpoint sadece POST isteklerini destekler'}), 405
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Resim dosyası bulunamadı'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Dosya seçilmedi'}), 400
        
        language = request.form.get('language', 'tur')
        text_type = request.form.get('text-type', 'handwritten')
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        text = ocr_image(file_path, language, text_type)
        
        # Metni analiz etme
        analysis = analyze_text(text)
        
        try:
            os.remove(file_path)
        except Exception as e:
            logging.warning(f"Dosya silme hatası: {str(e)}")
        
        return jsonify({
            'text': text,
            'longest_word': analysis['longest_word'],
            'shortest_word': analysis['shortest_word'],
            'word_count': analysis['word_count'],
            'char_count': analysis['char_count'],
            'turkish_char_count': analysis['turkish_char_count']
        })
    except Exception as e:
        logging.error(f"Upload endpoint hatası: {str(e)}")
        return jsonify({'error': f"Sunucu hatası: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)