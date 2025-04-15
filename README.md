# OCR Form Automation Project

This project automates form filling on TutorialsNinja using Tesseract OCR to detect field labels and buttons.

## Setup
1. Install Python 3.10.
2. Install Tesseract:
   - Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
   - Install to `C:\Program Files\Tesseract-OCR` and add to PATH.

cd C:\Users\<NewUser>\Desktop\ocr_automation
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
playwright install
del /S *.pyc
behave