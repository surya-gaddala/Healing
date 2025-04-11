import configparser
import logging
import csv
import re
import difflib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from playwright.sync_api import sync_playwright

# OCR and Image Processing Libs
import cv2
import numpy as np
import pytesseract
import easyocr

# Browser Libs
from playwright.sync_api import sync_playwright, Page, Playwright, Browser, BrowserContext, Error as PlaywrightError

# Configure logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / 'config' / 'config.ini'

# --- Configuration ---

def load_config() -> configparser.ConfigParser:
    """Loads config and sets Tesseract path."""
    if not CONFIG_FILE.exists(): raise FileNotFoundError(f"Config not found: {CONFIG_FILE}")
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    tesseract_path_str = config.get('OCR', 'tesseract_path', fallback=None)
    tesseract_found = False
    if tesseract_path_str:
        tesseract_exe = Path(tesseract_path_str)
        if tesseract_exe.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_exe)
            logger.info(f"Using Tesseract from config: {tesseract_exe}")
            tesseract_found = True
        else:
            logger.warning(f"Tesseract path from config not found: {tesseract_exe}.")
    if not tesseract_found: # Check PATH if not found via config
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract found in system PATH.")
            tesseract_found = True
        except pytesseract.TesseractNotFoundError:
             pass # Logged later if actually needed by engine choice
    if not tesseract_found and config.get('OCR', 'ocr_engine', fallback='combined').lower() in ['tesseract', 'combined']:
         logger.error("Tesseract selected as OCR engine but not found. Check PATH or config.ini.")
         raise RuntimeError("Tesseract not found.")
    return config

# --- Browser Functions ---
# (init_browser, close_browser, save_screenshot - Keep these similar to previous versions)

def init_browser(config) -> tuple[Playwright, Browser, BrowserContext, Page]:
    """Initialize Playwright browser instance."""
    headless = config.getboolean('Playwright', 'headless', fallback=False)
    try:
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(no_viewport=True)
        page = context.new_page()
        if not headless:
             try: page.evaluate("() => { window.moveTo(0, 0); window.resizeTo(screen.width, screen.height); }")
             except PlaywrightError as e: logger.warning(f"Could not maximize window via JS: {e}")
        logger.info(f"Browser initialized (headless={headless})")
        return playwright, browser, context, page
    except Exception as e: logger.error(f"Browser init failed: {e}"); raise

# def init_browser(config):
#     headless = config.getboolean('Playwright', 'headless', fallback=False)
#     browser_name = config.get('Playwright', 'browser', fallback='chromium')

#     p = sync_playwright().start()

#     if browser_name == 'chromium':
#         browser = p.chromium.launch(headless=headless)
#     elif browser_name == 'firefox':
#         browser = p.firefox.launch(headless=headless)
#     elif browser_name == 'webkit':
#         browser = p.webkit.launch(headless=headless)
#     else:
#         raise ValueError(f"Unsupported browser: {browser_name}")

#     context = browser.new_context()
#     page = context.new_page()

#     # Add this to set a large viewport size
#     page.set_viewport_size({'width': 1920, 'height': 1080}) # Or any desired large resolution

#     return p, browser, context, page

def close_browser(playwright: Playwright | None, browser: Browser | None):
    """Safely close Playwright resources."""
    logger.info("Closing browser resources...")
    try:
        if browser and browser.is_connected(): browser.close()
        if playwright: playwright.stop()
        logger.info("Browser closed.")
    except Exception as e: logger.error(f"Error closing browser: {e}")

def save_screenshot(page: Page, config, name_prefix: str) -> Optional[Path]:
    """Saves a full-page screenshot."""
    screenshot_dir_str = config.get('General', 'screenshot_dir', fallback='screenshots')
    screenshot_dir = PROJECT_ROOT / screenshot_dir_str
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = screenshot_dir / f"{name_prefix}_{timestamp}.png"
    try: page.screenshot(path=path, full_page=True); logger.info(f"Screenshot saved: {path}"); return path
    except Exception as e: logger.error(f"Failed to save screenshot {path}: {e}"); return None

# --- Image Preprocessing ---
# (preprocess_image - Keep similar to previous OCR detection version)

def preprocess_image(config, image_path: Path) -> Optional[np.ndarray]:
    """Loads and preprocesses the image for OCR based on config."""
    if not image_path or not image_path.exists(): logger.error(f"Screenshot path invalid: {image_path}"); return None
    try:
        img = cv2.imread(str(image_path)); assert img is not None, "Failed to load image"
        logger.debug(f"Preprocessing image: {image_path}"); processed_img = img.copy()
        # Cropping
        if config.getboolean('Preprocessing', 'crop_enabled', fallback=False):
            h, w = processed_img.shape[:2]; x = int(w * max(0.0, config.getfloat('Preprocessing', 'crop_x', fallback=0.0)))
            y = int(h * max(0.0, config.getfloat('Preprocessing', 'crop_y', fallback=0.0))); cw = int(w * min(1.0, config.getfloat('Preprocessing', 'crop_width', fallback=1.0)))
            ch = int(h * min(1.0, config.getfloat('Preprocessing', 'crop_height', fallback=1.0))); end_x = min(w, x + cw); end_y = min(h, y + ch)
            if end_y > y and end_x > x: processed_img = processed_img[y:end_y, x:end_x]; logger.debug(f"Cropped image to: x={x}, y={y}, w={end_x-x}, h={end_y-y}")
            else: logger.warning("Invalid crop dimensions, skipping crop.")
        # Grayscale
        if config.getboolean('Preprocessing', 'grayscale', fallback=True):
            if len(processed_img.shape) == 3: processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY); logger.debug("Converted image to grayscale.")
        # Add other preprocessing steps here if needed
        return processed_img
    except Exception as e: logger.exception(f"Error during image preprocessing: {e}"); return None

# --- OCR Execution ---
# (initialize_easyocr, run_tesseract, run_easyocr, run_ocr - Keep similar)

easyocr_reader = None
def initialize_easyocr(config):
    global easyocr_reader;
    if easyocr_reader is None:
        try: use_gpu = config.getboolean('OCR', 'use_gpu', fallback=False); easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu); logger.info(f"EasyOCR Reader initialized (GPU: {use_gpu})")
        except Exception as e: logger.error(f"Failed to initialize EasyOCR Reader: {e}."); easyocr_reader = "init_failed"

def run_tesseract(image: np.ndarray, config) -> List[Dict]:
    results = []; min_conf_perc = config.getfloat('OCR', 'confidence_threshold', fallback=0.4) * 100
    try: data = pytesseract.image_to_data(image, config=r'--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError: logger.error("Tesseract not found."); raise
    except Exception as e: logger.error(f"Tesseract execution error: {e}"); return results
    for i in range(len(data['level'])):
        text = data['text'][i].strip(); conf = int(data['conf'][i])
        if conf >= min_conf_perc and text:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            if w > 5 and h > 5:
                results.append({'Label': text, 'X': x, 'Y': y, 'Width': w, 'Height': h, 'Confidence': round(conf / 100.0, 4), 'Engine': 'Tesseract'})
    logger.debug(f"Tesseract found {len(results)} results meeting confidence.")
    return results

def run_easyocr(image: np.ndarray, config) -> List[Dict]:
    global easyocr_reader;
    if easyocr_reader is None: initialize_easyocr(config)
    if easyocr_reader == "init_failed": return []
    results = []; min_conf = config.getfloat('OCR', 'confidence_threshold', fallback=0.4)
    try: ocr_output = easyocr_reader.readtext(image, detail=1, paragraph=False)
    except Exception as e: logger.error(f"EasyOCR execution error: {e}"); return results
    for (bbox, text, conf) in ocr_output:
         text = text.strip()
         if conf >= min_conf and text:
             x_coords = [int(point[0]) for point in bbox]; y_coords = [int(point[1]) for point in bbox]
             x, y = min(x_coords), min(y_coords); w = max(x_coords) - x; h = max(y_coords) - y
             if w > 5 and h > 5: results.append({'Label': text, 'X': x, 'Y': y, 'Width': w, 'Height': h, 'Confidence': round(conf, 4), 'Engine': 'EasyOCR'})
    logger.debug(f"EasyOCR found {len(results)} results meeting confidence.")
    return results

def run_ocr(config, processed_image: np.ndarray) -> List[Dict]:
    if processed_image is None: return []
    engine = config.get('OCR', 'ocr_engine', fallback='combined').lower(); all_results = []
    img_for_tesseract = processed_image if len(processed_image.shape) == 2 else cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    img_for_easyocr = processed_image if len(processed_image.shape) == 3 else cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    if engine in ['tesseract', 'combined']: all_results.extend(run_tesseract(img_for_tesseract, config))
    if engine in ['easyocr', 'combined']: all_results.extend(run_easyocr(img_for_easyocr, config))
    # Add deduplication for 'combined' if needed
    all_results.sort(key=lambda item: (item['Y'], item['X'])) # Sort for consistency
    logger.info(f"OCR run complete. Found {len(all_results)} total results from engine(s): '{engine}'.")
    return all_results

# --- Coordinate/Label Handling ---

def save_coordinates_csv(config, ocr_results: List[Dict]):
    """Saves the detected coordinates to the runtime CSV file."""
    output_csv_str = config.get('General', 'runtime_ocr_csv', fallback='detected_coordinates.csv')
    output_path = PROJECT_ROOT / output_csv_str
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Label', 'X', 'Y', 'Width', 'Height', 'Confidence', 'Engine', 'Timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader(); timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for result in ocr_results: result['Timestamp'] = timestamp; writer.writerow(result)
        logger.info(f"Runtime OCR coordinates saved to: {output_path}")
    except Exception as e: logger.error(f"Failed to write runtime CSV {output_path}: {e}")

def load_coordinates_csv(config) -> List[Dict]:
    """Loads coordinates from the runtime CSV file."""
    output_csv_str = config.get('General', 'runtime_ocr_csv', fallback='detected_coordinates.csv')
    input_path = PROJECT_ROOT / output_csv_str
    if not input_path.exists(): logger.warning(f"Runtime OCR CSV not found: {input_path}"); return []
    results = []
    try:
        with open(input_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames or []
            required_fields = ['Label', 'X', 'Y', 'Width', 'Height', 'Confidence']
            if not all(f in fieldnames for f in required_fields):
                 logger.error(f"CSV file {input_path} missing required columns. Expected: {required_fields}")
                 return []
            for row in reader:
                 try: # Add type conversion and validation
                      results.append({
                          'Label': row['Label'],
                          'X': int(float(row['X'])), 'Y': int(float(row['Y'])),
                          'Width': int(float(row['Width'])), 'Height': int(float(row['Height'])),
                          'Confidence': float(row['Confidence']),
                          'Engine': row.get('Engine', '') # Optional field
                      })
                 except (ValueError, KeyError) as conv_err:
                      logger.warning(f"Skipping row due to conversion error: {row} - {conv_err}")
        logger.debug(f"Loaded {len(results)} coordinates from {input_path}")
        return results
    except Exception as e: logger.error(f"Failed to load runtime CSV {input_path}: {e}"); return []

def normalize_text(text: str) -> str:
    """Normalize text for matching (lower, alphanumeric)."""
    return re.sub(r'[^a-z0-9]', '', text.lower().strip())

def find_element_by_label(config, target_label: str, ocr_results: List[Dict]) -> Optional[Dict]:
    """Find the best match for a label in OCR results using similarity."""
    if not ocr_results: return None
    norm_target = normalize_text(target_label)
    best_match = None
    best_score = 0.0
    match_threshold = config.getfloat('Matching', 'label_match_threshold', fallback=0.6)

    for result in ocr_results:
        norm_current = normalize_text(result['Label'])
        if not norm_current: continue # Skip empty labels after normalization

        # Use SequenceMatcher for similarity score
        similarity = difflib.SequenceMatcher(None, norm_target, norm_current).ratio()

        # Optional: Boost score slightly if one contains the other?
        # if norm_target in norm_current or norm_current in norm_target:
        #     similarity = min(1.0, similarity + 0.1) # Small boost

        if similarity > best_score:
            best_score = similarity
            best_match = result
            logger.debug(f"New best match candidate for '{target_label}': '{result['Label']}' score={similarity:.2f}")

    if best_match and best_score >= match_threshold:
        logger.info(f"Found best match for '{target_label}': '{best_match['Label']}' (Score: {best_score:.2f} >= Threshold: {match_threshold:.2f})")
        return best_match
    else:
        logger.warning(f"No suitable match found for '{target_label}'. Best score: {best_score:.2f} (Threshold: {match_threshold:.2f})")
        return None

def calculate_click_coordinates(bbox: dict, element_type: str = 'button') -> Tuple[float, float]:
    """Calculate center click point. Maybe adjust logic for 'field' later."""
    center_x = bbox['X'] + bbox['Width'] / 2
    center_y = bbox['Y'] + bbox['Height'] / 2
    # Simple center click for now for both buttons and assumed field labels
    # Add offset logic if needed, e.g., clicking below a field label
    # if element_type == 'field': center_y += bbox['Height'] # Example: click below label
    return center_x, center_y

# --- Coordinate-Based Actions ---
# (click_coordinates, fill_coordinates - Keep similar to previous versions)

def click_coordinates(page: Page, bbox: dict):
    if not all(k in bbox for k in ['X', 'Y', 'Width', 'Height']): raise ValueError("Invalid bbox for click.")
    center_x, center_y = calculate_click_coordinates(bbox, 'button') # Assume button-like center click
    try: logger.info(f"Clicking element via coordinates: x={center_x:.0f}, y={center_y:.0f}"); page.mouse.click(center_x, center_y, delay=50); page.wait_for_timeout(500)
    except Exception as e: logger.error(f"Failed to click coordinates ({center_x:.0f}, {center_y:.0f}): {e}"); raise

def fill_coordinates(page: Page, bbox: dict, value: str):
    if not all(k in bbox for k in ['X', 'Y', 'Width', 'Height']): raise ValueError("Invalid bbox for fill.")
    # Assume clicking the *label's* coordinates focuses the associated field nearby
    # More advanced logic might find the label, then search nearby for an input field's coordinates
    center_x, center_y = calculate_click_coordinates(bbox, 'field') # Assume field-like center click for focus
    try:
        logger.info(f"Filling element via coordinates: x={center_x:.0f}, y={center_y:.0f}")
        page.mouse.click(center_x, center_y, delay=50); page.wait_for_timeout(100)
        page.keyboard.press("Control+A"); page.keyboard.press("Delete"); page.wait_for_timeout(50)
        page.keyboard.type(str(value), delay=50); page.wait_for_timeout(500)
    except Exception as e: logger.error(f"Failed to fill coordinates ({center_x:.0f}, {center_y:.0f}): {e}"); raise