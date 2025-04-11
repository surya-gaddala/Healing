import logging
from behave import *
from playwright.sync_api import expect # Using playwright's assertions
from behave import given  # You might want to use @when instead of @given for this step
import time

# Import helper functions
# Assuming environment.py added 'src' to sys.path
from automation_ocr_utils import (
    save_screenshot, preprocess_image, run_ocr, save_coordinates_csv,
    load_coordinates_csv, find_element_by_label,
    click_coordinates, fill_coordinates
)

logger = logging.getLogger(__name__)

# Use type hints for context for better IDE support
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from behave.runner import Context
    from configparser import ConfigParser
    from playwright.sync_api import Page

@given('I navigate to "{url}" and perform OCR')
def step_navigate_and_ocr(context: 'Context', url: str):
    """Navigates and performs the initial OCR scan."""
    page: 'Page' = context.page
    config: 'ConfigParser' = context.config
    logger.info(f"Navigating to: {url}")
    timeout = config.getint('Playwright', 'navigation_timeout', fallback=60000)
    try:
        page.goto(url, timeout=timeout, wait_until='networkidle')
        logger.info("Navigation complete. Performing initial OCR scan...")

        # --- OCR Process ---
        screenshot_path = save_screenshot(page, config, "initial_scan")
        if not screenshot_path: raise Exception("Failed to save screenshot for OCR.")

        processed_image = preprocess_image(config, screenshot_path)
        if processed_image is None: raise Exception("Image preprocessing failed.")

        ocr_results = run_ocr(config, processed_image)
        if not ocr_results: logger.warning("Initial OCR scan found no results.")

        save_coordinates_csv(config, ocr_results) # Save/overwrite the runtime CSV
        # --- End OCR Process ---

        # Optional: Store results in context if needed immediately by next steps,
        # but loading from CSV in action steps is generally more robust.
        # context.current_ocr_results = ocr_results

    except Exception as e:
        logger.exception(f"Failed during navigation or initial OCR for {url}: {e}")
        raise # Fail the step


@when('I enter "{value}" using label "{target_label}"')
def step_enter_text_by_label(context: 'Context', value: str, target_label: str):
    """Finds element by label in last OCR results and fills using coordinates."""
    page: 'Page' = context.page
    config: 'ConfigParser' = context.config
    logger.info(f"Attempting to enter value into field labeled '{target_label}'")
    try:
        # Load the latest OCR results
        ocr_results = load_coordinates_csv(config)
        if not ocr_results: raise Exception("No OCR results loaded from CSV to find label.")

        # Find the element corresponding to the label
        target_element = find_element_by_label(config, target_label, ocr_results)
        if not target_element:
            save_screenshot(page, config, f"LABEL_NOT_FOUND_{target_label}") # Save screenshot for debugging
            raise Exception(f"Label '{target_label}' not found in OCR results with sufficient confidence.")

        # Perform the fill action using the found coordinates
        fill_coordinates(page, target_element, value)
        logger.info(f"Successfully entered value for label '{target_label}'.")
        # Optional: Screenshot after action
        # save_screenshot(page, config, f"filled_{target_label}")

    except Exception as e:
        logger.exception(f"Failed to enter '{value}' for label '{target_label}': {e}")
        raise # Fail the step


@when('I click using label "{target_label}"')
@step('I click using label "{target_label}"') # Allow using And/When etc.
def step_click_by_label(context: 'Context', target_label: str):
    """Finds element by label in last OCR results and clicks using coordinates."""
    page: 'Page' = context.page
    config: 'ConfigParser' = context.config
    logger.info(f"Attempting to click element labeled '{target_label}'")
    try:
        # Load the latest OCR results
        ocr_results = load_coordinates_csv(config)
        if not ocr_results: raise Exception("No OCR results loaded from CSV to find label.")

        # Find the element corresponding to the label
        target_element = find_element_by_label(config, target_label, ocr_results)
        if not target_element:
            save_screenshot(page, config, f"LABEL_NOT_FOUND_{target_label}") # Save screenshot for debugging
            raise Exception(f"Label '{target_label}' not found in OCR results with sufficient confidence.")

        # Perform the click action using the found coordinates
        click_coordinates(page, target_element)
        logger.info(f"Successfully clicked label '{target_label}'.")
        page.wait_for_load_state('networkidle', timeout=10000) # Wait for potential page change
        # Optional: Screenshot after action
        # save_screenshot(page, config, f"clicked_{target_label}")

        # --- Optional: Re-run OCR after click if state changes significantly ---
        # logger.info("Performing OCR scan after click...")
        # screenshot_path = save_screenshot(page, config, f"after_click_{target_label}")
        # if screenshot_path:
        #     processed_image = preprocess_image(config, screenshot_path)
        #     if processed_image is not None:
        #         ocr_results = run_ocr(config, processed_image)
        #         save_coordinates_csv(config, ocr_results) # Update CSV
        # --- End Optional Re-run OCR ---

    except Exception as e:
        logger.exception(f"Failed to click label '{target_label}': {e}")
        raise # Fail the step


@then('I see the page title as "{expected_title}"')
def step_verify_title(context: 'Context', expected_title: str):
    """Verifies the page title."""
    page: 'Page' = context.page
    logger.info(f"Verifying page title is '{expected_title}'.")
    try:
        # Using Playwright's assertion library 'expect'
        expect(page).to_have_title(expected_title, timeout=5000)
        logger.info(f"Page title verification successful.")
    except Exception as e:
        actual_title = page.title()
        logger.error(f"Page title verification FAILED. Expected: '{expected_title}', Actual: '{actual_title}'")
        logger.exception(e)
        raise # Fail the step

# Add your step definitions here
@when('I wait for {seconds:d} seconds')
def step_wait_for_seconds(context: 'Context', seconds: int):
    """Waits for the specified number of seconds."""
    time.sleep(seconds)

@when(u'I see the label "{target_label}"')
def step_impl(context, target_label):
    """Checks if the specified label is found on the page using OCR results."""
    ocr_results = load_coordinates_csv(context.custom_config)
    found_element = find_element_by_label(context.custom_config, target_label, ocr_results)
    assert found_element is not None, f"Label '{target_label}' was not found on the page."