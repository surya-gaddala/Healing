from contextvars import Context
import logging
import sys
import os
from pathlib import Path
from behave import step

# Get the absolute path to the directory containing the environment.py file (features directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path to the project's root directory (one level up)
project_root = os.path.dirname(current_dir)
# Get the absolute path to the src directory
src_path = os.path.join(project_root, 'src')
# Add the src directory to the Python path
sys.path.insert(0, src_path)

from automation_ocr_utils import save_screenshot, load_config, init_browser, close_browser, PROJECT_ROOT
# Configure Behave Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[ logging.StreamHandler(sys.stdout) ] # Log to console
    # Optional: Add FileHandler for behave logs
    # handlers=[ logging.FileHandler(PROJECT_ROOT / "behave_run.log"), logging.StreamHandler(sys.stdout) ]
)
logger = logging.getLogger("behave") # Use behave's logger

def before_all(context):
    """Executes once before all features."""
    logger.info("Loading configuration...")
    try:
        context.config = load_config()
        # Make project root available if needed
        context.project_root = PROJECT_ROOT
        logger.info("Configuration loaded.")
    except Exception as e:
        logger.exception(f"CRITICAL: Failed to load configuration: {e}")
        raise # Stop execution if config fails

def before_scenario(context, scenario):
    """Executes before each scenario."""
    logger.info(f"=== Starting Scenario: {scenario.name} ===")
    try:
        # Pass config to init_browser
        context.playwright, context.browser, context.browser_context, context.page = init_browser(context.config)
    except Exception as e:
        logger.exception(f"CRITICAL: Scenario setup failed (browser init): {e}")
        raise # Stop if browser doesn't start

def after_scenario(context, scenario):
    """Executes after each scenario."""
    logger.info(f"=== Finished Scenario: {scenario.name} - Status: {scenario.status.name} ===")
    if hasattr(context, 'playwright'): # Check if browser was initialized
        close_browser(context.playwright, context.browser)
    else:
         logger.warning("Browser resources not found for cleanup in after_scenario.")

def after_step(context, step):
    """Executes after each step."""
    # Take screenshot only on failure
    if step.status == "failed":
        logger.error(f"--- Step Failed: {step.keyword} {step.name} ---")
        if hasattr(context, 'page') and context.page and not context.page.is_closed():
             step_name_safe = "".join(c if c.isalnum() else "_" for c in step.name)[:80]
             save_screenshot(context.page, context.config, f"FAILED_{context.scenario.name}_{step_name_safe}")
        else:
             logger.warning("Could not take failure screenshot - page object not available or closed.")

