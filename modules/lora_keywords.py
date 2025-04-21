import os
import json
import requests
import time
import threading
from queue import Queue

import gradio as gr
from modules.hash_cache import sha256_from_cache
from modules.util import get_file_from_folder_list
import modules.config as config

# Constants and globals
CIVITAI_API_BASE = "https://civitai.com/api/v1"
request_queue = Queue()
request_times = []
request_thread = None
is_running = False

# Simple rate-limited API request processor
def process_requests():
    global is_running, request_times
    while is_running:
        if not request_queue.empty():
            # Rate limiting - clean old request timestamps
            now = time.time()
            request_times = [t for t in request_times if now - t < 60]
            
            # Wait if we've hit rate limit
            if len(request_times) >= 10:  # 10 requests per minute
                time.sleep(max(0, 60 - (now - request_times[0])))
            
            # Process request
            func, args, result_queue = request_queue.get()
            try:
                request_times.append(time.time())
                result = func(*args)
                result_queue.put((True, result))
            except Exception as e:
                result_queue.put((False, str(e)))
            finally:
                request_queue.task_done()
        else:
            time.sleep(0.1)

def queue_api_request(func, *args):
    global request_thread, is_running
    result_queue = Queue()
    request_queue.put((func, args, result_queue))
    
    # Start thread if not running
    if not is_running or not request_thread or not request_thread.is_alive():
        is_running = True
        request_thread = threading.Thread(target=process_requests, daemon=True)
        request_thread.start()
    
    return result_queue

# Metadata functions
def get_metadata_path(lora_path):
    return f"{lora_path}.civitai.json"

def load_metadata(lora_path):
    try:
        metadata_path = get_metadata_path(lora_path)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
    return None

def save_metadata(lora_path, metadata):
    try:
        metadata_path = get_metadata_path(lora_path)
        os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False

# CivitAI API functions
def get_civitai_api_key():
    return getattr(config, "civitai_api_key", "") or ""

def fetch_from_civitai(hash_value):
    url = f"{CIVITAI_API_BASE}/model-versions/by-hash/{hash_value}"
    headers = {}
    api_key = get_civitai_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None

def get_model_by_hash(hash_value):
    result_queue = queue_api_request(fetch_from_civitai, hash_value)
    try:
        success, result = result_queue.get(timeout=30)
        return result if success else None
    except:
        return None

def get_lora_metadata(lora_filename, refresh=False):
    # Skip if None
    if not lora_filename or lora_filename == "None":
        return None
    
    # Get LoRA file path
    if os.path.exists(lora_filename):
        lora_path = lora_filename
    else:
        lora_path = get_file_from_folder_list(lora_filename, config.paths_loras)
    
    if not lora_path or not os.path.exists(lora_path):
        return None
    
    # Use cached metadata if available
    if not refresh:
        metadata = load_metadata(lora_path)
        if metadata:
            return metadata
    
    # Fetch from API
    hash_value = sha256_from_cache(lora_path)
    if not hash_value:
        return None
    
    api_data = get_model_by_hash(hash_value)
    if not api_data:
        return None
    
    # Create metadata
    metadata = {
        "hash": hash_value,
        "name": api_data.get("name", "Unknown"),
        "model_id": api_data.get("modelId"),
        "keywords": [],
        "trigger_words": api_data.get("trainedWords", []),
        "base_model": api_data.get("baseModel", "Unknown"),
    }
    
    # Add tags as keywords
    if "model" in api_data and "tags" in api_data["model"]:
        metadata["keywords"] = api_data["model"]["tags"]
    
    # Add trigger words
    for word in metadata["trigger_words"]:
        if word not in metadata["keywords"]:
            metadata["keywords"].append(word)
    
    # Add preview image
    if "images" in api_data and api_data["images"]:
        metadata["preview_image"] = api_data["images"][0].get("url", "")
    
    save_metadata(lora_path, metadata)
    return metadata

# UI components
def generate_keywords_html(keywords, selected=None):
    if not keywords:
        return ""
    
    selected = selected or []
    html = '<div class="lora-keywords-container">'
    
    for kw in keywords:
        cls = " selected" if kw in selected else ""
        html += f'<div class="keyword-tag{cls}" data-keyword="{kw}" onclick="toggleKeyword(this)">{kw}</div>'
    
    html += '</div>'
    return html

def update_keywords_display(lora_filename, enabled):
    # Always show the UI container, but display different content based on state
    if not enabled or lora_filename == "None":
        # Even when disabled, show a message rather than hiding completely
        message = "<div class='keywords-info-message'>Select a LoRA and enable it to see keywords</div>"
        return gr.update(visible=True), gr.update(value=message), gr.update(value="")
    
    # Get metadata for the selected LoRA
    try:
        metadata = get_lora_metadata(lora_filename)
        
        # If no keywords found, show a helpful message instead of hiding
        if not metadata or not metadata.get("keywords") or len(metadata.get("keywords", [])) == 0:
            print(f"No keywords found for LoRA: {lora_filename}")
            message = f"<div class='keywords-info-message'>No keywords found for LoRA: {lora_filename}</div>"
            message += "<div>Checking CivitAI for metadata...</div>"
            
            # Try to fetch metadata in the background
            threading.Thread(target=lambda: get_lora_metadata(lora_filename, refresh=True), daemon=True).start()
            
            return gr.update(visible=True), gr.update(value=message), gr.update(value="")
        
        # Generate HTML for keywords display
        keywords = metadata.get("keywords", [])
        html = generate_keywords_html(keywords)
        print(f"Found {len(keywords)} keywords for LoRA: {lora_filename}")
        
        # Make visible and return the HTML
        return gr.update(visible=True), gr.update(value=html), gr.update(value="")
    except Exception as e:
        print(f"Error displaying keywords for {lora_filename}: {e}")
        error_message = f"<div class='keywords-error-message'>Error retrieving keywords: {str(e)}</div>"
        return gr.update(visible=True), gr.update(value=error_message), gr.update(value="")

def create_keywords_ui(gr_lib, lora_dropdown, lora_enabled):
    # Create a container for the keywords UI - set visible by default to ensure it's rendered
    with gr_lib.Column(elem_classes=["keywords-ui-container"], visible=True) as keywords_column:
        # Add a clear header to make it stand out
        gr_lib.Markdown("### LoRA Keywords", elem_classes=["keywords-header"])
        
        # Make elements more visible with improved styling
        keywords_html = gr_lib.HTML(
            label="Keywords - Click to select",
            elem_id=f"lora_keywords_{lora_dropdown.elem_id}",
            elem_classes=["keywords-display"]
        )
        selected_keywords = gr_lib.Textbox(
            label="Selected Keywords",
            elem_id=f"lora_selected_{lora_dropdown.elem_id}",
            elem_classes=["selected-keywords"],
            lines=2  # Make the textbox taller for better visibility
        )
        
        with gr_lib.Row():
            copy_button = gr_lib.Button(
                "Copy to Clipboard",
                elem_id=f"copy_keywords_{lora_dropdown.elem_id}",
                elem_classes=["copy-keywords-button", "small-button"],
                variant="primary"  # Make the button more visible
            )
            add_button = gr_lib.Button(
                "Add to Prompt",
                elem_id=f"add_keywords_{lora_dropdown.elem_id}",
                elem_classes=["add-keywords-button", "small-button"],
                variant="primary"  # Make the button more visible
            )
    
    # Connect LoRA dropdown change event to update keywords
    lora_dropdown.change(
        fn=update_keywords_display,
        inputs=[lora_dropdown, lora_enabled],
        outputs=[keywords_column, keywords_html, selected_keywords]
    )
    
    # Also update when the enabled checkbox changes
    lora_enabled.change(
        fn=update_keywords_display,
        inputs=[lora_dropdown, lora_enabled],
        outputs=[keywords_column, keywords_html, selected_keywords]
    )
    
    return {
        "column": keywords_column,
        "html": keywords_html,
        "selected": selected_keywords,
        "copy_button": copy_button,
        "add_button": add_button
    }

# JavaScript for keyword selection
keywords_js = """
// Less invasive initialization - only run once the DOM is fully loaded
(function() {
    // Wait for the page to be fully loaded
    if (document.readyState === 'complete') {
        initializeKeywords();
    } else {
        window.addEventListener('load', initializeKeywords);
    }
    
    function initializeKeywords() {
        console.log('LoRA Keywords: Initializing keyword functionality');
        setupKeywordFunctionality();
        
        // Create a more targeted observer that only watches for our keyword elements
        const observer = new MutationObserver(function(mutations) {
            for (const mutation of mutations) {
                // Only process if we have added nodes
                if (mutation.addedNodes && mutation.addedNodes.length > 0) {
                    // Check if any of the added nodes contain our keywords containers
                    for (const node of mutation.addedNodes) {
                        if (node.querySelector && node.querySelector('.keyword-tag, [id^="lora_keywords_"]')) {
                            setupKeywordFunctionality();
                            break;
                        }
                    }
                }
            }
        });
        
        // Only observe the main content area where our elements would be added
        const mainContent = document.querySelector('.gradio-container');
        if (mainContent) {
            observer.observe(mainContent, {
                childList: true,
                subtree: true,
                attributes: false,
                characterData: false
            });
        }
    }
})();

function setupKeywordFunctionality() {
    // Use a specific selector that only targets our keyword tags
    document.querySelectorAll('[id^="lora_keywords_"] .keyword-tag').forEach(function(element) {
        // Remove any existing handler first to avoid duplicates
        element.removeEventListener('click', keywordClickHandler);
        // Add our handler
        element.addEventListener('click', keywordClickHandler);
    });
}

function keywordClickHandler(event) {
    // Stop event propagation to prevent interfering with other elements
    event.stopPropagation();
    event.preventDefault();
    
    try {
        toggleKeyword(this);
    } catch(e) {
        console.error('Error toggling keyword:', e);
    }
    
    return false;
}

function toggleKeyword(element) {
    // Don't do anything that could affect the broader UI
    // Just toggle our specific element's class in a contained way
    if (!element || typeof element.classList === 'undefined') return;
    
    try {
        // Simply toggle the selected class - minimal DOM manipulation
        element.classList.toggle('selected');
        
        // Find the container in a direct, specific way
        const container = element.closest('.lora-keywords-container');
        if (!container) return;
        
        // Get the ID of the keywords element to find the corresponding selected area
        // Look for a nearby element with id that matches our pattern
        let keywordsElement = container;
        while (keywordsElement && (!keywordsElement.id || !keywordsElement.id.includes('lora_keywords_'))) {
            keywordsElement = keywordsElement.parentElement;
            if (!keywordsElement || keywordsElement.classList.contains('gradio-container')) break;
        }
        
        if (!keywordsElement || !keywordsElement.id) return;
        
        // Extract the matching part to find the related selected input
        const idParts = keywordsElement.id.split('lora_keywords_');
        if (idParts.length !== 2) return;
        
        // Find the matching selected keywords input
        const selectedElementId = 'lora_selected_' + idParts[1];
        const selectedArea = document.getElementById(selectedElementId);
        
        if (!selectedArea) return;
        
        // Get only selected keywords from our container
        const selectedKeywords = Array.from(
            container.querySelectorAll('.keyword-tag.selected')
        ).map(el => el.textContent.trim()).join(', ');
        
        // Update the value without affecting the rest of the UI
        selectedArea.value = selectedKeywords;
        
        // Trigger the input event in a safe way
        try {
            // Create a contained input event
            const inputEvent = new Event('input', {
                bubbles: false,  // Don't bubble up to avoid affecting other components
                cancelable: false
            });
            selectedArea.dispatchEvent(inputEvent);
        } catch (ignored) {}
    } catch (ignored) {}
}

document.head.insertAdjacentHTML('beforeend', `<style>
.lora-keywords-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 5px !important;
    margin-top: 8px !important;
    max-height: 150px !important;
    overflow-y: auto !important;
    padding: 10px !important;
    border-radius: 8px !important;
    background-color: rgba(0, 0, 0, 0.075) !important;
    border: 2px solid rgba(33, 150, 243, 0.3) !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

.keywords-header {
    margin-bottom: 5px !important;
    color: #2196F3 !important;
    font-weight: bold !important;
}

.keywords-info-message {
    padding: 15px !important;
    background-color: rgba(33, 150, 243, 0.1) !important;
    border-left: 4px solid #2196F3 !important;
    margin: 8px 0 !important;
    color: #333 !important;
}

.keywords-error-message {
    padding: 15px !important;
    background-color: rgba(244, 67, 54, 0.1) !important;
    border-left: 4px solid #f44336 !important;
    margin: 8px 0 !important;
    color: #333 !important;
}
.keyword-tag {
    display: inline-block !important;
    background-color: #e0e0e0 !important;
    border-radius: 4px !important;
    padding: 8px 12px !important;
    font-size: 14px !important;
    cursor: pointer !important;
    user-select: none !important;
    transition: all 0.2s ease !important;
    border: 1px solid rgba(0, 0, 0, 0.15) !important;
    margin: 4px !important;
    position: relative !important;
    z-index: 10 !important;
    text-align: center !important;
    font-weight: normal !important;
}

/* Style the buttons */
.copy-keywords-button, .add-keywords-button {
    min-width: 130px !important;
    height: auto !important;
    padding: 8px 16px !important;
}

/* Make the selected keywords textbox more visible */
.selected-keywords textarea {
    border: 2px solid rgba(33, 150, 243, 0.5) !important;
    padding: 8px !important;
    font-weight: bold !important;
    background-color: rgba(33, 150, 243, 0.05) !important;
}

.keywords-ui-container {
    margin-top: 10px !important;
    margin-bottom: 20px !important;
    padding: 10px !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 8px !important;
    background-color: rgba(255, 255, 255, 0.5) !important;
}
.keyword-tag:hover {
    background-color: #d0d0d0 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2) !important;
    z-index: 20 !important;
    cursor: pointer !important;
    pointer-events: auto !important;
    color: #000 !important;
}
.keyword-tag.selected {
    background-color: #2196F3 !important;
    color: white !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.25) !important;
    font-weight: bold !important;
    transform: translateY(-1px) !important;
    border-color: #1976D2 !important;
}
/* Target only our specific elements without affecting the rest of the UI */
div[id^="lora_keywords_"] {
    pointer-events: auto !important;
}
div.keyword-tag {
    pointer-events: auto !important;
}
</style>`);
"""

# Initialize
is_running = True
request_thread = threading.Thread(target=process_requests, daemon=True)
request_thread.start()