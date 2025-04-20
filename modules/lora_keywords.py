import os
import json
import requests
import time
import threading
from queue import Queue

import gradio as gr
from modules.hash_cache import sha256_from_cache
from modules.util import get_file_from_folder_list

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
    return getattr(modules.config, "civitai_api_key", "") or ""

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
        lora_path = get_file_from_folder_list(lora_filename, modules.config.paths_loras)
    
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
    if not enabled or lora_filename == "None":
        return gr.update(visible=False), gr.update(value=""), gr.update(value="")
    
    metadata = get_lora_metadata(lora_filename)
    if not metadata or not metadata.get("keywords"):
        return gr.update(visible=False), gr.update(value=""), gr.update(value="")
    
    html = generate_keywords_html(metadata.get("keywords", []))
    return gr.update(visible=True), gr.update(value=html), gr.update(value="")

def create_keywords_ui(gr_lib, lora_dropdown, lora_enabled):
    with gr_lib.Column(visible=False) as keywords_column:
        keywords_html = gr_lib.HTML(label="Keywords", elem_id=f"lora_keywords_{lora_dropdown.elem_id}")
        selected_keywords = gr_lib.Textbox(label="Selected Keywords", elem_id=f"lora_selected_{lora_dropdown.elem_id}")
        
        with gr_lib.Row():
            copy_button = gr_lib.Button("Copy to Clipboard", elem_id=f"copy_keywords_{lora_dropdown.elem_id}")
            add_button = gr_lib.Button("Add to Prompt", elem_id=f"add_keywords_{lora_dropdown.elem_id}")
    
    # Connect change handler
    lora_dropdown.change(
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
function toggleKeyword(element) {
    element.classList.toggle('selected');
    
    const container = element.closest('.lora-keywords-container');
    const selectedArea = container.parentElement.parentElement.querySelector('input[id*="lora_selected"]');
    const selectedKeywords = Array.from(
        container.querySelectorAll('.keyword-tag.selected')
    ).map(el => el.textContent).join(', ');
    
    selectedArea.value = selectedKeywords;
    selectedArea.dispatchEvent(new Event('input', {bubbles: true}));
}

document.head.insertAdjacentHTML('beforeend', `<style>
.lora-keywords-container {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 8px;
    max-height: 120px;
    overflow-y: auto;
    padding: 5px;
    border-radius: 4px;
    background-color: rgba(0, 0, 0, 0.03);
}
.keyword-tag {
    background-color: #e0e0e0;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
    cursor: pointer;
    user-select: none;
    transition: all 0.2s ease;
    border: 1px solid rgba(0, 0, 0, 0.1);
    margin: 2px;
}
.keyword-tag:hover {
    background-color: #d0d0d0;
    transform: translateY(-1px);
}
.keyword-tag.selected {
    background-color: #2196F3;
    color: white;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}
</style>`);
"""

# Initialize
is_running = True
request_thread = threading.Thread(target=process_requests, daemon=True)
request_thread.start()