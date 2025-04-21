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
            request_times.append(time.time())
            func, args, result_queue = request_queue.get()
            try:
                result = func(*args)
                result_queue.put((True, result))
            except Exception as e:
                print(f"Error processing request: {e}")
                result_queue.put((False, None))
            
            request_queue.task_done()
        else:
            time.sleep(0.1)

# Queue an API request for rate limiting
def queue_api_request(func, *args):
    result_queue = Queue()
    request_queue.put((func, args, result_queue))
    return result_queue

# Metadata helpers
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

# Process keywords function - split comma-separated keywords and join them
def process_keywords(keywords):
    if not keywords:
        return ""
    
    # Process and flatten the keywords
    processed = []
    for kw in keywords:
        if ',' in kw:
            # Split comma-separated keywords
            parts = [p.strip() for p in kw.split(',')]
            processed.extend([p for p in parts if p])
        else:
            # Add single keyword
            kw_stripped = kw.strip()
            if kw_stripped:
                processed.append(kw_stripped)
    
    # Remove duplicates while maintaining order
    seen = set()
    unique_keywords = []
    for kw in processed:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    # Join all keywords with commas
    return ', '.join(unique_keywords)

def update_keywords_display(lora_filename, enabled):
    # Return nothing if LoRA is not enabled or no filename
    if not enabled or lora_filename == "None":
        return gr.update(visible=True), gr.update(value="")
    
    # Get metadata for the selected LoRA
    try:
        metadata = get_lora_metadata(lora_filename)
        
        # If no keywords found, show a helpful message and try to fetch
        if not metadata or not metadata.get("keywords") or len(metadata.get("keywords", [])) == 0:
            print(f"No keywords found for LoRA: {lora_filename}")
            
            # Try to fetch metadata in the background
            threading.Thread(target=lambda: get_lora_metadata(lora_filename, refresh=True), daemon=True).start()
            
            return gr.update(visible=True), gr.update(value="No keywords found. Checking CivitAI...")
        
        # Process keywords - split comma-separated values and join
        keywords = metadata.get("keywords", [])
        processed_keywords = process_keywords(keywords)
        print(f"Found {len(keywords)} keywords for LoRA: {lora_filename}")
        
        # Return the processed keywords
        return gr.update(visible=True), gr.update(value=processed_keywords)
    except Exception as e:
        print(f"Error displaying keywords for {lora_filename}: {e}")
        return gr.update(visible=True), gr.update(value=f"Error retrieving keywords: {str(e)}")

def create_keywords_ui(gr_lib, lora_dropdown, lora_enabled):
    # Create a container for the keywords UI
    with gr_lib.Column(elem_classes=["keywords-ui-container"], visible=True) as keywords_column:
        # Add a header
        gr_lib.Markdown("### LoRA Keywords", elem_classes=["keywords-header"])
        
        # Create a textbox to display keywords
        keywords_textbox = gr_lib.Textbox(
            label="Keywords", 
            elem_id=f"lora_keywords_{lora_dropdown.elem_id}",
            elem_classes=["keywords-display"],
            lines=3,
            interactive=True,
            placeholder="Keywords will appear here when a LoRA is selected"
        )
    
    # Connect LoRA dropdown change event to update keywords
    lora_dropdown.change(
        fn=update_keywords_display,
        inputs=[lora_dropdown, lora_enabled],
        outputs=[keywords_column, keywords_textbox]
    )
    
    # Also update when the enabled checkbox changes
    lora_enabled.change(
        fn=update_keywords_display,
        inputs=[lora_dropdown, lora_enabled],
        outputs=[keywords_column, keywords_textbox]
    )
    
    return {
        "column": keywords_column,
        "keywords": keywords_textbox
    }

# Initialize
is_running = True
request_thread = threading.Thread(target=process_requests, daemon=True)
request_thread.start()