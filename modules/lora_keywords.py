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

# UI components
def generate_keywords_html(keywords, selected=None):
    if not keywords:
        return ""
    
    selected = selected or []
    html = '<div class="lora-keywords-container">'
    
    # Process each keyword - split comma-separated values
    processed_keywords = []
    for kw in keywords:
        # Split by comma and strip whitespace
        if ',' in kw:
            split_keywords = [k.strip() for k in kw.split(',')]
            processed_keywords.extend(split_keywords)
        else:
            processed_keywords.append(kw.strip())
    
    # Make them unique
    processed_keywords = list(dict.fromkeys(processed_keywords))
    
    # Generate HTML for each keyword
    for kw in processed_keywords:
        if not kw:  # Skip empty keywords
            continue
        cls = " selected" if kw in selected else ""
        # Add data attributes for the keyword and the ID of the textbox to update
        html += f'<div class="keyword-tag{cls}" data-keyword="{kw}" data-input-id="{selected}" onclick="clickKeywordTag(event, this)">{kw}</div>'
    
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
            lines=2,  # Make the textbox taller for better visibility
            interactive=False,  # Make it read-only
            placeholder="Click on keywords above to select them"
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
    
        # Add dummy functions for the buttons
        # Simple function that returns its input
        def noop(x):
            return x
        
        # Connect the buttons with simplified JavaScript-only handling
        copy_button.click(
            fn=noop,
            inputs=[selected_keywords],
            outputs=[selected_keywords],
            _js="""
            function(text) {
                // Create a temporary element
                const el = document.createElement('textarea');
                el.value = text || '';
                document.body.appendChild(el);
                el.select();
                
                try {
                    // Copy text to clipboard
                    document.execCommand('copy');
                    
                    // Show feedback
                    const button = document.activeElement;
                    if (button) {
                        const originalText = button.textContent;
                        button.textContent = "Copied! ✓";
                        setTimeout(() => {
                            button.textContent = originalText;
                        }, 1000);
                    }
                } catch(e) {
                    console.error('Copy failed:', e);
                } finally {
                    document.body.removeChild(el);
                }
                
                return text || '';
            }
            """
        )
        
        # Add to prompt without Python dependency
        add_button.click(
            fn=noop,
            inputs=[selected_keywords],
            outputs=[selected_keywords],
            _js="""
            function(selectedText) {
                // Get the prompt element directly
                const promptElement = document.getElementById('positive_prompt');
                
                if (promptElement && selectedText) {
                    // Get current prompt content
                    const currentPrompt = promptElement.value || '';
                    
                    // Add the selected keywords
                    if (currentPrompt && currentPrompt.trim()) {
                        promptElement.value = currentPrompt.trim() + ', ' + selectedText;
                    } else {
                        promptElement.value = selectedText;
                    }
                    
                    // Trigger input event
                    promptElement.dispatchEvent(new Event('input', { bubbles: true }));
                    
                    // Show feedback
                    const button = document.activeElement;
                    if (button) {
                        const originalText = button.textContent;
                        button.textContent = "Added! ✓";
                        setTimeout(() => {
                            button.textContent = originalText;
                        }, 1000);
                    }
                }
                
                return selectedText || '';
            }
            """
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
// Reusable function to find the related select textbox for clicked keywords
function findRelatedSelectedTextbox(keywordContainer) {
    // Get the parent div that contains the keywords HTML and the textbox
    let parent = keywordContainer.closest('.keywords-ui-container');
    if (!parent) {
        // Fallback to traversing up the DOM
        parent = keywordContainer;
        for (let i = 0; i < 5 && parent; i++) {
            if (parent.classList && parent.classList.contains('keywords-ui-container')) {
                break;
            }
            parent = parent.parentElement;
            if (!parent) break;
        }
    }
    
    // Find the textbox within this container
    if (parent) {
        const textbox = parent.querySelector('textarea[id*="lora_selected"]');
        return textbox;
    }
    
    // Final fallback: try to find any lora_selected textbox nearby
    return document.querySelector('textarea[id*="lora_selected"]');
}

// Click handler for keyword tags - enhanced for reliability
function clickKeywordTag(event, element) {
    try {
        // Prevent default browser behavior
        if (event) {
            event.preventDefault();
            event.stopPropagation();
        }
        
        // Debug log
        console.log('Keyword clicked:', element.textContent);
        
        // Toggle selection class
        element.classList.toggle('selected');
        
        // Find the container with all keywords
        const container = element.closest('.lora-keywords-container');
        if (!container) {
            console.error('Could not find keywords container');
            return;
        }
        
        // Try multiple approaches to find the textbox
        let textbox = null;
        
        // Approach 1: Using the parent form or container
        const parentContainer = container.closest('.keywords-ui-container');
        if (parentContainer) {
            textbox = parentContainer.querySelector('textarea[id*="lora_selected"]');
            if (textbox) console.log('Found textbox via parent container');
        }
        
        // Approach 2: Look in the closest form
        if (!textbox) {
            const form = container.closest('form');
            if (form) {
                const boxes = form.querySelectorAll('textarea[id*="lora_selected"]');
                if (boxes.length > 0) {
                    textbox = boxes[0];
                    console.log('Found textbox via form');
                }
            }
        }
        
        // Approach 3: Search in DOM vicinity
        if (!textbox) {
            let currentNode = container;
            for (let i = 0; i < 5 && currentNode; i++) {
                currentNode = currentNode.parentElement;
                if (!currentNode) break;
                
                // Look for the textarea in this parent element
                const found = currentNode.querySelector('textarea[id*="lora_selected"]');
                if (found) {
                    textbox = found;
                    console.log('Found textbox via DOM traversal');
                    break;
                }
            }
        }
        
        // Last resort - grab any textbox with lora_selected in its ID
        if (!textbox) {
            const allBoxes = document.querySelectorAll('textarea[id*="lora_selected"]');
            if (allBoxes.length > 0) {
                textbox = allBoxes[0];
                console.log('Found textbox via document query');
            }
        }
        
        if (!textbox) {
            console.error('Could not find selected keywords textbox');
            return;
        }
        
        // Get all selected keywords
        const selectedElements = container.querySelectorAll('.keyword-tag.selected');
        const selectedKeywords = Array.from(selectedElements)
            .map(el => el.textContent.trim())
            .join(', ');
        
        console.log('Selected keywords:', selectedKeywords);
        console.log('Updating textbox:', textbox.id);
        
        // Update the textbox directly
        textbox.value = selectedKeywords;
        
        // Use multiple event dispatching methods for reliability
        try {
            // Update using direct property setting and events
            textbox.textContent = selectedKeywords;
            textbox.dispatchEvent(new Event('input', { bubbles: true }));
            textbox.dispatchEvent(new Event('change', { bubbles: true }));
            
            // Force UI update through parent reflow
            textbox.parentElement.style.display = textbox.parentElement.style.display;
        } catch (err) {
            console.error('Error dispatching events:', err);
        }
    } catch (error) {
        console.error('Error in clickKeywordTag:', error);
    }
}

// Copy selected keywords to clipboard
function copySelectedKeywords(selectedText) {
    if (!selectedText) {
        console.log('No keywords to copy');
        return;
    }
    
    // Copy using modern clipboard API with fallback
    try {
        navigator.clipboard.writeText(selectedText)
            .then(() => {
                console.log('Copied to clipboard successfully');
                showButtonFeedback('copy');
            })
            .catch(err => {
                console.error('Clipboard API failed:', err);
                copyWithExecCommand(selectedText);
            });
    } catch (e) {
        console.error('Error using Clipboard API:', e);
        copyWithExecCommand(selectedText);
    }
}

// Fallback copy method using execCommand
function copyWithExecCommand(text) {
    try {
        // Create temporary element
        const el = document.createElement('textarea');
        el.value = text;
        el.setAttribute('readonly', '');
        el.style.position = 'absolute';
        el.style.left = '-9999px';
        document.body.appendChild(el);
        
        // Select and copy
        el.select();
        const success = document.execCommand('copy');
        document.body.removeChild(el);
        
        if (success) {
            console.log('Copied with execCommand');
            showButtonFeedback('copy');
        } else {
            console.error('execCommand copy failed');
        }
    } catch (e) {
        console.error('Failed to copy with execCommand:', e);
    }
}

// Add selected keywords to prompt
function addKeywordsToPrompt(selectedText) {
    if (!selectedText) {
        console.log('No keywords to add');
        return;
    }
    
    try {
        // Find the prompt textbox
        const promptBox = document.getElementById('positive_prompt');
        if (!promptBox) {
            console.error('Could not find prompt textbox');
            return;
        }
        
        // Get current prompt
        let currentPrompt = promptBox.value || '';
        
        // Add keywords to prompt
        if (currentPrompt && currentPrompt.trim()) {
            promptBox.value = currentPrompt.trim() + ', ' + selectedText;
        } else {
            promptBox.value = selectedText;
        }
        
        // Trigger input event
        promptBox.dispatchEvent(new Event('input', { bubbles: true }));
        
        console.log('Keywords added to prompt');
        showButtonFeedback('add');
    } catch (e) {
        console.error('Error adding keywords to prompt:', e);
    }
}

// Show visual feedback on button click
function showButtonFeedback(type) {
    let button;
    
    if (type === 'copy') {
        button = document.querySelector('.copy-keywords-button');
    } else if (type === 'add') {
        button = document.querySelector('.add-keywords-button');
    }
    
    if (button) {
        const originalText = button.textContent;
        button.textContent = type === 'copy' ? 'Copied! ✓' : 'Added! ✓';
        
        setTimeout(() => {
            button.textContent = originalText;
        }, 1000);
    }
}

// Initialize styles and event handlers on page load
document.addEventListener('DOMContentLoaded', function() {
    // Add styles if not already added
    if (!document.getElementById('lora-keywords-style')) {
        const style = document.createElement('style');
        style.id = 'lora-keywords-style';
        style.textContent = `
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
            
            /* Make the selected keywords textbox clearly read-only */
            .selected-keywords textarea {
                background-color: rgba(240, 240, 240, 0.5) !important;
                border: 1px solid rgba(0, 0, 0, 0.2) !important;
                color: #444 !important;
                cursor: default !important;
            }
            
            /* Style for copy and add buttons */
            .copy-keywords-button, .add-keywords-button {
                min-width: 120px !important;
                transition: all 0.2s !important;
            }
            
            .copy-keywords-button:hover, .add-keywords-button:hover {
                transform: translateY(-1px) !important;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
            }
        `;
        document.head.appendChild(style);
    }
    
    // Set up a mutation observer to handle dynamically added keywords
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes && mutation.addedNodes.length > 0) {
                // Look for added nodes that might contain our keywords
                mutation.addedNodes.forEach(function(node) {
                    if (node.querySelectorAll) {
                        // Find any keyword tags added to the DOM
                        const tags = node.querySelectorAll('.keyword-tag');
                        if (tags.length > 0) {
                            console.log('Found new keyword tags:', tags.length);
                        }
                    }
                });
            }
        });
    });
    
    // Start observing
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});
"""

# Initialize
is_running = True
request_thread = threading.Thread(target=process_requests, daemon=True)
request_thread.start()