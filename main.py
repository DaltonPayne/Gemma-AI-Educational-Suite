#!/usr/bin/env python3
import sys
import os
import subprocess
import importlib.util
import platform

def install_dependencies():
    print("Checking dependencies...")
    required = [
        "torch", "transformers", "pillow", "numpy", 
        "sounddevice", "soundfile", "pyperclip", "PyMuPDF", "python-docx", "tiktoken"
    ]
    
    for module in required:
        module_name = "fitz" if module == "PyMuPDF" else "docx" if module == "python-docx" else module
        if not importlib.util.find_spec(module_name):
            print(f"Installing {module}...")
            subprocess.run([sys.executable, "-m", "pip", "install", module], check=True)
    
    if "COLAB_" in "".join(os.environ.keys()):
        colab_cmds = [
            "pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo",
            "pip install sentencepiece protobuf 'datasets>=3.4.1,<4.0.0' 'huggingface_hub>=0.34.0' hf_transfer",
            "pip install --no-deps unsloth"
        ]
        for cmd in colab_cmds:
            subprocess.run(cmd, shell=True, check=True)
    else:
        if not importlib.util.find_spec("unsloth"):
            subprocess.run([sys.executable, "-m", "pip", "install", "unsloth"], check=True)

try:
    install_dependencies()
except:
    print("Auto-install failed. Please install dependencies manually.")

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import gc
import torch
import tempfile
import threading
import queue
from PIL import Image, ImageGrab, ImageTk
import numpy as np
from unsloth import FastModel
from transformers import TextStreamer
import sounddevice as sd
import soundfile as sf
import pyperclip
import time
import fitz
import io
import docx
from html.parser import HTMLParser
import re
import tiktoken
import json
import urllib.request
import urllib.error

torch._dynamo.config.cache_size_limit = 64

class ModelManager:
    def __init__(self):
        self.models_dir = os.path.join(os.path.expanduser("~"), ".educational_ai_tutor", "models")
        self.config_file = os.path.join(os.path.expanduser("~"), ".educational_ai_tutor", "config.json")
        self.ensure_directories()
        self.config = self.load_config()
        
    def ensure_directories(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
    def load_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {"downloaded_models": [], "last_used_model": None, "offline_mode": False}
        
    def save_config(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
            
    def is_online(self):
        try:
            urllib.request.urlopen('https://huggingface.co', timeout=5)
            return True
        except:
            return False
            
    def get_model_path(self, model_name):
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        return os.path.join(self.models_dir, safe_name)
        
    def is_model_downloaded(self, model_name):
        model_path = self.get_model_path(model_name)
        return os.path.exists(model_path) and os.path.isdir(model_path)
        
    def download_model(self, model_name, progress_callback=None):
        try:
            if progress_callback:
                progress_callback("Checking internet connection...")
                
            if not self.is_online():
                raise Exception("No internet connection available")
                
            if progress_callback:
                progress_callback(f"Downloading {model_name}...")
                
            # Load model which will download it if not cached
            model, tokenizer = FastModel.from_pretrained(
                model_name=model_name,
                dtype=None,
                max_seq_length=1024,
                load_in_4bit=True,
                full_finetuning=False,
            )
            
            if progress_callback:
                progress_callback("Saving model locally...")
            
            # Save to our local models directory
            model_path = self.get_model_path(model_name)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            # Update config
            if model_name not in self.config["downloaded_models"]:
                self.config["downloaded_models"].append(model_name)
            self.config["last_used_model"] = model_name
            self.save_config()
            
            if progress_callback:
                progress_callback("Model downloaded and saved successfully!")
                
            return model, tokenizer
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Download failed: {str(e)}")
            raise e
            
    def load_local_model(self, model_name, progress_callback=None):
        try:
            model_path = self.get_model_path(model_name)
            
            if not self.is_model_downloaded(model_name):
                raise Exception(f"Model {model_name} not found locally")
                
            if progress_callback:
                progress_callback(f"Loading local model {model_name}...")
                
            model, tokenizer = FastModel.from_pretrained(
                model_name=model_path,
                dtype=None,
                max_seq_length=1024,
                load_in_4bit=True,
                full_finetuning=False,
            )
            
            self.config["last_used_model"] = model_name
            self.save_config()
            
            if progress_callback:
                progress_callback("Local model loaded successfully!")
                
            return model, tokenizer
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Failed to load local model: {str(e)}")
            raise e
            
    def load_model(self, model_name, progress_callback=None):
        # Try local first, then download if needed
        if self.is_model_downloaded(model_name):
            try:
                return self.load_local_model(model_name, progress_callback)
            except Exception as e:
                print(f"Failed to load local model, will try to download: {e}")
                
        # If local loading failed or model not downloaded, try to download
        return self.download_model(model_name, progress_callback)
        
    def get_downloaded_models(self):
        return self.config.get("downloaded_models", [])
        
    def delete_model(self, model_name):
        try:
            model_path = self.get_model_path(model_name)
            if os.path.exists(model_path):
                import shutil
                shutil.rmtree(model_path)
                
            if model_name in self.config["downloaded_models"]:
                self.config["downloaded_models"].remove(model_name)
                
            if self.config["last_used_model"] == model_name:
                self.config["last_used_model"] = None
                
            self.save_config()
            return True
        except Exception as e:
            print(f"Error deleting model: {e}")
            return False
            
    def get_model_size(self, model_name):
        try:
            model_path = self.get_model_path(model_name)
            if not os.path.exists(model_path):
                return 0
                
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(model_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(file_path)
            return total_size
        except:
            return 0
            
    def format_size(self, size_bytes):
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

class TokenManager:
    def __init__(self):
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoder = None
        self.max_context_tokens = 32000
        self.image_tokens = 256
        self.audio_tokens_per_second = 6.25
        
    def count_text_tokens(self, text):
        if not text:
            return 0
        if self.encoder:
            return len(self.encoder.encode(str(text)))
        else:
            return len(str(text).split()) * 1.3
            
    def count_image_tokens(self, image_count):
        return image_count * self.image_tokens
        
    def count_audio_tokens(self, duration_seconds):
        return int(duration_seconds * self.audio_tokens_per_second)
        
    def get_color_for_usage(self, token_count):
        usage_percent = (token_count / self.max_context_tokens) * 100
        if usage_percent < 50:
            return '#28a745'
        elif usage_percent < 75:
            return '#ffc107' 
        elif usage_percent < 90:
            return '#fd7e14'
        else:
            return '#dc3545'

class DocumentViewer:
    def __init__(self, parent, doc_path, callback=None):
        self.parent = parent
        self.doc_path = doc_path
        self.callback = callback
        self.file_ext = os.path.splitext(doc_path)[1].lower()
        
        if self.file_ext == '.pdf':
            self.doc = fitz.open(doc_path)
            self.total_pages = len(self.doc)
            self.is_pdf = True
        else:
            self.doc = None
            self.total_pages = 1
            self.is_pdf = False
            
        self.current_page = 0
        self.selected_pages = set()
        self.zoom_level = 1.0
        
        self.setup_viewer()
        self.display_content()
        
    def setup_viewer(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"Document Viewer - {os.path.basename(self.doc_path)}")
        self.window.geometry("800x900")
        self.window.resizable(True, True)
        self.window.minsize(600, 500)
        
        toolbar = ttk.Frame(self.window)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        if self.is_pdf:
            ttk.Button(toolbar, text="◀", command=self.prev_page, width=3).pack(side=tk.LEFT, padx=2)
            ttk.Button(toolbar, text="▶", command=self.next_page, width=3).pack(side=tk.LEFT, padx=2)
            
            self.page_var = tk.StringVar(value=f"1 / {self.total_pages}")
            ttk.Label(toolbar, textvariable=self.page_var).pack(side=tk.LEFT, padx=10)
            
            ttk.Button(toolbar, text="🔍+", command=self.zoom_in, width=4).pack(side=tk.LEFT, padx=2)
            ttk.Button(toolbar, text="🔍-", command=self.zoom_out, width=4).pack(side=tk.LEFT, padx=2)
            
            ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
            
            ttk.Label(toolbar, text="Select Pages:").pack(side=tk.LEFT, padx=5)
            self.page_from = tk.StringVar(value="1")
            self.page_to = tk.StringVar(value=str(self.total_pages))
            
            ttk.Entry(toolbar, textvariable=self.page_from, width=4).pack(side=tk.LEFT, padx=2)
            ttk.Label(toolbar, text="to").pack(side=tk.LEFT, padx=2)
            ttk.Entry(toolbar, textvariable=self.page_to, width=4).pack(side=tk.LEFT, padx=2)
            
            ttk.Button(toolbar, text="Select Range", command=self.select_range).pack(side=tk.LEFT, padx=5)
            ttk.Button(toolbar, text="Clear Selection", command=self.clear_selection).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="Use All Text", command=self.use_all_text).pack(side=tk.LEFT, padx=5)
        if self.is_pdf:
            ttk.Button(toolbar, text="Use Selected", command=self.use_selected).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="Open in System", command=self.open_in_system).pack(side=tk.RIGHT, padx=5)
        
        if self.is_pdf:
            self.canvas_frame = ttk.Frame(self.window)
            self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.canvas = tk.Canvas(self.canvas_frame, bg='white')
            v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
            h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
            
            self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            self.canvas.bind('<Button-1>', self.toggle_page_selection)
            self.canvas.bind('<MouseWheel>', self.on_mousewheel)
        else:
            text_frame = ttk.Frame(self.window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.text_display = scrolledtext.ScrolledText(
                text_frame,
                wrap=tk.WORD,
                font=('Arial', 10),
                bg='white'
            )
            self.text_display.pack(fill=tk.BOTH, expand=True)
            
        self.selection_label = ttk.Label(self.window, text="", foreground='blue')
        self.selection_label.pack(pady=5)
        
        if self.is_pdf:
            self.canvas_frame.columnconfigure(0, weight=1)
            self.canvas_frame.rowconfigure(0, weight=1)
        
    def display_content(self):
        if self.is_pdf:
            self.display_pdf_page()
        else:
            self.display_text_content()
            
    def display_pdf_page(self):
        if 0 <= self.current_page < self.total_pages:
            page = self.doc[self.current_page]
            mat = fitz.Matrix(self.zoom_level, self.zoom_level)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            
            img = Image.open(io.BytesIO(img_data))
            self.photo = ImageTk.PhotoImage(img)
            
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            page_selected = (self.current_page + 1) in self.selected_pages
            bg_color = '#ffffcc' if page_selected else 'white'
            self.canvas.configure(bg=bg_color)
            
            self.page_var.set(f"{self.current_page + 1} / {self.total_pages}")
            self.update_selection_label()
            
    def display_text_content(self):
        try:
            content = self.extract_text()
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(1.0, content)
            self.selection_label.config(text=f"Document loaded: {len(content)} characters")
        except Exception as e:
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(1.0, f"Error loading document: {str(e)}")
            
    def extract_text(self):
        try:
            if self.file_ext == '.pdf':
                if self.selected_pages:
                    text_content = ""
                    for page_num in sorted(self.selected_pages):
                        page = self.doc[page_num - 1]
                        text_content += f"\n--- Page {page_num} ---\n"
                        text_content += page.get_text()
                    return text_content
                else:
                    text_content = ""
                    for page_num in range(len(self.doc)):
                        page = self.doc[page_num]
                        text_content += f"\n--- Page {page_num + 1} ---\n"
                        text_content += page.get_text()
                    return text_content
                    
            elif self.file_ext == '.docx':
                doc = docx.Document(self.doc_path)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                
            elif self.file_ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.xml', '.json']:
                with open(self.doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
                    
            elif self.file_ext == '.rtf':
                with open(self.doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                    rtf_content = f.read()
                    return re.sub(r'\\[a-z]+\d*\s?|\{|\}', '', rtf_content)
                    
            else:
                with open(self.doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
                    
        except Exception as e:
            return f"Error extracting text: {str(e)}"
            
    def open_in_system(self):
        try:
            system = platform.system()
            if system == 'Windows':
                os.startfile(self.doc_path)
            elif system == 'Darwin':
                subprocess.run(['open', self.doc_path])
            else:
                subprocess.run(['xdg-open', self.doc_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file in system viewer: {str(e)}")
            
    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.display_pdf_page()
            
    def next_page(self):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.display_pdf_page()
            
    def zoom_in(self):
        self.zoom_level = min(3.0, self.zoom_level + 0.2)
        self.display_pdf_page()
        
    def zoom_out(self):
        self.zoom_level = max(0.5, self.zoom_level - 0.2)
        self.display_pdf_page()
        
    def toggle_page_selection(self, event):
        if not self.is_pdf:
            return
        page_num = self.current_page + 1
        if page_num in self.selected_pages:
            self.selected_pages.remove(page_num)
        else:
            self.selected_pages.add(page_num)
        self.display_pdf_page()
        
    def select_range(self):
        if not self.is_pdf:
            return
        try:
            start = int(self.page_from.get())
            end = int(self.page_to.get())
            
            start = max(1, min(start, self.total_pages))
            end = max(start, min(end, self.total_pages))
            
            for page_num in range(start, end + 1):
                self.selected_pages.add(page_num)
                
            self.display_pdf_page()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid page numbers")
            
    def clear_selection(self):
        if not self.is_pdf:
            return
        self.selected_pages.clear()
        self.display_pdf_page()
        
    def use_selected(self):
        if not self.is_pdf or not self.selected_pages:
            messagebox.showwarning("No Selection", "Please select pages first")
            return
            
        text_content = self.extract_text()
        if self.callback:
            self.callback(text_content, list(sorted(self.selected_pages)))
            
        self.window.destroy()
        
    def use_all_text(self):
        text_content = self.extract_text()
        if self.callback:
            if self.is_pdf:
                all_pages = list(range(1, self.total_pages + 1))
                self.callback(text_content, all_pages)
            else:
                self.callback(text_content, [1])
            
        self.window.destroy()
        
    def update_selection_label(self):
        if not self.is_pdf:
            return
        if self.selected_pages:
            pages_list = sorted(self.selected_pages)
            if len(pages_list) <= 5:
                pages_str = ", ".join(map(str, pages_list))
            else:
                pages_str = f"{', '.join(map(str, pages_list[:3]))}, ... and {len(pages_list)-3} more"
            self.selection_label.config(text=f"Selected pages: {pages_str}")
        else:
            self.selection_label.config(text="No pages selected - click pages to select")
            
    def on_mousewheel(self, event):
        if self.is_pdf:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

class ModelManagerViewer:
    def __init__(self, parent, model_manager, app_callback=None):
        self.parent = parent
        self.model_manager = model_manager
        self.app_callback = app_callback
        self.setup_viewer()
        self.refresh_display()
        
    def setup_viewer(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title("🤖 Offline Model Manager")
        self.window.geometry("800x600")
        self.window.resizable(True, True)
        self.window.minsize(600, 400)
        
        # Header
        header_frame = ttk.Frame(self.window)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = ttk.Label(header_frame, text="🤖 Offline Model Manager", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        # Connection status
        self.status_label = ttk.Label(header_frame, text="", font=('Arial', 10))
        self.status_label.pack(side=tk.RIGHT)
        self.update_connection_status()
        
        # Controls
        controls_frame = ttk.Frame(self.window)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(controls_frame, text="📥 Download Model", command=self.download_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="🔄 Refresh", command=self.refresh_display).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="🧹 Clean Cache", command=self.clean_cache).pack(side=tk.LEFT, padx=5)
        
        # Offline mode toggle
        self.offline_var = tk.BooleanVar(value=self.model_manager.config.get("offline_mode", False))
        offline_check = ttk.Checkbutton(controls_frame, text="🌐 Offline Mode", 
                                       variable=self.offline_var, command=self.toggle_offline_mode)
        offline_check.pack(side=tk.RIGHT, padx=5)
        
        # Model list
        list_frame = ttk.LabelFrame(self.window, text="Downloaded Models", padding="5")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for models
        columns = ('Model', 'Size', 'Status', 'Actions')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=12)
        
        self.tree.heading('Model', text='Model Name')
        self.tree.heading('Size', text='Size')
        self.tree.heading('Status', text='Status')
        self.tree.heading('Actions', text='Actions')
        
        self.tree.column('Model', width=300)
        self.tree.column('Size', width=100)
        self.tree.column('Status', width=100)
        self.tree.column('Actions', width=150)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action buttons
        action_frame = ttk.Frame(list_frame)
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="🔄 Load Model", command=self.load_selected_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="🗑️ Delete Model", command=self.delete_selected_model).pack(side=tk.LEFT, padx=5)
        
        # Info panel
        info_frame = ttk.LabelFrame(self.window, text="Information", padding="5")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD, font=('Arial', 9), bg='#f8f8ff')
        self.info_text.pack(fill=tk.X)
        
        # Bind events
        self.tree.bind('<Double-1>', lambda e: self.load_selected_model())
        
        # Close button
        close_frame = ttk.Frame(self.window)
        close_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(close_frame, text="Close", command=self.window.destroy).pack(side=tk.RIGHT)
        
    def update_connection_status(self):
        if self.model_manager.is_online():
            self.status_label.config(text="🟢 Online", foreground='green')
        else:
            self.status_label.config(text="🔴 Offline", foreground='red')
            
    def toggle_offline_mode(self):
        self.model_manager.config["offline_mode"] = self.offline_var.get()
        self.model_manager.save_config()
        if self.app_callback:
            self.app_callback("offline_mode_changed")
            
    def refresh_display(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Add downloaded models
        downloaded_models = self.model_manager.get_downloaded_models()
        last_used = self.model_manager.config.get("last_used_model")
        
        for model_name in downloaded_models:
            size = self.model_manager.get_model_size(model_name)
            size_str = self.model_manager.format_size(size)
            
            status = "Active" if model_name == last_used else "Downloaded"
            
            self.tree.insert('', tk.END, values=(model_name, size_str, status, ""))
            
        # Update info
        self.update_info()
        self.update_connection_status()
        
    def update_info(self):
        self.info_text.delete(1.0, tk.END)
        
        downloaded_models = self.model_manager.get_downloaded_models()
        total_size = sum(self.model_manager.get_model_size(model) for model in downloaded_models)
        
        info_text = f"📊 OFFLINE MODEL STATUS\n\n"
        info_text += f"Downloaded Models: {len(downloaded_models)}\n"
        info_text += f"Total Storage Used: {self.model_manager.format_size(total_size)}\n"
        info_text += f"Models Directory: {self.model_manager.models_dir}\n"
        info_text += f"Offline Mode: {'Enabled' if self.offline_var.get() else 'Disabled'}\n"
        info_text += f"Internet Connection: {'Available' if self.model_manager.is_online() else 'Not Available'}\n\n"
        
        if self.offline_var.get():
            info_text += "⚠️ Offline Mode is enabled. Only local models will be used.\n"
        else:
            info_text += "🌐 Online Mode: Will download models if not available locally.\n"
            
        self.info_text.insert(1.0, info_text)
        
    def download_model(self):
        dialog = ModelDownloadDialog(self.window, self.model_manager, self.refresh_display)
        
    def load_selected_model(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a model to load.")
            return
            
        item = self.tree.item(selection[0])
        model_name = item['values'][0]
        
        if self.app_callback:
            self.app_callback("load_model", model_name)
        else:
            messagebox.showinfo("Model Selected", f"Selected model: {model_name}")
            
    def delete_selected_model(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a model to delete.")
            return
            
        item = self.tree.item(selection[0])
        model_name = item['values'][0]
        
        if messagebox.askyesno("Confirm Delete", 
                              f"Are you sure you want to delete the model '{model_name}'?\n\n"
                              "This will free up disk space but you'll need to download it again to use it."):
            if self.model_manager.delete_model(model_name):
                messagebox.showinfo("Success", f"Model '{model_name}' deleted successfully.")
                self.refresh_display()
                if self.app_callback:
                    self.app_callback("model_deleted", model_name)
            else:
                messagebox.showerror("Error", f"Failed to delete model '{model_name}'.")
                
    def clean_cache(self):
        if messagebox.askyesno("Clean Cache", 
                              "This will clean up temporary files and unused model cache.\n\n"
                              "Continue?"):
            try:
                # Clean up transformers cache
                import transformers
                cache_dir = transformers.file_utils.default_cache_path
                if os.path.exists(cache_dir):
                    import shutil
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                messagebox.showinfo("Success", "Cache cleaned successfully.")
                self.refresh_display()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clean cache: {str(e)}")

class ModelDownloadDialog:
    def __init__(self, parent, model_manager, refresh_callback):
        self.parent = parent
        self.model_manager = model_manager
        self.refresh_callback = refresh_callback
        self.setup_dialog()
        
    def setup_dialog(self):
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("📥 Download Model")
        self.dialog.geometry("500x400")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (self.parent.winfo_rootx() + 150, self.parent.winfo_rooty() + 100))
        
        # Header
        header_frame = ttk.Frame(self.dialog)
        header_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Label(header_frame, text="📥 Download AI Model", font=('Arial', 14, 'bold')).pack()
        
        # Model selection
        model_frame = ttk.LabelFrame(self.dialog, text="Select Model", padding="10")
        model_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.model_var = tk.StringVar(value="unsloth/gemma-3n-E4B-it")
        
        models = [
            "unsloth/gemma-3n-E4B-it",
            "unsloth/gemma-3n-8B-it", 
            "unsloth/gemma-3n-5B-it",
            "unsloth/gemma-3n-2B-it"
        ]
        
        for model in models:
            ttk.Radiobutton(model_frame, text=model, variable=self.model_var, value=model).pack(anchor=tk.W, pady=2)
            
        # Custom model entry
        custom_frame = ttk.Frame(model_frame)
        custom_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(custom_frame, text="Custom:", variable=self.model_var, value="custom").pack(side=tk.LEFT)
        self.custom_entry = ttk.Entry(custom_frame, width=30)
        self.custom_entry.pack(side=tk.LEFT, padx=5)
        
        # Progress
        progress_frame = ttk.LabelFrame(self.dialog, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.progress_var = tk.StringVar(value="Ready to download...")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        
        self.download_btn = ttk.Button(button_frame, text="📥 Download", command=self.start_download)
        self.download_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
    def start_download(self):
        model_name = self.model_var.get()
        if model_name == "custom":
            model_name = self.custom_entry.get().strip()
            if not model_name:
                messagebox.showwarning("Invalid Input", "Please enter a custom model name.")
                return
                
        if not self.model_manager.is_online():
            messagebox.showerror("No Internet", "Internet connection required to download models.")
            return
            
        if self.model_manager.is_model_downloaded(model_name):
            if not messagebox.askyesno("Model Exists", f"Model '{model_name}' is already downloaded.\n\nRe-download?"):
                return
                
        self.download_btn.config(state=tk.DISABLED)
        self.progress_bar.start()
        
        def download_thread():
            try:
                def progress_callback(message):
                    self.dialog.after(0, lambda: self.progress_var.set(message))
                    
                self.model_manager.download_model(model_name, progress_callback)
                
                self.dialog.after(0, self.download_complete)
                
            except Exception as e:
                self.dialog.after(0, lambda: self.download_error(str(e)))
                
        threading.Thread(target=download_thread, daemon=True).start()
        
    def download_complete(self):
        self.progress_bar.stop()
        self.progress_var.set("Download completed successfully!")
        messagebox.showinfo("Success", "Model downloaded and saved for offline use!")
        if self.refresh_callback:
            self.refresh_callback()
        self.dialog.destroy()
        
    def download_error(self, error_message):
        self.progress_bar.stop()
        self.progress_var.set(f"Download failed: {error_message}")
        self.download_btn.config(state=tk.NORMAL)
        messagebox.showerror("Download Failed", f"Failed to download model:\n\n{error_message}")

class HelpViewer:
    def __init__(self, parent):
        self.parent = parent
        self.setup_viewer()
        
    def setup_viewer(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title("📖 Educational AI Tutor - Help & Information")
        self.window.geometry("900x700")
        self.window.resizable(True, True)
        self.window.minsize(700, 500)
        
        # Header
        header_frame = ttk.Frame(self.window)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = ttk.Label(header_frame, text="📖 Educational AI Tutor - Complete Guide", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        version_label = ttk.Label(header_frame, text="v1.0 - Multimodal Learning Assistant", 
                                 font=('Arial', 10), foreground='gray')
        version_label.pack(side=tk.RIGHT)
        
        # Notebook for different help sections
        self.help_notebook = ttk.Notebook(self.window)
        self.help_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.setup_overview_tab()
        self.setup_offline_tab()
        self.setup_getting_started_tab()
        self.setup_educational_tools_tab()
        self.setup_multimodal_tab()
        self.setup_token_management_tab()
        self.setup_tips_tab()
        self.setup_troubleshooting_tab()
        
        # Close button
        close_frame = ttk.Frame(self.window)
        close_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(close_frame, text="Close Help", command=self.window.destroy).pack(side=tk.RIGHT)
        
    def setup_overview_tab(self):
        frame = ttk.Frame(self.help_notebook)
        self.help_notebook.add(frame, text="🏠 Overview")
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = """
🎓 EDUCATIONAL AI TUTOR - MULTIMODAL LEARNING ASSISTANT

Welcome to your comprehensive AI-powered learning companion! This application combines advanced AI with educational tools to create a personalized learning environment.

═══════════════════════════════════════════════════════════════════════════════

🌟 KEY FEATURES:

📚 Intelligent Tutoring
• Powered by Gemma 3n language model
• Adaptive responses based on your learning level
• Subject-specific expertise across multiple domains
• Step-by-step explanations and guided learning

🌐 Complete Offline Operation
• Download AI models once, use forever without internet
• Total privacy - all processing happens on your device
• Perfect for remote areas or limited connectivity
• No data usage after initial model download

🖼️ Multimodal Support
• Text conversations with advanced AI
• Image analysis and discussion
• Audio recording and file import
• Screen capture for visual problems
• Document analysis (PDF, DOCX, TXT, and more)

🎯 Educational Tools
• Concept explanation and mapping
• Homework assistance with hints (not answers!)
• Quiz generation and adaptive practice
• Assessment creation and skill evaluation
• Progress tracking and study recommendations

🔧 Smart Features
• 32K token context window with intelligent management
• Smart clipboard integration
• Real-time token usage monitoring
• Chat history management and export
• Resizable interface with multiple panels
• Advanced model management for offline use

═══════════════════════════════════════════════════════════════════════════════

💡 LEARNING PHILOSOPHY:

This tutor is designed to HELP YOU LEARN, not give you answers. It focuses on:
• Understanding concepts deeply
• Developing problem-solving skills
• Building confidence through practice
• Encouraging critical thinking
• Providing personalized guidance

🔒 PRIVACY & INDEPENDENCE:

Your learning is completely private and independent:
• All AI processing happens on YOUR device only
• No data ever sent to external servers
• Works completely offline after initial setup
• Perfect for sensitive educational content
• Student privacy fully protected

═══════════════════════════════════════════════════════════════════════════════

🎯 WHO CAN BENEFIT:

✅ Students (Elementary through Graduate level)
✅ Homeschool families
✅ Teachers seeking AI assistance
✅ Self-learners and lifelong learners
✅ Anyone wanting personalized education support

═══════════════════════════════════════════════════════════════════════════════

🚀 GET STARTED:

1. Download an AI model using the Model Manager (🤖 button)
2. Choose your study mode in the Educational Tools panel
3. Upload study materials or enable general subject settings
4. Start asking questions and exploring!
5. Everything works offline after model download!

Continue to the Offline Mode tab for complete setup instructions.
        """
        
        text_widget.insert(1.0, content)
        text_widget.config(state=tk.DISABLED)
        
    def setup_offline_tab(self):
        frame = ttk.Frame(self.help_notebook)
        self.help_notebook.add(frame, text="🌐 Offline Mode")
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = """
🌐 OFFLINE MODE - COMPLETE PRIVACY & INDEPENDENCE

Experience true privacy and freedom with offline AI that runs entirely on your device!

═══════════════════════════════════════════════════════════════════════════════

🚀 WHY OFFLINE MATTERS:

🔒 Complete Privacy
• Your conversations never leave your device
• No data sent to external servers
• Student information stays completely secure
• Perfect for sensitive educational content

⚡ Always Available
• Works without internet connection
• Perfect for remote areas or unreliable connections
• No downtime or server issues
• Ideal for schools with limited connectivity

🎯 Superior Performance
• Faster responses (no network latency)
• Consistent performance regardless of internet speed
• No bandwidth usage after initial download
• Works on planes, trains, and anywhere offline

═══════════════════════════════════════════════════════════════════════════════

🤖 MODEL MANAGER GUIDE

📥 Downloading Models:

1. Click "🤖 Model Manager" button (top of interface)
2. Click "📥 Download Model" in the manager window
3. Choose from available Gemma 3n models:
   • gemma-3n-E4B-it (Recommended - balanced performance)
   • gemma-3n-8B-it (Largest - best quality, needs more RAM)
   • gemma-3n-5B-it (Good balance of size and performance)
   • gemma-3n-2B-it (Smallest - fastest, lower quality)
4. Click "📥 Download" and wait for completion
5. Model is now available for offline use!

📱 Model Sizes & Requirements:
• 2B Model: ~1.5GB disk space, 4GB+ RAM recommended
• 4B Model: ~2.5GB disk space, 6GB+ RAM recommended  
• 5B Model: ~3.5GB disk space, 8GB+ RAM recommended
• 8B Model: ~5GB disk space, 12GB+ RAM recommended

🎛️ Model Management:
• View all downloaded models and their sizes
• Switch between models instantly
• Delete unused models to free space
• Monitor total storage usage
• Clean cache and temporary files

═══════════════════════════════════════════════════════════════════════════════

⚙️ OFFLINE MODE SETTINGS

🌐 Offline Mode Toggle:
• Enable: Only use locally downloaded models
• Disable: Download models automatically if needed
• Status shown in Model Manager interface
• Setting saved between sessions

🔄 Connection Status:
• Green 🟢: Internet available
• Red 🔴: Offline (no internet)
• Model Manager shows current connection status
• Works regardless of internet availability

📊 Storage Management:
• Models stored in: ~/.educational_ai_tutor/models/
• Config saved in: ~/.educational_ai_tutor/config.json
• View total storage usage in Model Manager
• Clean cache to free temporary files

═══════════════════════════════════════════════════════════════════════════════

🎯 OFFLINE WORKFLOW

📋 First-Time Setup:
1. Connect to internet (for initial download)
2. Open Model Manager (🤖 button)
3. Download your preferred model (E4B-it recommended)
4. Enable "🌐 Offline Mode" checkbox
5. Close Model Manager - you're now fully offline!

🔄 Daily Usage:
1. Start the application (no internet needed)
2. Model loads automatically from local storage
3. All features work exactly the same
4. Chat, upload documents, use educational tools
5. Everything processed locally on your device

📚 Educational Benefits:
• Students can study anywhere without internet
• Teachers can use in classrooms without WiFi concerns
• Complete privacy for sensitive educational content
• No data usage - perfect for limited internet plans
• Consistent performance regardless of network

═══════════════════════════════════════════════════════════════════════════════

🛠️ TECHNICAL DETAILS

🏗️ Architecture:
• Models run using Unsloth + Transformers locally
• 4-bit quantization for memory efficiency
• CPU and GPU acceleration support
• Optimized for consumer hardware

💾 Storage Structure:
~/.educational_ai_tutor/
├── models/
│   ├── unsloth_gemma-3n-E4B-it/
│   └── unsloth_gemma-3n-8B-it/
└── config.json

🎛️ Configuration Options:
• downloaded_models: List of available local models
• last_used_model: Automatically load this model
• offline_mode: Force offline-only operation

═══════════════════════════════════════════════════════════════════════════════

⚠️ TROUBLESHOOTING OFFLINE ISSUES

❌ "No models downloaded" Error:
1. Connect to internet temporarily
2. Open Model Manager
3. Download at least one model
4. Enable offline mode
5. Disconnect internet - will work offline

❌ Model won't load offline:
1. Check Model Manager - ensure model shows as "Downloaded"
2. Try loading a different model
3. Clear cache and restart application
4. Re-download model if corrupted

❌ "Internet required" message:
1. Enable "🌐 Offline Mode" in Model Manager
2. Ensure you have downloaded models locally
3. Restart application after enabling offline mode

💡 Optimization Tips:
• Download models during off-peak hours
• Use external drive for models if space limited
• 8GB+ RAM recommended for larger models
• SSD storage recommended for faster loading

═══════════════════════════════════════════════════════════════════════════════

🎓 EDUCATIONAL USE CASES

🏫 Classroom Deployment:
• Install on multiple computers with one internet download
• Students work offline during lessons
• No network congestion issues
• Perfect for computer labs

📱 Personal Study:
• Study on planes, trains, remote locations
• No data usage concerns
• Complete privacy for personal learning
• Works during internet outages

🌍 Global Education:
• Serve areas with limited internet
• Reduce digital divide
• Educational access without connectivity requirements
• Perfect for developing regions

Remember: Once downloaded, the AI tutor works completely offline with full functionality - it's like having a personal AI teacher that never needs the internet!
        """
        
        text_widget.insert(1.0, content)
        text_widget.config(state=tk.DISABLED)
        
    def setup_getting_started_tab(self):
        frame = ttk.Frame(self.help_notebook)
        self.help_notebook.add(frame, text="🚀 Getting Started")
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = """
🚀 GETTING STARTED GUIDE

Follow these steps to begin your learning journey:

═══════════════════════════════════════════════════════════════════════════════

📋 STEP 1: DOWNLOAD AI MODEL (First Time Only)

🤖 Model Setup:
1. Click the "🤖 Model Manager" button (top of interface)
2. Click "📥 Download Model" in the manager window
3. Select recommended model: "unsloth/gemma-3n-E4B-it"
4. Click "📥 Download" and wait for completion (~2-3GB download)
5. Enable "🌐 Offline Mode" checkbox for complete privacy
6. Close Model Manager - you're now ready for offline use!

💡 Model Recommendations:
• For most users: gemma-3n-E4B-it (balanced performance)
• For powerful computers: gemma-3n-8B-it (best quality)
• For older computers: gemma-3n-2B-it (fastest)

⚠️ One-time internet required: After download, works completely offline!

═══════════════════════════════════════════════════════════════════════════════

📋 STEP 2: FAMILIARIZE WITH INTERFACE
   • Left panel: Educational Tools
   • Center panel: Chat and Input
   • Right panel: Context & History

═══════════════════════════════════════════════════════════════════════════════

⚙️ STEP 2: CHOOSE YOUR STUDY MODE

🎯 Option A: Study Materials Only (Recommended for specific coursework)
• Keep "Use General Subject Settings" unchecked
• Upload your textbooks, notes, or study materials
• AI will focus exclusively on your uploaded content

🎯 Option B: General Subject Settings
• Check "Use General Subject Settings"
• Select subject, grade level, and difficulty
• AI will provide general knowledge in that subject area

═══════════════════════════════════════════════════════════════════════════════

📄 STEP 3: UPLOAD STUDY MATERIALS (if using Option A)

1. Click "Add Document" button
2. Select your files (supports many formats):
   • PDF files (with page selection)
   • Word documents (.docx)
   • Text files (.txt, .md)
   • Code files (.py, .js, .css, etc.)
   • Web files (.html, .xml, .json)

3. Use the Document Viewer:
   • Navigate through pages (for PDFs)
   • Select specific pages or use all content
   • Zoom in/out for better readability
   • Click "Use All Text" or "Use Selected"

═══════════════════════════════════════════════════════════════════════════════

💬 STEP 4: ADD MULTIMEDIA CONTENT (Optional)

📸 Images:
• Click "Add Image" to select image files
• Use screen capture for screenshots
• Copy images to clipboard for auto-detection

🎵 Audio:
• Click "Add Audio" to import existing audio files
• Click "Record Audio" for live recording
• Supports MP3, WAV, M4A, AAC, OGG, FLAC, WMA formats

📄 Documents:
• Click "Add Document" for study materials
• Supports PDF, DOCX, TXT, MD, RTF, HTML, code files
• Use Document Viewer for page selection

💬 STEP 5: START LEARNING

1. Type your question in the message input area
2. Use Ctrl+Enter to send (or click Send button)
3. Watch the AI provide detailed explanations
4. Ask follow-up questions to deepen understanding

Examples of good questions:
• "Explain the concept of photosynthesis in simple terms"
• "Can you break down this math problem step by step?"
• "What are the key themes in this chapter?"
• "Create a concept map for cellular respiration"

═══════════════════════════════════════════════════════════════════════════════

🎓 STEP 6: USE EDUCATIONAL TOOLS

Instead of typing questions, try the pre-built educational tools:

📚 Learning Tools:
• "Explain Concept" - Get detailed explanations
• "Create Concept Map" - Visual learning aids

📝 Homework Helper:
• "Get Hints" - Guidance without giving answers
• "Step-by-Step Guide" - Detailed problem-solving
• "Find Similar Problems" - Practice variations

🧪 Quiz & Practice:
• "Generate Quiz" - Custom quizzes based on material
• "Adaptive Practice" - Progressive difficulty sessions

═══════════════════════════════════════════════════════════════════════════════

🔍 MONITORING YOUR PROGRESS

• Check the Token Usage panel for context tracking
• Use "Chat History" to review past conversations
• Monitor "Progress Overview" in Educational Tools
• Export chat history for later review

═══════════════════════════════════════════════════════════════════════════════

⌨️ KEYBOARD SHORTCUTS

• Ctrl+Enter: Send message
• Escape: Stop AI generation
• Mouse wheel: Scroll through chat history
• Click pages in PDF viewer to select them

═══════════════════════════════════════════════════════════════════════════════

💡 FIRST SESSION TIPS:

1. Start with simple questions to get comfortable
2. Upload one document at a time initially
3. Monitor token usage to understand limits
4. Experiment with different educational tools
5. Ask the AI to explain its own responses if unclear

Ready to learn? Head to the Educational Tools tab for more detailed feature explanations!
        """
        
        text_widget.insert(1.0, content)
        text_widget.config(state=tk.DISABLED)
        
    def setup_educational_tools_tab(self):
        frame = ttk.Frame(self.help_notebook)
        self.help_notebook.add(frame, text="🎓 Educational Tools")
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = """
🎓 EDUCATIONAL TOOLS DETAILED GUIDE

The Educational Tools panel provides specialized learning functions designed to enhance your study experience.

═══════════════════════════════════════════════════════════════════════════════

⚙️ STUDY SETTINGS

🎯 Study Mode Selection:
• Materials Only: Focus on your uploaded documents
• General Settings: Use AI's broad knowledge base
• Toggle anytime to switch between modes

📊 Subject Configuration (when General Settings enabled):
• Subject: Choose from 14+ subject areas
• Grade Level: Elementary → Graduate
• Difficulty: Beginner → Expert
• These settings customize AI responses to your level

═══════════════════════════════════════════════════════════════════════════════

📚 LEARNING TOOLS

◯ Explain Concept
PURPOSE: Get comprehensive explanations of key topics
WHEN TO USE: When you need to understand a new concept
WHAT IT DOES:
• Identifies important concepts in your subject/materials
• Provides clear, level-appropriate definitions
• Gives real-world examples and applications
• Breaks down complex ideas step-by-step
• Highlights common misconceptions
• Suggests memory techniques
• Provides practice questions

EXAMPLE: Click this for a physics concept and get detailed explanations of force, acceleration, practical examples, common mistakes, and practice problems.

◈ Create Concept Map
PURPOSE: Visual learning through structured concept relationships
WHEN TO USE: When you need to see how ideas connect
WHAT IT DOES:
• Creates hierarchical concept maps
• Shows relationships between main and sub-topics
• Includes connections to related concepts
• Provides prerequisites and follow-up topics
• Uses tree-structure visualization

EXAMPLE: Creates a visual map showing "Photosynthesis" connected to "Light Reactions," "Calvin Cycle," "Chloroplasts," etc.

═══════════════════════════════════════════════════════════════════════════════

📝 HOMEWORK HELPER

⚠️ IMPORTANT: These tools provide GUIDANCE, not direct answers!

? Get Hints
PURPOSE: Gentle guidance without spoiling the learning process
WHEN TO USE: When stuck on homework problems
WHAT IT PROVIDES:
• Guiding questions to help you think
• Relevant concepts to review
• Similar example problems
• Step-by-step approaches (without answers)
• Common mistakes to avoid

EXAMPLE: For a math problem, gives you questions like "What formula might apply here?" rather than showing the solution.

∞ Step-by-Step Guide
PURPOSE: Learn the problem-solving process
WHEN TO USE: When you need to understand methodology
WHAT IT PROVIDES:
• Detailed step-by-step breakdowns
• Explanation of WHY each step is taken
• Reasoning behind each action
• Multiple examples of the same process
• Connections between steps

EXAMPLE: Shows "Step 1: Identify the type of equation (and why this matters), Step 2: Apply the quadratic formula (and when to use it)..."

◎ Find Similar Problems
PURPOSE: Practice with variations of the same concept
WHEN TO USE: When you want more practice
WHAT IT PROVIDES:
• 3-5 similar practice problems
• Same concept with different numbers/contexts
• Progressive difficulty levels
• Solution approaches for each
• Concept reinforcement notes

═══════════════════════════════════════════════════════════════════════════════

🧪 QUIZ & PRACTICE

Quiz Options:
• Questions: 1-20 (customizable)
• Types: Multiple Choice, True/False, Fill in Blank, Short Answer, Mixed

◦ Generate Quiz
PURPOSE: Test your knowledge
WHEN TO USE: Before exams or to check understanding
WHAT IT PROVIDES:
• Custom questions based on your materials
• Multiple choice options with explanations
• Correct answer identification
• Detailed explanations for all options
• Progressive difficulty levels

↻ Adaptive Practice
PURPOSE: Personalized learning progression
WHEN TO USE: For systematic skill building
WHAT IT PROVIDES:
• Starts with easier confidence-building questions
• Gradually increases difficulty
• Immediate feedback after each question
• Questions that build on previous answers
• Customized pacing based on performance

═══════════════════════════════════════════════════════════════════════════════

📊 ASSESSMENT & REVIEW

✎ Create Practice Test
PURPOSE: Exam preparation
FEATURES:
• Mix of question types
• Comprehensive topic coverage
• Detailed answer keys
• Point values and time suggestions
• Study tips for difficult concepts

⌚ Timed Assessment
PURPOSE: Simulate real exam conditions
FEATURES:
• 20-minute timed sessions
• Strategic question distribution (60% quick recall, 30% problem-solving, 10% analysis)
• Time management guidance
• Pacing strategies
• Performance scoring

◦ Skill Evaluation
PURPOSE: Comprehensive skill assessment
FEATURES:
• Four-area evaluation framework
• Diagnostic questions for each skill level
• Performance indicators (Beginner/Intermediate/Advanced)
• Targeted improvement recommendations
• Next steps for skill development

═══════════════════════════════════════════════════════════════════════════════

🔍 STUDY ANALYSIS

◯ Get Study Recommendations
PURPOSE: Personalized improvement strategies
PROVIDES:
• Areas needing more focus
• Effective study techniques
• Methods to deepen understanding
• Next topics to explore
• Information retention strategies

◦ Identify My Strengths
PURPOSE: Recognize and build on natural abilities
ANALYZES:
• Learning style preferences
• Areas of highest engagement
• Question patterns and preferences
• Problem-solving approaches
• Natural talents and inclinations

═══════════════════════════════════════════════════════════════════════════════

📈 PROGRESS OVERVIEW

The mini statistics display shows:
• Current study mode and settings
• Study materials token count
• Chat activity summary
• Questions asked vs responses received
• Total conversation tokens

💡 PRO TIP: Use this regularly to track your learning journey and adjust study strategies!
        """
        
        text_widget.insert(1.0, content)
        text_widget.config(state=tk.DISABLED)
        
    def setup_multimodal_tab(self):
        frame = ttk.Frame(self.help_notebook)
        self.help_notebook.add(frame, text="🖼️ Multimodal Features")
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = """
🖼️ MULTIMODAL FEATURES GUIDE

The AI tutor supports multiple input types beyond text, making learning more interactive and comprehensive.

═══════════════════════════════════════════════════════════════════════════════

📄 DOCUMENT SUPPORT

🔍 Supported Formats:
• PDF files (.pdf) - Full viewer with page selection
• Word documents (.docx) - Text extraction
• Text files (.txt, .md) - Direct import
• Rich text (.rtf) - Formatted text
• Web files (.html, .css, .js) - Code analysis
• Programming files (.py, .xml, .json) - Syntax understanding

📋 Document Viewer Features:

For PDF Files:
• Navigation: ◀ ▶ buttons to move between pages
• Zoom: 🔍+ and 🔍- for better readability
• Page Selection: Click on pages to select/deselect them
• Range Selection: Enter page numbers (e.g., "1 to 5")
• Visual Feedback: Selected pages highlighted in yellow

For All Files:
• System Integration: "Open in System" button
• Text Preview: See content before importing
• Selective Import: Choose specific sections
• Token Counting: See token usage before adding

📚 How to Use Documents:
1. Click "Add Document" in the main interface
2. Select your file using the file browser
3. Use the Document Viewer to preview and select content
4. Click "Use All Text" or "Use Selected" to import
5. The content appears in the Context panel
6. Ask questions about the imported material

💡 BEST PRACTICES:
• Upload one document at a time initially
• For large PDFs, select only relevant pages
• Monitor token usage when importing large texts
• Use descriptive file names for organization

═══════════════════════════════════════════════════════════════════════════════

🖼️ IMAGE SUPPORT

📷 Image Input Methods:
• File Selection: Click "Add Image" to browse for files
• Clipboard Auto-Detection: Copy images and they're automatically added
• Screen Capture: Click "Capture Screen" for selective screenshots

🎯 Supported Image Types:
• Photos (.jpg, .jpeg) - Pictures of textbooks, whiteboards, notes
• Graphics (.png, .gif, .bmp) - Diagrams, charts, screenshots
• Any visual content relevant to your studies

📋 Image Processing:
• Automatic token allocation (256 tokens per image)
• Visual analysis and description capabilities
• Integration with text conversations
• Support for multiple images per conversation

💡 EDUCATIONAL USE CASES:
• Photograph textbook pages or problems
• Capture whiteboard content from classes
• Screenshot online resources or websites
• Include diagrams and charts in discussions
• Share visual homework problems

⚠️ Note: Images are stored locally and processed for educational analysis.

═══════════════════════════════════════════════════════════════════════════════

🎤 AUDIO SUPPORT

🔊 Audio Input Methods:
• File Selection: Click "Add Audio" to browse for existing audio files
• Real-time Recording: Click "Record Audio" for live recording
• Multiple format support for imported files

🎵 Supported Audio Formats:
• MP3 (.mp3) - Most common format
• WAV (.wav) - Uncompressed audio
• M4A (.m4a) - Apple/iTunes format
• AAC (.aac) - Advanced Audio Codec
• OGG (.ogg) - Open source format
• FLAC (.flac) - Lossless compression
• WMA (.wma) - Windows Media Audio

📋 Audio File Import:
1. Click "Add Audio" button
2. Select one or more audio files from your computer
3. Files are analyzed for duration and format
4. Token count calculated automatically
5. Files added to attachment list

🔊 Audio Recording Features:
• Real-time recording with visual feedback
• Automatic file conversion and storage
• Duration tracking and token calculation
• Integration with text conversations

📋 How to Use Audio Recording:
1. Click "Record Audio" to start recording
2. Speak your question or provide audio content
3. Button changes to "Stop Recording" with red status
4. Click again to stop and save the recording
5. Audio is automatically added to your attachments

⏱️ Token Usage:
• Calculated at 6.25 tokens per second
• Real-time duration tracking for recordings
• Automatic duration analysis for imported files
• Displayed in attachment summary

💡 EDUCATIONAL APPLICATIONS:
• Import lecture recordings for analysis
• Add audio notes and explanations
• Include pronunciation examples for language learning
• Process recorded problem explanations
• Analyze audio from educational videos
• Record yourself explaining concepts for AI feedback
• Import podcast segments or audio books
• Add voice memos and study recordings

🎵 Technical Details:
• Sample Rate: 44.1 kHz
• Format: MP3 compression
• Channels: Mono recording
• Quality: Optimized for speech recognition

═══════════════════════════════════════════════════════════════════════════════

📱 SCREEN CAPTURE

🖥️ Screen Capture Features:
• Full-screen overlay with crosshair cursor
• Click and drag to select specific areas
• Real-time selection preview
• Automatic crop and save functionality

📋 How to Use Screen Capture:
1. Click "Capture Screen" button
2. Application minimizes temporarily
3. Click and drag to select the area you want
4. Release mouse to capture the selection
5. Application returns with captured image added

⌨️ Keyboard Controls:
• Escape: Cancel capture and return to application
• Mouse: Click, drag, and release to select area

💡 EDUCATIONAL USE CASES:
• Capture online quiz questions or problems
• Screenshot educational websites or resources
• Grab specific diagrams from digital textbooks
• Capture error messages for troubleshooting
• Save visual content from educational videos

🎯 Selection Tips:
• Be precise with your selection rectangle
• Capture just the relevant content
• Avoid including unnecessary UI elements
• Consider image clarity and readability

═══════════════════════════════════════════════════════════════════════════════

🔄 SMART CLIPBOARD INTEGRATION

📋 Automatic Detection:
• Monitors clipboard for new images
• Automatically adds copied images
• Provides notifications when images are detected
• Seamless integration with other applications

💡 Workflow Integration:
• Copy images from web browsers
• Paste screenshots from other applications
• Import images from photo editing software
• Seamless workflow with research activities

⚙️ How It Works:
• Runs in background continuously
• Checks clipboard every second
• Detects new image content
• Adds images with system notifications
• Prevents duplicate additions

═══════════════════════════════════════════════════════════════════════════════

📊 ATTACHMENT MANAGEMENT

📝 Attachment Tracking:
• Real-time token count for all attachments
• Type categorization (images, audio, documents)
• Total token usage calculation
• Clear attachment summary display

🗂️ Organization Features:
• Automatic file naming and storage
• Temporary file management
• Cleanup on application exit
• Attachment history tracking

💡 OPTIMIZATION TIPS:
• Monitor total attachment tokens
• Remove unnecessary attachments
• Use selective document imports
• Balance multimodal input with token limits

Remember: All multimodal inputs are designed to enhance your learning experience while maintaining focus on educational outcomes!
        """
        
        text_widget.insert(1.0, content)
        text_widget.config(state=tk.DISABLED)
        
    def setup_token_management_tab(self):
        frame = ttk.Frame(self.help_notebook)
        self.help_notebook.add(frame, text="🔢 Token Management")
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = """
🔢 TOKEN MANAGEMENT COMPREHENSIVE GUIDE

Understanding and managing tokens is crucial for optimal performance with the AI tutor.

═══════════════════════════════════════════════════════════════════════════════

📖 WHAT ARE TOKENS?

🔤 Token Basics:
• Tokens are units of text that the AI processes
• Roughly 1 token = 0.75 words in English
• Numbers, punctuation, and spaces count as tokens
• The AI has a 32,000 token context limit

💡 Examples:
• "Hello world!" = ~2 tokens
• "The quick brown fox" = ~4 tokens
• "Photosynthesis is the process..." = ~5 tokens

⚡ Why Tokens Matter:
• They determine how much context the AI can remember
• Larger inputs require more tokens
• Token limits prevent information overload
• Efficient token use = better performance

═══════════════════════════════════════════════════════════════════════════════

📊 TOKEN USAGE BREAKDOWN

📝 Text Tokens:
• Your input messages
• AI responses
• Context from uploaded documents
• Chat history

🖼️ Image Tokens:
• Each image = 256 tokens (fixed)
• Regardless of image size or complexity
• Counted when image is added to conversation

🎵 Audio Tokens:
• Calculated at 6.25 tokens per second
• 1 minute of audio = ~375 tokens
• Based on audio duration, not content

📄 Document Tokens:
• Variable based on text length
• Calculated using advanced tokenization
• Shown before importing documents

═══════════════════════════════════════════════════════════════════════════════

🎛️ TOKEN DISPLAY INTERFACE

📈 Token Usage Panel (Top of Interface):
Shows real-time breakdown:
• Total: Current total tokens / 32,000 limit (percentage)
• Remaining: How many tokens are still available
• Input: Tokens from your current input + context
• History: Tokens from recent chat messages

🚨 Color-Coded Warnings:
• 🟢 Green (0-50%): Safe usage levels
• 🟡 Yellow (50-75%): Monitor usage
• 🟠 Orange (75-90%): Approach caution
• 🔴 Red (90%+): Near limit, take action

📊 Individual Token Counters:
• Input field: Shows tokens as you type
• Context panel: Updates when editing context
• History panel: Shows tokens per message
• Attachments: Displays token cost

═══════════════════════════════════════════════════════════════════════════════

⚙️ TOKEN MANAGEMENT CONTROLS

🎚️ Max Output Tokens:
• Controls AI response length
• Range: 128 - 2,048 tokens
• Default: 512 tokens
• Shorter = quicker responses, Longer = detailed responses

📜 Chat History Messages:
• Controls how many previous messages to include
• Range: 1 - 50 messages
• Default: 10 messages
• More history = better context, More tokens used

🔄 Refresh Functions:
• "Refresh Tokens": Recalculate all token counts
• "Chat History": View detailed token breakdown
• Auto-updates when content changes

═══════════════════════════════════════════════════════════════════════════════

🛠️ OPTIMIZATION STRATEGIES

📉 Reducing Token Usage:

1. History Management:
   • Reduce "Messages to include" count
   • Clear chat history periodically
   • Export important conversations before clearing

2. Context Optimization:
   • Use only relevant sections of documents
   • Edit context to remove unnecessary parts
   • Clear context when switching topics

3. Input Efficiency:
   • Be concise but clear in questions
   • Avoid repeating information
   • Use educational tool buttons instead of long prompts

4. Attachment Management:
   • Remove unnecessary images/audio
   • Use selective PDF page imports
   • Monitor attachment token totals

📈 Maximizing Token Efficiency:

1. Smart Document Use:
   • Import only relevant pages/sections
   • Use Document Viewer's selection features
   • Preview token count before importing

2. Conversation Strategy:
   • Start with specific questions
   • Build on previous responses
   • Use follow-up questions effectively

3. Educational Tool Usage:
   • Use pre-built prompts for common tasks
   • Educational tools are optimized for token efficiency
   • Combine multiple learning objectives

═══════════════════════════════════════════════════════════════════════════════

⚠️ MANAGING TOKEN LIMITS

🚨 When Approaching Limits:

1. Immediate Actions:
   • Clear old chat history
   • Remove unnecessary context
   • Reduce max output tokens temporarily

2. Context Strategies:
   • Save important context to files
   • Use "Save Context" before clearing
   • Load context as needed for specific topics

3. Conversation Management:
   • Export chat history before clearing
   • Start fresh conversations for new topics
   • Use "Clear Chat" for complete reset

📋 Best Practices:

✅ DO:
• Monitor token usage regularly
• Use educational tools efficiently
• Save important conversations
• Clear context between topics
• Import only relevant document sections

❌ DON'T:
• Upload entire textbooks at once
• Keep unnecessary attachments
• Ignore token warnings
• Use maximum settings unnecessarily
• Repeat information in conversations

═══════════════════════════════════════════════════════════════════════════════

🔍 TROUBLESHOOTING TOKEN ISSUES

❌ "Token Limit Exceeded" Error:
1. Check total token usage in display
2. Clear some chat history
3. Remove large context sections
4. Reduce attachment count
5. Try again with reduced input

🐛 Inaccurate Token Counts:
1. Click "Refresh Tokens" button
2. Restart conversation if needed
3. Check for hidden characters in context
4. Clear and reload problematic content

⚡ Performance Issues:
1. Keep total tokens under 75% of limit
2. Use shorter output token limits
3. Manage history more aggressively
4. Consider breaking large tasks into smaller parts

═══════════════════════════════════════════════════════════════════════════════

📚 ADVANCED TOKEN TECHNIQUES

🎯 Efficient Learning Workflows:

1. Topic-Based Sessions:
   • Dedicate conversations to single topics
   • Clear context between subjects
   • Use educational tools for structured learning

2. Progressive Learning:
   • Start with basic concepts (fewer tokens)
   • Build complexity gradually
   • Use previous responses as context

3. Document Strategies:
   • Process large documents in sections
   • Use page selection for relevant content
   • Combine related sections efficiently

💡 Pro Tips:
• Educational tool prompts are pre-optimized
• Context editing can significantly reduce tokens
• Regular token monitoring prevents issues
• Save important insights before clearing

Remember: Effective token management enhances your learning experience while maintaining AI performance!
        """
        
        text_widget.insert(1.0, content)
        text_widget.config(state=tk.DISABLED)
        
    def setup_tips_tab(self):
        frame = ttk.Frame(self.help_notebook)
        self.help_notebook.add(frame, text="💡 Tips & Best Practices")
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = """
💡 TIPS & BEST PRACTICES

Master the Educational AI Tutor with these proven strategies and techniques.

═══════════════════════════════════════════════════════════════════════════════

🎯 EFFECTIVE LEARNING STRATEGIES

📚 Question Formulation:

✅ GOOD Questions:
• "Explain the concept of photosynthesis and why it's important"
• "Break down this quadratic equation step by step"
• "What are the main themes in Chapter 3 and how do they connect?"
• "Create a concept map for cellular respiration"
• "What's the difference between mitosis and meiosis?"

❌ AVOID:
• "Do my homework" (too vague)
• "What's the answer to problem 5?" (seeking answers, not understanding)
• "Tell me everything about history" (too broad)
• Single word questions without context

🎓 Learning Progression:
1. Start with basic concept explanations
2. Ask for examples and real-world applications
3. Request practice problems or quizzes
4. Seek connections between concepts
5. Test understanding with assessments

═══════════════════════════════════════════════════════════════════════════════

📋 STUDY SESSION OPTIMIZATION

⏰ Session Planning:

1. Focused Sessions (30-60 minutes):
   • Choose one subject/topic per session
   • Upload relevant materials at the start
   • Use educational tools systematically
   • End with quiz or assessment

2. Review Sessions (15-30 minutes):
   • Load previous context
   • Ask for concept summaries
   • Request practice questions
   • Identify areas needing more work

3. Assessment Sessions (45-90 minutes):
   • Use timed assessments and practice tests
   • Focus on comprehensive evaluation
   • Request study recommendations
   • Plan next learning steps

🎯 Pre-Session Checklist:
□ Clear previous context if changing topics
□ Upload relevant study materials
□ Set appropriate subject/grade settings
□ Monitor token usage from the start
□ Have specific learning objectives

═══════════════════════════════════════════════════════════════════════════════

📚 DOCUMENT MANAGEMENT STRATEGIES

📄 Efficient Document Use:

1. Textbook Strategy:
   • Upload one chapter at a time
   • Use page selection for specific topics
   • Focus on key concepts and examples
   • Combine with educational tools

2. Note Organization:
   • Convert handwritten notes to text files
   • Upload study guides and summaries
   • Include problem sets and solutions
   • Organize by topic or date

3. Multi-Source Learning:
   • Combine textbook + lecture notes + assignments
   • Use selective imports for each source
   • Monitor token usage across sources
   • Create comprehensive context gradually

💡 Document Best Practices:
• Name files descriptively
• Use PDF page selection effectively
• Preview content before importing
• Save context frequently
• Clear and reload for new topics

🎵 Audio Best Practices:
• Use MP3 or WAV formats for best compatibility
• Keep audio files under 10 minutes for optimal processing
• Import lecture recordings in segments for better analysis
• Record in quiet environments for better quality
• Use descriptive file names for organization

═══════════════════════════════════════════════════════════════════════════════

🔧 TECHNICAL OPTIMIZATION

⚡ Performance Tips:

1. Token Management:
   • Keep total usage under 75% of limit
   • Use 256-512 token outputs for most tasks
   • Clear history every 20-30 exchanges
   • Monitor attachment token costs

2. Response Quality:
   • Be specific in questions
   • Provide context when needed
   • Use educational tools for structured tasks
   • Ask follow-up questions for clarification

3. Interface Usage:
   • Use keyboard shortcuts (Ctrl+Enter, Escape)
   • Organize panels for your workflow
   • Save important conversations
   • Export chat history regularly

🎛️ Settings Optimization:
• Set grade level accurately for appropriate responses
• Choose difficulty level honestly for best results
• Adjust max output tokens based on task complexity
• Use history message count strategically

═══════════════════════════════════════════════════════════════════════════════

🎓 SUBJECT-SPECIFIC STRATEGIES

🔬 STEM Subjects (Math, Science, Engineering):
• Upload problem sets and work through solutions
• Request step-by-step explanations
• Use concept mapping for complex topics
• Generate practice problems for repetition
• Focus on understanding WHY, not just HOW

📚 Humanities (History, Literature, Philosophy):
• Upload reading materials and primary sources
• Request thematic analysis and connections
• Use essay structure guidance
• Ask for historical context and significance
• Generate discussion questions

🗣️ Languages (English, Foreign Languages):
• Upload text excerpts for analysis
• Request grammar explanations and examples
• Practice with conversation scenarios
• Use vocabulary building exercises
• Focus on practical application

💼 Applied Subjects (Business, Economics, etc.):
• Upload case studies and real-world examples
• Request practical application scenarios
• Use problem-solving frameworks
• Generate situational practice problems
• Connect theory to current events

═══════════════════════════════════════════════════════════════════════════════

🚀 ADVANCED TECHNIQUES

🎯 Multi-Modal Learning:
• Combine text, images, and audio for complex topics
• Import existing audio files (lectures, recordings, etc.)
• Use screen capture for online resources
• Record yourself explaining concepts for AI feedback
• Include diagrams and charts in discussions

📈 Progressive Skill Building:
1. Assessment Phase: Use skill evaluation tools
2. Foundation Phase: Focus on basic concepts
3. Application Phase: Practice with varied problems
4. Integration Phase: Connect multiple concepts
5. Mastery Phase: Teach back to the AI

🔄 Iterative Learning:
• Start with broad overview questions
• Dive deeper into specific areas
• Return to connections and relationships
• Test understanding with assessments
• Identify gaps and repeat cycle

═══════════════════════════════════════════════════════════════════════════════

⚠️ COMMON PITFALLS TO AVOID

❌ Don't:
• Use AI as a simple answer generator
• Upload massive documents without selection
• Ignore token warnings
• Ask for direct homework answers
• Skip understanding checks

✅ Do:
• Focus on learning processes
• Manage context and tokens actively
• Use educational tools systematically
• Ask for explanations and reasoning
• Test your understanding regularly

═══════════════════════════════════════════════════════════════════════════════

🏆 SUCCESS INDICATORS

You're using the AI tutor effectively when:
✅ You understand concepts better after sessions
✅ You can explain topics to others
✅ You're asking deeper, more sophisticated questions
✅ You're making connections between different concepts
✅ You feel more confident in your subject knowledge
✅ You're using multiple educational tools naturally
✅ You can solve similar problems independently

═══════════════════════════════════════════════════════════════════════════════

📱 WORKFLOW INTEGRATION

🔗 Daily Learning Routine:
1. Morning: Review previous session notes
2. During Study: Use AI for concept clarification
3. After Classes: Upload new materials for analysis
4. Evening: Generate quizzes for self-testing
5. Weekly: Use assessment tools for progress tracking

🎯 Exam Preparation:
1. Upload all relevant materials
2. Create comprehensive concept maps
3. Generate practice tests and quizzes
4. Use timed assessments for practice
5. Request study recommendations
6. Focus on identified weak areas

Remember: The AI tutor is most effective when you actively engage with the learning process rather than passively consuming information!
        """
        
        text_widget.insert(1.0, content)
        text_widget.config(state=tk.DISABLED)
        
    def setup_troubleshooting_tab(self):
        frame = ttk.Frame(self.help_notebook)
        self.help_notebook.add(frame, text="🔧 Troubleshooting")
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = """
🔧 TROUBLESHOOTING GUIDE

Solutions for common issues and technical problems with the Educational AI Tutor.

═══════════════════════════════════════════════════════════════════════════════

🚨 STARTUP ISSUES

❌ Problem: "Model loading failed" error
🔧 Solutions:
1. Check internet connection (required for initial model download)
2. Ensure sufficient disk space (2-4 GB needed)
3. Restart the application
4. Check if firewall is blocking downloads
5. Try running as administrator (Windows) or with sudo (Linux)

❌ Problem: Application won't start
🔧 Solutions:
1. Verify Python installation (3.8+ required)
2. Install missing dependencies manually:
   pip install torch transformers pillow numpy sounddevice soundfile pyperclip PyMuPDF python-docx tiktoken unsloth
3. Check for system compatibility
4. Review error messages in console/terminal

❌ Problem: Slow model loading
🔧 Expected Behavior:
• First launch: 2-5 minutes (downloading model)
• Subsequent launches: 30-60 seconds
• GPU systems: Faster loading
• CPU-only systems: Slower but functional

═══════════════════════════════════════════════════════════════════════════════

💬 CONVERSATION ISSUES

❌ Problem: AI responses are cut off or incomplete
🔧 Solutions:
1. Increase "Max Output Tokens" setting (try 1024 or 1536)
2. Check if generation was stopped accidentally
3. Reduce input length to allow more output space
4. Clear some chat history to free up context

❌ Problem: AI responses don't relate to uploaded documents
🔧 Solutions:
1. Verify document was properly imported (check Context panel)
2. Ensure "Study Materials Only" mode is selected
3. Check if context was accidentally cleared
4. Re-upload document and try again
5. Ask more specific questions about document content

❌ Problem: Generation seems stuck or very slow
🔧 Solutions:
1. Click the Stop button (⏹ STOP) or press Escape
2. Wait a moment and try again
3. Reduce max output tokens
4. Check system resources (CPU/Memory usage)
5. Restart application if problem persists

═══════════════════════════════════════════════════════════════════════════════

📄 DOCUMENT HANDLING ISSUES

❌ Problem: PDF won't open in Document Viewer
🔧 Solutions:
1. Ensure file isn't password-protected
2. Try a different PDF reader to verify file integrity
3. Check file permissions
4. Try converting to a different format
5. Use text extraction software if needed

❌ Problem: Text extraction is garbled or incorrect
🔧 Solutions:
1. Try using OCR software for scanned documents
2. Copy-paste text manually if possible
3. Convert document to plain text format
4. Check document encoding (try UTF-8)
5. Use alternative document format

❌ Problem: Document Viewer crashes
🔧 Solutions:
1. Check file size (very large files may cause issues)
2. Close other applications to free memory
3. Try processing document in smaller sections
4. Restart application and try again
5. Use system document viewer as alternative

═══════════════════════════════════════════════════════════════════════════════

🔢 TOKEN MANAGEMENT ISSUES

❌ Problem: Token count seems incorrect
🔧 Solutions:
1. Click "Refresh Tokens" button
2. Clear and re-add context if needed
3. Check for hidden characters or formatting
4. Restart conversation if counts remain wrong
5. Export and clear history, then restart

❌ Problem: Reaching token limits too quickly
🔧 Solutions:
1. Reduce "Messages to include" in history
2. Clear old chat history more frequently
3. Use selective document imports (specific pages)
4. Edit context to remove unnecessary content
5. Use shorter, more focused inputs

❌ Problem: Token warnings not appearing
🔧 Solutions:
1. Check if token display is visible
2. Refresh token counts manually
3. Restart application if display seems frozen
4. Monitor usage manually with export feature

═══════════════════════════════════════════════════════════════════════════════

🖼️ MULTIMEDIA ISSUES

❌ Problem: Images not being detected from clipboard
🔧 Solutions:
1. Try copying image again
2. Check if image format is supported
3. Restart clipboard monitoring (restart app)
4. Use "Add Image" button as alternative
5. Check system clipboard permissions

❌ Problem: Screen capture not working
🔧 Solutions:
1. Check screen capture permissions (macOS/Linux)
2. Try alternative: use system screenshot tool + clipboard
3. Ensure no other screen capture software is interfering
4. Restart application and try again
5. Use "Add Image" to manually select screenshots

❌ Problem: Audio recording fails
🔧 Solutions:
1. Check microphone permissions
2. Verify audio device is connected and working
3. Test with system audio recorder first
4. Check if other applications are using microphone
5. Restart application and try again

❌ Problem: Audio file import fails
🔧 Solutions:
1. Check if audio format is supported (MP3, WAV, M4A, AAC, OGG, FLAC, WMA)
2. Verify file is not corrupted (test in media player)
3. Check file permissions and accessibility
4. Try converting to MP3 or WAV format
5. Ensure file size is reasonable (under 100MB recommended)

❌ Problem: Audio duration detection incorrect
🔧 Solutions:
1. Check if file format is fully supported
2. File may be corrupted - test in audio player
3. Application will use file size estimation as fallback
4. Convert to standard format (MP3/WAV) for better detection
5. Manual token estimation: ~6.25 tokens per second

═══════════════════════════════════════════════════════════════════════════════

🎓 EDUCATIONAL TOOL ISSUES

❌ Problem: Educational tools generate generic responses
🔧 Solutions:
1. Upload more specific study materials
2. Enable "General Subject Settings" if no materials uploaded
3. Set appropriate grade level and difficulty
4. Be more specific in your study focus
5. Try different educational tools for variety

❌ Problem: Quiz questions don't match study materials
🔧 Solutions:
1. Verify study materials are loaded in Context panel
2. Ensure "Study Materials Only" mode is active
3. Check that uploaded materials contain relevant content
4. Try regenerating quiz with different parameters
5. Use more focused document sections

❌ Problem: Assessment tools seem too easy/hard
🔧 Solutions:
1. Adjust difficulty setting in Study Settings
2. Change grade level to match your needs
3. Upload more appropriate study materials
4. Use skill evaluation to determine proper level
5. Customize quiz parameters manually

═══════════════════════════════════════════════════════════════════════════════

💻 PERFORMANCE ISSUES

❌ Problem: Application running slowly
🔧 Solutions:
1. Close unnecessary applications to free memory
2. Reduce max output tokens
3. Clear chat history more frequently
4. Limit number of attachments
5. Restart application periodically

❌ Problem: Interface becomes unresponsive
🔧 Solutions:
1. Wait for current operation to complete
2. Use Stop button if AI is generating
3. Try clicking in different interface areas
4. Restart application if completely frozen
5. Check system resource usage

❌ Problem: High memory usage
🔧 Solutions:
1. Clear chat history regularly
2. Remove large attachments
3. Restart application to clear memory
4. Close unused document viewers
5. Monitor token usage to prevent overload

═══════════════════════════════════════════════════════════════════════════════

🔄 DATA MANAGEMENT ISSUES

❌ Problem: Lost chat history
🔧 Prevention:
1. Export important conversations regularly
2. Save context before clearing
3. Use "Chat History" viewer to backup data
4. Don't rely on application memory for long-term storage

❌ Problem: Context disappeared
🔧 Solutions:
1. Check if accidentally cleared
2. Re-upload documents if needed
3. Load saved context files
4. Verify Context panel is visible
5. Check if switched study modes

❌ Problem: Can't export chat history
🔧 Solutions:
1. Try different file formats (JSON vs TXT)
2. Choose different save location
3. Check file permissions in target directory
4. Close other applications that might lock files
5. Save in smaller sections if large

═══════════════════════════════════════════════════════════════════════════════

⚙️ SYSTEM COMPATIBILITY

🖥️ Operating System Issues:
• Windows: Ensure Windows 10+ for best compatibility
• macOS: Requires macOS 10.14+ and appropriate permissions
• Linux: Most distributions supported, may need additional packages

🔧 Hardware Requirements:
• RAM: 8GB minimum, 16GB recommended
• Storage: 5GB free space for models and cache
• CPU: Modern multi-core processor recommended
• GPU: Optional but improves performance significantly

📱 Resolution and Display:
• Minimum: 1024x768 resolution
• Recommended: 1920x1080 or higher
• Multiple monitors: Supported
• High DPI: May require system scaling adjustments

═══════════════════════════════════════════════════════════════════════════════

🆘 GETTING ADDITIONAL HELP

📞 When to Seek Help:
• Persistent crashes or errors
• Model fails to load repeatedly
• Data corruption or loss
• Performance severely degraded
• Features completely non-functional

📋 Information to Provide:
• Operating system and version
• Error messages (exact text)
• Steps to reproduce the problem
• System specifications
• Recent changes to system or software

💡 Self-Help Resources:
• Check console/terminal for detailed error messages
• Review system requirements
• Update system and drivers
• Try running with administrator/sudo privileges
• Test with minimal configurations

═══════════════════════════════════════════════════════════════════════════════

🔒 PRIVACY AND SECURITY

🛡️ Data Handling:
• All processing happens locally on your device
• No data sent to external servers (except initial model download)
• Temporary files cleaned up on exit
• Chat history stored locally only

🔐 File Security:
• Uploaded documents processed locally
• Temporary files use system security
• No persistent storage of sensitive content
• User controls all data retention

Remember: Most issues can be resolved by restarting the application or clearing temporary data. When in doubt, try the simplest solution first!
        """
        
        text_widget.insert(1.0, content)
        text_widget.config(state=tk.DISABLED)

class ChatHistoryViewer:
    def __init__(self, parent, chat_history, token_manager):
        self.parent = parent
        self.chat_history = chat_history
        self.token_manager = token_manager
        self.setup_viewer()
        self.update_display()
        
    def setup_viewer(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title("Chat History & Token Usage")
        self.window.geometry("700x600")
        self.window.resizable(True, True)
        self.window.minsize(500, 400)
        
        toolbar = ttk.Frame(self.window)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="Clear History", command=self.clear_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Export History", command=self.export_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Refresh", command=self.update_display).pack(side=tk.LEFT, padx=5)
        
        self.tree = ttk.Treeview(self.window, columns=('Role', 'Tokens', 'Content'), show='tree headings')
        self.tree.heading('#0', text='#')
        self.tree.heading('Role', text='Role')
        self.tree.heading('Tokens', text='Tokens')
        self.tree.heading('Content', text='Content Preview')
        
        self.tree.column('#0', width=50)
        self.tree.column('Role', width=80)
        self.tree.column('Tokens', width=80)
        self.tree.column('Content', width=400)
        
        scrollbar = ttk.Scrollbar(self.window, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        summary_frame = ttk.Frame(self.window)
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.summary_label = ttk.Label(summary_frame, text="", font=('Arial', 10, 'bold'))
        self.summary_label.pack()
        
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        
    def update_display(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        total_tokens = 0
        for i, entry in enumerate(self.chat_history):
            if isinstance(entry, dict) and 'role' in entry:
                role = entry['role']
                content = str(entry.get('content', ''))
                
                tokens = 0
                if 'tokens' in entry:
                    tokens = entry['tokens']
                else:
                    tokens = self.token_manager.count_text_tokens(content)
                
                preview = content[:60] + "..." if len(content) > 60 else content
                preview = preview.replace('\n', ' ')
                
                total_tokens += tokens
                
                self.tree.insert('', tk.END, text=str(i+1), values=(role, tokens, preview))
        
        usage_percent = (total_tokens / self.token_manager.max_context_tokens) * 100
        color = self.token_manager.get_color_for_usage(total_tokens)
        
        self.summary_label.config(
            text=f"Total History: {total_tokens:,} tokens ({usage_percent:.1f}% of 32K limit)",
            foreground=color
        )
        
    def clear_history(self):
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all chat history?"):
            self.chat_history.clear()
            self.update_display()
            
    def export_history(self):
        file_path = filedialog.asksaveasfilename(
            title="Export Chat History",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
        )
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.chat_history, f, indent=2, ensure_ascii=False)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for entry in self.chat_history:
                            f.write(f"{entry.get('role', 'unknown').upper()}: {entry.get('content', '')}\n\n")
                messagebox.showinfo("Export Complete", f"History exported to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Could not export history: {str(e)}")

class ScreenCapture:
    def __init__(self, callback):
        self.callback = callback
        self.screenshot = ImageGrab.grab()
        
        self.root = tk.Toplevel()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.bind('<Button-1>', self.on_click)
        self.root.bind('<B1-Motion>', self.on_drag)
        self.root.bind('<ButtonRelease-1>', self.on_release)
        self.root.bind('<Escape>', lambda e: self.root.destroy())

        self.tk_image = ImageTk.PhotoImage(self.screenshot)

        self.canvas = tk.Canvas(self.root, cursor='cross')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

        self.start_x = self.start_y = None
        self.rect = None

    def on_click(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect:
            self.canvas.delete(self.rect)

    def on_drag(self, event):
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)
        else:
            self.rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline='red', width=2, fill='white', stipple='gray25'
            )

    def on_release(self, event):
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        cropped_img = self.screenshot.crop((x1, y1, x2, y2))
        self.root.destroy()
        self.callback(cropped_img)

class ModelThread(threading.Thread):
    def __init__(self, model, tokenizer, messages, response_queue, max_tokens=512):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.messages = messages
        self.response_queue = response_queue
        self.max_tokens = max_tokens
        self.daemon = True
        self.stop_generation = threading.Event()
        
    def stop(self):
        self.stop_generation.set()
        
    def run(self):
        class StreamCapture(TextStreamer):
            def __init__(self, tokenizer, queue, stop_event):
                super().__init__(tokenizer, skip_prompt=True)
                self.queue = queue
                self.stop_event = stop_event
                
            def on_finalized_text(self, text, stream_end=False):
                if self.stop_event.is_set():
                    return
                cleaned_text = text.replace('<end_of_turn>', '').replace('<|end_of_text|>', '').replace('<|endoftext|>', '').replace('<start_of_turn>', '')
                if cleaned_text:
                    self.queue.put(('chunk', cleaned_text))
                    
            def put(self, value):
                if self.stop_event.is_set():
                    return
                super().put(value)
                
        try:
            if self.stop_generation.is_set():
                return
                
            if not self.model or not self.tokenizer:
                self.response_queue.put(('error', "Model or tokenizer not loaded"))
                return
            
            conversation_text = ""
            for msg in self.messages:
                if self.stop_generation.is_set():
                    return
                    
                if not isinstance(msg, dict):
                    continue
                    
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    content = ' '.join(text_parts)
                elif not isinstance(content, str):
                    content = str(content)
                    
                if content.strip():
                    if role == 'user':
                        conversation_text += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                    else:
                        conversation_text += f"<start_of_turn>model\n{content}<end_of_turn>\n"
            
            conversation_text += "<start_of_turn>model\n"
            
            if not conversation_text.strip() or self.stop_generation.is_set():
                if not self.stop_generation.is_set():
                    self.response_queue.put(('error', "No valid messages to process"))
                return
                
            print(f"Conversation format: {conversation_text[:200]}...")
            
            try:
                if hasattr(self.tokenizer, 'tokenizer'):
                    actual_tokenizer = self.tokenizer.tokenizer
                else:
                    actual_tokenizer = self.tokenizer
                
                encoding = actual_tokenizer(conversation_text, return_tensors="pt", add_special_tokens=False, padding=False)
                input_ids = encoding['input_ids']
                attention_mask = encoding.get('attention_mask', torch.ones_like(input_ids))
                
                if input_ids is None or self.stop_generation.is_set():
                    if not self.stop_generation.is_set():
                        raise Exception("Tokenization returned None")
                    return
                    
                print(f"Manual tokenization successful, shape: {input_ids.shape}")
                
                if torch.cuda.is_available():
                    input_ids = input_ids.to("cuda")
                    attention_mask = attention_mask.to("cuda")
                
                with torch.no_grad():
                    if self.stop_generation.is_set():
                        return
                        
                    streamer = StreamCapture(actual_tokenizer, self.response_queue, self.stop_generation)
                    
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_tokens,
                        temperature=0.7,
                        top_p=0.95,
                        top_k=64,
                        do_sample=True,
                        pad_token_id=actual_tokenizer.pad_token_id if hasattr(actual_tokenizer, 'pad_token_id') and actual_tokenizer.pad_token_id is not None else actual_tokenizer.eos_token_id,
                        eos_token_id=actual_tokenizer.eos_token_id,
                        streamer=streamer
                    )
                
                print(f"Generation completed successfully")
                
            except Exception as manual_error:
                if self.stop_generation.is_set():
                    return
                print(f"Manual tokenization error: {manual_error}")
                
                try:
                    simple_text = ""
                    for msg in self.messages:
                        role = msg.get('role', 'user').capitalize()
                        content = str(msg.get('content', ''))
                        simple_text += f"{role}: {content}\n"
                    simple_text += "Assistant:"
                    
                    if hasattr(self.tokenizer, 'tokenizer'):
                        encoding = self.tokenizer.tokenizer(simple_text, return_tensors="pt", padding=False)
                    else:
                        encoding = self.tokenizer(simple_text, return_tensors="pt", padding=False)
                    
                    input_ids = encoding['input_ids']
                    attention_mask = encoding.get('attention_mask', torch.ones_like(input_ids))
                    
                    if torch.cuda.is_available():
                        input_ids = input_ids.to("cuda")
                        attention_mask = attention_mask.to("cuda")
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=self.max_tokens,
                            temperature=0.7,
                            top_p=0.95,
                            top_k=64,
                            do_sample=True
                        )
                        
                        if outputs is not None and len(outputs) > 0:
                            new_tokens = outputs[0][input_ids.shape[1]:]
                            if hasattr(self.tokenizer, 'tokenizer'):
                                response_text = self.tokenizer.tokenizer.decode(new_tokens, skip_special_tokens=True)
                            else:
                                response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                            
                            if response_text.strip():
                                self.response_queue.put(('chunk', response_text))
                        
                except Exception as final_error:
                    raise Exception(f"All tokenization methods failed: {manual_error}, {final_error}")
            
            if 'input_ids' in locals():
                del input_ids
            if 'attention_mask' in locals():
                del attention_mask
            if 'outputs' in locals():
                del outputs
            torch.cuda.empty_cache()
            gc.collect()
            
            if not self.stop_generation.is_set():
                self.response_queue.put(('complete', None))
            else:
                self.response_queue.put(('stopped', None))
                
        except Exception as e:
            if not self.stop_generation.is_set():
                error_msg = f"Generation failed: {str(e)}"
                print(f"ModelThread error: {error_msg}")
                self.response_queue.put(('error', error_msg))

class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("◆ Gemma AI Educational Suite")
        
        self.root.geometry("1600x900")
        
        self.root.resizable(True, True)
        self.root.wm_resizable(True, True)
        
        try:
            self.root.wm_attributes('-type', 'normal')
        except:
            pass
        
        self.token_manager = TokenManager()
        self.model_manager = ModelManager()
        
        self.attachments = []
        self.chat_context = ""
        self.context_pages = []
        self.chat_history = []
        self.history_messages_var = tk.StringVar(value="10")
        self.recording = False
        self.audio_data = []
        self.response_queue = queue.Queue()
        self.model_loaded = False
        self.max_output_tokens = 512
        self.current_generation_thread = None
        self.is_generating = False
        self.model = None
        self.tokenizer = None
        
        self.current_difficulty = "Intermediate"
        self.current_grade_level = "High School"
        
        self.setup_ui()
        self.setup_clipboard_monitor()
        self.check_response_queue()
        
        self.root.bind('<Escape>', lambda e: self.stop_generation() if self.is_generating else None)
        self.root.bind('<F1>', lambda e: self.show_help())
        
        self.root.after(100, self.init_model_async)
        
        self.root.after(200, self.ensure_resizable)
        
    def ensure_resizable(self):
        try:
            self.root.resizable(True, True)
            self.root.wm_resizable(True, True)
            
            try:
                self.root.wm_attributes('-type', 'normal')
            except:
                pass
                
            self.root.update()
            self.root.update_idletasks()
            
            current_geo = self.root.geometry()
            print(f"Current geometry: {current_geo}")
            print("Window resizability should now be enabled")
            
            self.root.minsize(500, 350)
            
        except Exception as e:
            print(f"Error ensuring resizability: {e}")
        
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.paned_window = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        self.setup_educational_panel()
        
        main_frame = ttk.Frame(self.paned_window, padding="10")
        self.paned_window.add(main_frame, weight=3)
        
        token_info_frame = ttk.LabelFrame(main_frame, text="Token Usage", padding="5")
        token_info_frame.grid(row=0, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.token_display = tk.Text(token_info_frame, height=2, wrap=tk.WORD, font=('Arial', 9), bg='#f0f8ff')
        self.token_display.pack(fill=tk.X)
        self.token_display.config(state=tk.DISABLED)
        
        token_controls = ttk.Frame(token_info_frame)
        token_controls.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(token_controls, text="Max Output Tokens:").pack(side=tk.LEFT)
        self.max_tokens_var = tk.StringVar(value=str(self.max_output_tokens))
        max_tokens_spinbox = ttk.Spinbox(token_controls, from_=128, to=2048, textvariable=self.max_tokens_var, width=8, increment=128)
        max_tokens_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        max_tokens_spinbox.bind('<KeyRelease>', self.update_max_tokens)
        
        ttk.Button(token_controls, text="Chat History", command=self.show_chat_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(token_controls, text="🤖 Model Manager", command=self.show_model_manager).pack(side=tk.LEFT, padx=5)
        ttk.Button(token_controls, text="📖 Help & Info", command=self.show_help).pack(side=tk.LEFT, padx=5)
        ttk.Button(token_controls, text="Refresh Tokens", command=self.update_token_display).pack(side=tk.LEFT, padx=5)
        
        chat_frame = ttk.LabelFrame(main_frame, text="Chat History", padding="5")
        chat_frame.grid(row=1, column=0, columnspan=5, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            height=20,
            font=('Arial', 10),
            bg='#f0f0f0'
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.chat_display.config(state=tk.DISABLED)
        
        self.chat_display.tag_config('user', foreground='#0084ff', font=('Arial', 10, 'bold'))
        self.chat_display.tag_config('assistant', foreground='#65676b', font=('Arial', 10, 'bold'))
        self.chat_display.tag_config('attachment', foreground='#808080', font=('Arial', 9, 'italic'))
        self.chat_display.tag_config('error', foreground='#ff0000', font=('Arial', 10, 'bold'))
        self.chat_display.tag_config('context', foreground='#9932cc', font=('Arial', 9, 'italic'))
        self.chat_display.tag_config('tokens', foreground='#666666', font=('Arial', 8))
        
        input_frame = ttk.LabelFrame(main_frame, text="Message Input", padding="5")
        input_frame.grid(row=2, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.input_text = tk.Text(input_frame, height=4, wrap=tk.WORD, font=('Arial', 10))
        self.input_text.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E))
        self.input_text.bind('<Control-Return>', lambda e: self.send_message())
        self.input_text.bind('<Escape>', lambda e: self.stop_generation() if self.is_generating else None)
        self.input_text.bind('<KeyRelease>', self.update_input_tokens)
        
        self.send_btn = ttk.Button(input_frame, text="Send\n(Ctrl+Enter)", command=self.send_message)
        self.send_btn.grid(row=0, column=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.stop_btn = ttk.Button(input_frame, text="⏹ Stop", command=self.stop_generation, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=4, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.input_token_label = ttk.Label(input_frame, text="Input: 0 tokens", font=('Arial', 8), foreground='gray')
        self.input_token_label.grid(row=1, column=0, columnspan=5, sticky=tk.W, pady=(2, 0))
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=5, pady=(0, 5), sticky=(tk.W, tk.E))
        button_frame.columnconfigure((0,1,2,3,4,5), weight=1)
        
        self.image_btn = ttk.Button(button_frame, text="Add Image", command=self.select_image)
        self.image_btn.grid(row=0, column=0, padx=2, sticky=(tk.W, tk.E))
        
        self.doc_btn = ttk.Button(button_frame, text="Add Document", command=self.select_document)
        self.doc_btn.grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        
        self.audio_file_btn = ttk.Button(button_frame, text="Add Audio", command=self.select_audio)
        self.audio_file_btn.grid(row=0, column=2, padx=2, sticky=(tk.W, tk.E))
        
        self.audio_btn = ttk.Button(button_frame, text="Record Audio", command=self.toggle_recording)
        self.audio_btn.grid(row=0, column=3, padx=2, sticky=(tk.W, tk.E))
        
        self.screen_btn = ttk.Button(button_frame, text="Capture Screen", command=self.capture_screen)
        self.screen_btn.grid(row=0, column=4, padx=2, sticky=(tk.W, tk.E))
        
        self.clear_btn = ttk.Button(button_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_btn.grid(row=0, column=5, padx=2, sticky=(tk.W, tk.E))
        
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=4, column=0, columnspan=5, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, text="Loading model...", foreground='orange')
        self.status_label.pack(side=tk.LEFT)
        
        self.attachment_label = ttk.Label(status_frame, text="", foreground='blue')
        self.attachment_label.pack(side=tk.RIGHT)
        
        context_frame = ttk.LabelFrame(self.paned_window, text="Context & History", padding="5")
        self.paned_window.add(context_frame, weight=1)
        
        try:
            self.paned_window.paneconfigure(main_frame, minsize=250)
            self.paned_window.paneconfigure(context_frame, minsize=150)
            self.paned_window.paneconfigure(self.edu_frame, minsize=150)
        except:
            pass
        
        self.context_notebook = ttk.Notebook(context_frame)
        self.context_notebook.pack(fill=tk.BOTH, expand=True)
        
        editable_frame = ttk.Frame(self.context_notebook)
        self.context_notebook.add(editable_frame, text="Editable Context")
        
        self.context_display = scrolledtext.ScrolledText(
            editable_frame,
            wrap=tk.WORD,
            height=20,
            font=('Arial', 9),
            bg='#f8f8ff'
        )
        self.context_display.pack(fill=tk.BOTH, expand=True)
        self.context_display.bind('<KeyRelease>', self.update_context_tokens)
        
        editable_controls = ttk.Frame(editable_frame)
        editable_controls.pack(fill=tk.X, pady=(5, 0))
        
        self.clear_context_btn = ttk.Button(editable_controls, text="Clear Context", command=self.clear_context)
        self.clear_context_btn.pack(side=tk.LEFT)
        
        self.save_context_btn = ttk.Button(editable_controls, text="Save Context", command=self.save_context)
        self.save_context_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        self.load_context_btn = ttk.Button(editable_controls, text="Load Context", command=self.load_context)
        self.load_context_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        self.context_info_label = ttk.Label(editable_controls, text="Context: 0 tokens", foreground='gray')
        self.context_info_label.pack(side=tk.RIGHT)
        
        history_frame = ttk.Frame(self.context_notebook)
        self.context_notebook.add(history_frame, text="Chat History")
        
        self.history_context_display = scrolledtext.ScrolledText(
            history_frame,
            wrap=tk.WORD,
            height=20,
            font=('Arial', 9),
            bg='#fff8f0',
            state=tk.DISABLED
        )
        self.history_context_display.pack(fill=tk.BOTH, expand=True)
        
        history_controls = ttk.Frame(history_frame)
        history_controls.pack(fill=tk.X, pady=(5, 0))
        
        self.refresh_history_btn = ttk.Button(history_controls, text="Refresh History", command=self.update_history_context)
        self.refresh_history_btn.pack(side=tk.LEFT)
        
        self.clear_history_btn = ttk.Button(history_controls, text="Clear History", command=self.clear_chat_history)
        self.clear_history_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Label(history_controls, text="Messages to include:").pack(side=tk.LEFT, padx=(10, 5))
        self.history_messages_var = tk.StringVar(value="10")
        history_spinbox = ttk.Spinbox(history_controls, from_=1, to=50, textvariable=self.history_messages_var, width=5)
        history_spinbox.pack(side=tk.LEFT)
        history_spinbox.bind('<KeyRelease>', lambda e: self.update_history_context())
        
        self.history_info_label = ttk.Label(history_controls, text="History: 0 tokens", foreground='gray')
        self.history_info_label.pack(side=tk.RIGHT)
        
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=0)
        main_frame.rowconfigure(2, weight=0)
        main_frame.rowconfigure(3, weight=0)
        main_frame.rowconfigure(4, weight=0)
        
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        input_frame.columnconfigure(0, weight=1)
        
        self.update_token_display()
        self.update_history_context()
        self.refresh_mini_progress()
        
        # Add welcome message
        self.add_system_message("🎓 Welcome to Educational AI Tutor! This app works completely offline once models are downloaded.", attachment=True)
        self.add_system_message("🤖 Click 'Model Manager' to download AI models for offline use. Click 'Help & Info' for complete usage guide.", attachment=True)
        
    def toggle_study_mode(self):
        if self.use_general_mode.get():
            for widget in self.subject_frame.winfo_children():
                widget.configure(state='normal')
        else:
            for widget in self.subject_frame.winfo_children():
                if isinstance(widget, (ttk.Combobox, ttk.Entry)):
                    widget.configure(state='disabled')
        
    def setup_educational_panel(self):
        self.edu_frame = ttk.Frame(self.paned_window, padding="5")
        self.paned_window.add(self.edu_frame, weight=1)
        
        header_frame = ttk.Frame(self.edu_frame)
        header_frame.pack(fill=tk.X, pady=(0, 5))
        
        title_label = ttk.Label(header_frame, text="◆ Educational Tools", font=('Arial', 12, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        ttk.Button(header_frame, text="📖 Help", command=self.show_help).pack(side=tk.RIGHT)
        
        self.edu_notebook = ttk.Notebook(self.edu_frame)
        self.edu_notebook.pack(fill=tk.BOTH, expand=True)
        
        self.setup_learning_practice_tab()
        self.setup_assessment_tab()
        
    def setup_learning_practice_tab(self):
        learning_frame = ttk.Frame(self.edu_notebook)
        self.edu_notebook.add(learning_frame, text="■ Learn & Practice")
        
        scrollable_frame = ttk.Frame(learning_frame)
        canvas = tk.Canvas(scrollable_frame)
        scrollbar = ttk.Scrollbar(scrollable_frame, orient="vertical", command=canvas.yview)
        scrollable_content = ttk.Frame(canvas)
        
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        def configure_canvas_window(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        scrollable_content.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_canvas_window)
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        scrollable_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollable_frame.columnconfigure(0, weight=1)
        scrollable_frame.rowconfigure(0, weight=1)
        
        settings_frame = ttk.LabelFrame(scrollable_content, text="⚙ Study Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=5, padx=5)
        
        mode_frame = ttk.Frame(settings_frame)
        mode_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.use_general_mode = tk.BooleanVar(value=False)
        mode_check = ttk.Checkbutton(mode_frame, text="Use General Subject Settings", 
                                   variable=self.use_general_mode, command=self.toggle_study_mode)
        mode_check.pack(side=tk.LEFT)
        
        ttk.Label(mode_frame, text="(When off: uses uploaded study materials only)", 
                 font=('Arial', 8), foreground='gray').pack(side=tk.LEFT, padx=(10, 0))
        
        self.subject_frame = ttk.Frame(settings_frame)
        self.subject_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        ttk.Label(self.subject_frame, text="Subject:").grid(row=0, column=0, sticky=tk.W, padx=2)
        self.subject_var = tk.StringVar(value="Mathematics")
        self.subject_combo = ttk.Combobox(self.subject_frame, textvariable=self.subject_var,
                                   values=["Mathematics", "Science", "Physics", "Chemistry", "Biology",
                                          "History", "Literature", "English", "Geography", "Economics",
                                          "Computer Science", "Art", "Music", "Philosophy"])
        self.subject_combo.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=2)
        
        ttk.Label(self.subject_frame, text="Grade Level:").grid(row=1, column=0, sticky=tk.W, padx=2)
        self.grade_var = tk.StringVar(value="High School")
        self.grade_combo = ttk.Combobox(self.subject_frame, textvariable=self.grade_var,
                                 values=["Elementary", "Middle School", "High School", "College", "Graduate"])
        self.grade_combo.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=2)
        
        ttk.Label(self.subject_frame, text="Difficulty:").grid(row=2, column=0, sticky=tk.W, padx=2)
        self.difficulty_var = tk.StringVar(value="Intermediate")
        self.difficulty_combo = ttk.Combobox(self.subject_frame, textvariable=self.difficulty_var,
                                      values=["Beginner", "Intermediate", "Advanced", "Expert"])
        self.difficulty_combo.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=2)
        
        self.subject_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(0, weight=1)
        
        learning_tools_frame = ttk.LabelFrame(scrollable_content, text="◘ Learning Tools", padding="5")
        learning_tools_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(learning_tools_frame, text="◯ Explain Concept", 
                  command=self.explain_concept).pack(fill=tk.X, pady=2, padx=2)
        ttk.Button(learning_tools_frame, text="◈ Create Concept Map", 
                  command=self.create_concept_map).pack(fill=tk.X, pady=2, padx=2)
        
        homework_frame = ttk.LabelFrame(scrollable_content, text="■ Homework Helper", padding="5")
        homework_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(homework_frame, text="? Get Hints", 
                  command=self.homework_hints).pack(fill=tk.X, pady=2, padx=2)
        ttk.Button(homework_frame, text="∞ Step-by-Step Guide", 
                  command=self.step_by_step_help).pack(fill=tk.X, pady=2, padx=2)
        ttk.Button(homework_frame, text="◎ Find Similar Problems", 
                  command=self.find_similar_problems).pack(fill=tk.X, pady=2, padx=2)
        
        quiz_frame = ttk.LabelFrame(scrollable_content, text="? Quiz & Practice", padding="5")
        quiz_frame.pack(fill=tk.X, pady=5, padx=5)
        
        quiz_options = ttk.Frame(quiz_frame)
        quiz_options.pack(fill=tk.X, pady=2)
        quiz_options.columnconfigure(3, weight=1)
        
        ttk.Label(quiz_options, text="Questions:").grid(row=0, column=0, sticky=tk.W)
        self.quiz_count_var = tk.StringVar(value="5")
        quiz_count_spin = ttk.Spinbox(quiz_options, from_=1, to=20, textvariable=self.quiz_count_var, width=5)
        quiz_count_spin.grid(row=0, column=1, padx=(5, 10), sticky=tk.W)
        
        ttk.Label(quiz_options, text="Type:").grid(row=0, column=2, sticky=tk.W)
        self.quiz_type_var = tk.StringVar(value="Multiple Choice")
        quiz_type_combo = ttk.Combobox(quiz_options, textvariable=self.quiz_type_var,
                                     values=["Multiple Choice", "True/False", "Fill in Blank", "Short Answer", "Mixed"])
        quiz_type_combo.grid(row=0, column=3, padx=5, sticky=(tk.W, tk.E))
        
        ttk.Button(quiz_frame, text="◦ Generate Quiz", 
                  command=self.generate_quiz).pack(fill=tk.X, pady=2, padx=2)
        ttk.Button(quiz_frame, text="↻ Adaptive Practice", 
                  command=self.adaptive_practice).pack(fill=tk.X, pady=2, padx=2)
        
        scrollable_content.columnconfigure(0, weight=1)
        
    def setup_assessment_tab(self):
        assessment_frame = ttk.Frame(self.edu_notebook)
        self.edu_notebook.add(assessment_frame, text="▲ Assess & Review")
        
        scrollable_frame = ttk.Frame(assessment_frame)
        canvas = tk.Canvas(scrollable_frame)
        scrollbar = ttk.Scrollbar(scrollable_frame, orient="vertical", command=canvas.yview)
        scrollable_content = ttk.Frame(canvas)
        
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        def configure_canvas_window(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        scrollable_content.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_canvas_window)
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        scrollable_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollable_frame.columnconfigure(0, weight=1)
        scrollable_frame.rowconfigure(0, weight=1)
        
        test_frame = ttk.LabelFrame(scrollable_content, text="▪ Assessments", padding="5")
        test_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(test_frame, text="✎ Create Practice Test", 
                  command=self.create_practice_test).pack(fill=tk.X, pady=2, padx=2)
        ttk.Button(test_frame, text="⌚ Timed Assessment", 
                  command=self.timed_assessment).pack(fill=tk.X, pady=2, padx=2)
        ttk.Button(test_frame, text="◦ Skill Evaluation", 
                  command=self.skill_assessment).pack(fill=tk.X, pady=2, padx=2)
        
        feedback_frame = ttk.LabelFrame(scrollable_content, text="◇ Study Analysis", padding="5")
        feedback_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(feedback_frame, text="◯ Get Study Recommendations", 
                  command=self.improvement_suggestions).pack(fill=tk.X, pady=2, padx=2)
        ttk.Button(feedback_frame, text="◦ Identify My Strengths", 
                  command=self.identify_strengths).pack(fill=tk.X, pady=2, padx=2)
        
        progress_frame = ttk.LabelFrame(scrollable_content, text="↗ Progress Overview", padding="5")
        progress_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.mini_stats_display = tk.Text(progress_frame, height=6, wrap=tk.WORD, 
                                        font=('Arial', 9), bg='#f8f8ff', state=tk.DISABLED)
        self.mini_stats_display.pack(fill=tk.BOTH, expand=True, padx=2)
        
        controls_frame = ttk.Frame(progress_frame)
        controls_frame.pack(fill=tk.X, pady=(5,0))
        controls_frame.columnconfigure(1, weight=1)
        
        ttk.Button(controls_frame, text="↻ Refresh Stats", 
                  command=self.refresh_mini_progress).grid(row=0, column=0, padx=2, sticky=tk.W)
        
        scrollable_content.columnconfigure(0, weight=1)
        
    def update_max_tokens(self, event=None):
        try:
            self.max_output_tokens = int(self.max_tokens_var.get())
        except:
            self.max_output_tokens = 512
            
    def update_input_tokens(self, event=None):
        text = self.input_text.get("1.0", tk.END).strip()
        tokens = self.token_manager.count_text_tokens(text)
        color = self.token_manager.get_color_for_usage(tokens)
        self.input_token_label.config(text=f"Input: {tokens} tokens", foreground=color)
        
    def update_context_tokens(self, event=None):
        context_text = self.context_display.get("1.0", tk.END).strip()
        tokens = self.token_manager.count_text_tokens(context_text)
        color = self.token_manager.get_color_for_usage(tokens)
        self.context_info_label.config(text=f"Context: {tokens} tokens", foreground=color)
        
    def calculate_total_input_tokens(self):
        text_tokens = self.token_manager.count_text_tokens(self.input_text.get("1.0", tk.END).strip())
        context_tokens = self.token_manager.count_text_tokens(self.context_display.get("1.0", tk.END).strip())
        
        try:
            max_messages = int(self.history_messages_var.get())
        except:
            max_messages = 10
        recent_history = self.chat_history[-max_messages:] if len(self.chat_history) > max_messages else self.chat_history
        history_tokens = sum(entry.get('tokens', 0) for entry in recent_history)
        
        image_count = sum(1 for a in self.attachments if a['type'] == 'image')
        image_tokens = self.token_manager.count_image_tokens(image_count)
        
        audio_tokens = 0
        for attachment in self.attachments:
            if attachment['type'] == 'audio':
                try:
                    audio_info = sf.info(attachment['audio'])
                    duration = audio_info.duration
                    audio_tokens += self.token_manager.count_audio_tokens(duration)
                except:
                    audio_tokens += 100
                    
        return text_tokens + context_tokens + history_tokens + image_tokens + audio_tokens
        
    def update_token_display(self):
        total_input = self.calculate_total_input_tokens()
        
        try:
            max_messages = int(self.history_messages_var.get())
        except:
            max_messages = 10
        recent_history = self.chat_history[-max_messages:] if len(self.chat_history) > max_messages else self.chat_history
        history_tokens = sum(entry.get('tokens', 0) for entry in recent_history)
        
        total_tokens = total_input
        remaining_tokens = self.token_manager.max_context_tokens - total_tokens
        usage_percent = (total_tokens / self.token_manager.max_context_tokens) * 100
        
        color = self.token_manager.get_color_for_usage(total_tokens)
        
        self.token_display.config(state=tk.NORMAL)
        self.token_display.delete(1.0, tk.END)
        
        status_text = f"Total: {total_tokens:,}/{self.token_manager.max_context_tokens:,} tokens ({usage_percent:.1f}%) | "
        status_text += f"Remaining: {remaining_tokens:,} | Input: {total_input - history_tokens:,} | History: {history_tokens:,}"
        
        if total_tokens > self.token_manager.max_context_tokens * 0.9:
            status_text += f"\nWARNING: Approaching token limit! Consider clearing history or reducing context."
        elif total_tokens > self.token_manager.max_context_tokens * 0.75:
            status_text += f"\nNotice: High token usage. Monitor context length."
            
        self.token_display.insert(1.0, status_text)
        self.token_display.tag_add('token_info', 1.0, tk.END)
        self.token_display.tag_config('token_info', foreground=color, font=('Arial', 9, 'bold'))
        self.token_display.config(state=tk.DISABLED)
        
    def stop_generation(self):
        if self.current_generation_thread and self.is_generating:
            self.current_generation_thread.stop()
            self.status_label.config(text="Stopping generation...", foreground='orange')
            
            def check_stopped():
                if self.current_generation_thread and self.current_generation_thread.is_alive():
                    self.root.after(100, check_stopped)
                else:
                    if self.is_generating:
                        self.status_label.config(text="Generation stopped", foreground='orange')
                        self.current_generation_thread = None
                        self.update_ui_for_generation(False)
            
            self.root.after(100, check_stopped)
            
    def show_chat_history(self):
        ChatHistoryViewer(self.root, self.chat_history, self.token_manager)
        
    def show_model_manager(self):
        ModelManagerViewer(self.root, self.model_manager, self.model_manager_callback)
        
    def model_manager_callback(self, action, data=None):
        if action == "load_model":
            self.load_specific_model(data)
        elif action == "model_deleted":
            if data == getattr(self, 'current_model_name', None):
                self.model_loaded = False
                self.model = None
                self.tokenizer = None
                self.status_label.config(text="Model deleted - please load a new model", foreground='orange')
        elif action == "offline_mode_changed":
            self.update_status_for_offline_mode()
            
    def load_specific_model(self, model_name):
        if self.is_generating:
            messagebox.showwarning("Generation Active", "Please stop current generation before changing models.")
            return
            
        self.model_loaded = False
        self.status_label.config(text=f"Loading {model_name}...", foreground='orange')
        
        def load_thread():
            try:
                def progress_callback(message):
                    self.root.after(0, lambda: self.status_label.config(text=message, foreground='blue'))
                    
                model, tokenizer = self.model_manager.load_model(model_name, progress_callback)
                
                self.root.after(0, lambda: self.model_load_complete(model, tokenizer, model_name))
                
            except Exception as e:
                self.root.after(0, lambda: self.model_load_error(str(e)))
                
        threading.Thread(target=load_thread, daemon=True).start()
        
    def model_load_complete(self, model, tokenizer, model_name):
        self.model = model
        self.tokenizer = tokenizer
        self.current_model_name = model_name
        self.model_loaded = True
        
        status_text = f"✅ {model_name.split('/')[-1]} loaded"
        if self.model_manager.config.get("offline_mode", False):
            status_text += " (Offline)"
        else:
            status_text += " (Online Ready)"
            
        self.status_label.config(text=status_text, foreground='green')
        self.add_system_message(f"🤖 Model {model_name} loaded successfully! {'(Offline Mode)' if self.model_manager.config.get('offline_mode', False) else '(Online Ready)'}")
        
    def model_load_error(self, error_message):
        self.status_label.config(text=f"Model loading failed: {error_message}", foreground='red')
        self.add_system_message(f"❌ Error loading model: {error_message}", error=True)
        
    def update_status_for_offline_mode(self):
        if self.model_loaded and hasattr(self, 'current_model_name'):
            status_text = f"✅ {self.current_model_name.split('/')[-1]} loaded"
            if self.model_manager.config.get("offline_mode", False):
                status_text += " (Offline)"
            else:
                status_text += " (Online Ready)"
            self.status_label.config(text=status_text, foreground='green')
        
    def show_help(self):
        HelpViewer(self.root)
            
    def update_ui_for_generation(self, generating=True):
        self.is_generating = generating
        if generating:
            self.send_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL, text="⏹ STOP (ESC)")
        else:
            self.send_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED, text="⏹ Stop")
        
    def update_history_context(self):
        try:
            max_messages = int(self.history_messages_var.get())
        except:
            max_messages = 10
            
        recent_history = self.chat_history[-max_messages:] if len(self.chat_history) > max_messages else self.chat_history
        
        self.history_context_display.config(state=tk.NORMAL)
        self.history_context_display.delete(1.0, tk.END)
        
        history_text = ""
        total_tokens = 0
        
        for entry in recent_history:
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')
            tokens = entry.get('tokens', 0)
            
            history_text += f"{role.upper()}: {content}\n\n"
            total_tokens += tokens
            
        self.history_context_display.insert(1.0, history_text)
        self.history_context_display.config(state=tk.DISABLED)
        
        color = self.token_manager.get_color_for_usage(total_tokens)
        self.history_info_label.config(text=f"History: {total_tokens} tokens ({len(recent_history)} messages)", foreground=color)
        
    def clear_chat_history(self):
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all chat history?"):
            self.chat_history.clear()
            self.update_history_context()
            self.update_token_display()
        
    def init_model_async(self):
        def load_model():
            try:
                # Check if we have a preferred model
                last_used_model = self.model_manager.config.get("last_used_model")
                if not last_used_model:
                    last_used_model = "unsloth/gemma-3n-E4B-it"
                    
                # Update status
                offline_mode = self.model_manager.config.get("offline_mode", False)
                is_online = self.model_manager.is_online()
                
                if offline_mode and not self.model_manager.is_model_downloaded(last_used_model):
                    self.root.after(0, lambda: self.status_label.config(
                        text="❌ Offline mode enabled but no models downloaded. Use Model Manager to download.", 
                        foreground='red'))
                    self.root.after(0, lambda: self.add_system_message(
                        "⚠️ Offline mode is enabled but no models are downloaded. Click '🤖 Model Manager' to download models for offline use.", 
                        error=True))
                    return
                    
                if not is_online and not self.model_manager.is_model_downloaded(last_used_model):
                    self.root.after(0, lambda: self.status_label.config(
                        text="❌ No internet connection and no local models. Cannot load AI model.", 
                        foreground='red'))
                    self.root.after(0, lambda: self.add_system_message(
                        "⚠️ No internet connection and no local models available. Connect to internet to download models.", 
                        error=True))
                    return
                
                def progress_callback(message):
                    self.root.after(0, lambda: self.status_label.config(text=message, foreground='blue'))
                
                self.root.after(0, lambda: self.status_label.config(text="Initializing AI model...", foreground='orange'))
                
                # Load the model
                model, tokenizer = self.model_manager.load_model(last_used_model, progress_callback)
                
                # Update UI on main thread
                self.root.after(0, lambda: self.model_load_complete(model, tokenizer, last_used_model))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.model_load_error(error_msg))
        
        threading.Thread(target=load_model, daemon=True).start()
        
    def setup_clipboard_monitor(self):
        self.last_clipboard = ""
        self.clipboard_images = set()
        self.check_clipboard()
        
    def check_clipboard(self):
        try:
            image = ImageGrab.grabclipboard()
            if image and id(image) not in self.clipboard_images:
                self.clipboard_images.add(id(image))
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    image.save(tmp.name)
                    self.attachments.append({"type": "image", "image": tmp.name})
                    self.update_attachment_label()
                    self.update_token_display()
                    self.add_system_message("Image from clipboard added", attachment=True)
        except:
            pass
        self.root.after(1000, self.check_clipboard)
        
    def add_system_message(self, message, error=False, attachment=False):
        self.chat_display.config(state=tk.NORMAL)
        if error:
            self.chat_display.insert(tk.END, f"\n[ERROR] {message}\n", 'error')
        elif attachment:
            self.chat_display.insert(tk.END, f"\n[{message}]\n", 'attachment')
        else:
            self.chat_display.insert(tk.END, f"\n[SYSTEM] {message}\n", 'attachment')
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
    def add_message(self, role, content, tokens=None):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"\n{role.upper()}: ", role.lower())
        self.chat_display.insert(tk.END, f"{content}")
        
        if tokens:
            self.chat_display.insert(tk.END, f" [{tokens} tokens]", 'tokens')
            
        self.chat_display.insert(tk.END, "\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        entry = {"role": role, "content": content, "timestamp": time.time()}
        if tokens:
            entry["tokens"] = tokens
        else:
            entry["tokens"] = self.token_manager.count_text_tokens(content)
        self.chat_history.append(entry)
        
        if role == "user":
            self.update_history_context()
        
    def send_message(self):
        if not self.model_loaded:
            messagebox.showwarning("Model Loading", "Please wait for the model to finish loading.")
            return
            
        if self.is_generating:
            messagebox.showwarning("Generation Active", "Please wait for the current generation to complete or stop it first.")
            return
            
        total_input_tokens = self.calculate_total_input_tokens()
        if total_input_tokens > self.token_manager.max_context_tokens:
            messagebox.showerror("Token Limit Exceeded", 
                               f"Input tokens ({total_input_tokens:,}) exceed the 32K limit. Please reduce context or attachments.")
            return
            
        text = self.input_text.get("1.0", tk.END).strip()
        context_text = self.context_display.get("1.0", tk.END).strip()
        
        if not text and not context_text:
            messagebox.showwarning("Empty Message", "Please enter a message or add context.")
            return
            
        full_message = text
        if context_text:
            full_message = f"Based on the context below, please answer: {text}\n\nContext:\n{context_text}"
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.insert(tk.END, f"\n[Chat context included in query]\n", 'context')
            self.chat_display.config(state=tk.DISABLED)
            
        input_tokens = self.token_manager.count_text_tokens(full_message)
        self.add_message("user", text, input_tokens)
        
        messages = []
        
        try:
            max_messages = int(self.history_messages_var.get())
        except:
            max_messages = 10
            
        recent_history = self.chat_history[:-1]
        if len(recent_history) > max_messages:
            recent_history = recent_history[-max_messages:]
            
        for entry in recent_history:
            role = entry.get('role', 'user')
            content = entry.get('content', '')
            if content.strip():
                messages.append({"role": role, "content": content})
        
        if full_message:
            messages.append({"role": "user", "content": full_message})
            
        if not messages:
            messages.append({"role": "user", "content": text or "Hello"})
            
        if self.attachments:
            self.add_system_message(f"Note: {len(self.attachments)} attachment(s) stored but not sent to model (format compatibility)", attachment=True)
        
        self.input_text.delete("1.0", tk.END)
        self.attachments = []
        self.update_attachment_label()
        self.update_token_display()
        self.update_history_context()
        
        self.current_generation_thread = ModelThread(self.model, self.tokenizer, messages, self.response_queue, self.max_output_tokens)
        self.current_generation_thread.start()
        
        self.update_ui_for_generation(True)
        self.add_message("assistant", "", 0)
        self.status_label.config(text="Generating response...", foreground='blue')
        
    def check_response_queue(self):
        try:
            while True:
                msg_type, content = self.response_queue.get_nowait()
                if msg_type == 'chunk':
                    self.chat_display.config(state=tk.NORMAL)
                    self.chat_display.insert(tk.END, content)
                    self.chat_display.see(tk.END)
                    self.chat_display.config(state=tk.DISABLED)
                    
                    if self.chat_history and self.chat_history[-1]['role'] == 'assistant':
                        self.chat_history[-1]['content'] += content
                        
                elif msg_type == 'complete':
                    self.chat_display.config(state=tk.NORMAL)
                    
                    if self.chat_history and self.chat_history[-1]['role'] == 'assistant':
                        final_content = self.chat_history[-1]['content']
                        response_tokens = self.token_manager.count_text_tokens(final_content)
                        self.chat_history[-1]['tokens'] = response_tokens
                        self.chat_display.insert(tk.END, f" [{response_tokens} tokens]")
                        
                    self.chat_display.insert(tk.END, "\n")
                    self.chat_display.config(state=tk.DISABLED)
                    self.status_label.config(text="Ready", foreground='green')
                    self.update_token_display()
                    self.update_history_context()
                    self.current_generation_thread = None
                    self.update_ui_for_generation(False)
                    
                elif msg_type == 'stopped':
                    self.chat_display.config(state=tk.NORMAL)
                    
                    if self.chat_history and self.chat_history[-1]['role'] == 'assistant':
                        final_content = self.chat_history[-1]['content']
                        if final_content.strip():
                            response_tokens = self.token_manager.count_text_tokens(final_content)
                            self.chat_history[-1]['tokens'] = response_tokens
                            self.chat_display.insert(tk.END, f" [{response_tokens} tokens] [STOPPED]")
                        else:
                            self.chat_history.pop()
                            
                    self.chat_display.insert(tk.END, "\n")
                    self.chat_display.config(state=tk.DISABLED)
                    self.status_label.config(text="Generation stopped", foreground='orange')
                    self.update_token_display()
                    self.update_history_context()
                    self.current_generation_thread = None
                    self.update_ui_for_generation(False)
                    
                elif msg_type == 'error':
                    self.add_system_message(f"Generation error: {content}", error=True)
                    self.status_label.config(text="Error occurred", foreground='red')
                    self.current_generation_thread = None
                    self.update_ui_for_generation(False)
        except queue.Empty:
            pass
        self.root.after(50, self.check_response_queue)
        
    def select_image(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        for file_path in file_paths:
            self.attachments.append({"type": "image", "image": file_path})
            self.add_system_message(f"Image added: {os.path.basename(file_path)} (256 tokens)", attachment=True)
        self.update_attachment_label()
        self.update_token_display()
        
    def select_audio(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Audio Files",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.aac *.ogg *.flac *.wma"),
                ("MP3 files", "*.mp3"),
                ("WAV files", "*.wav"),
                ("M4A files", "*.m4a"),
                ("AAC files", "*.aac"),
                ("OGG files", "*.ogg"),
                ("FLAC files", "*.flac"),
                ("WMA files", "*.wma"),
                ("All files", "*.*")
            ]
        )
        for file_path in file_paths:
            try:
                # Get audio duration for token calculation
                try:
                    audio_info = sf.info(file_path)
                    duration = audio_info.duration
                    audio_tokens = int(duration * self.token_manager.audio_tokens_per_second)
                except Exception as e:
                    # Fallback for unsupported formats - estimate based on file size
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    estimated_duration = file_size_mb * 60  # Rough estimate: 1MB ≈ 1 minute
                    duration = min(estimated_duration, 300)  # Cap at 5 minutes for safety
                    audio_tokens = int(duration * self.token_manager.audio_tokens_per_second)
                    print(f"Warning: Could not read audio info for {file_path}, using estimate")
                
                self.attachments.append({"type": "audio", "audio": file_path})
                self.add_system_message(f"Audio file added: {os.path.basename(file_path)} ({duration:.1f}s, {audio_tokens} tokens)", attachment=True)
                
            except Exception as e:
                messagebox.showerror("Audio Error", f"Could not process audio file {os.path.basename(file_path)}: {str(e)}")
                
        self.update_attachment_label()
        self.update_token_display()
        
    def select_document(self):
        file_path = filedialog.askopenfilename(
            title="Select Document",
            filetypes=[
                ("All supported", "*.pdf *.docx *.txt *.md *.rtf *.html *.py *.js *.css *.xml *.json"),
                ("PDF files", "*.pdf"),
                ("Word documents", "*.docx"),
                ("Text files", "*.txt *.md"),
                ("Rich text", "*.rtf"),
                ("Web files", "*.html *.css *.js"),
                ("Code files", "*.py *.js *.css *.xml *.json"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            try:
                DocumentViewer(self.root, file_path, self.on_document_processed)
            except Exception as e:
                messagebox.showerror("Document Error", f"Could not open document: {str(e)}")
                
    def on_document_processed(self, text_content, pages):
        self.chat_context = text_content
        self.context_pages = pages
        
        self.context_display.delete(1.0, tk.END)
        self.context_display.insert(tk.END, text_content)
        
        tokens = self.token_manager.count_text_tokens(text_content)
        color = self.token_manager.get_color_for_usage(tokens)
        
        if len(pages) > 1:
            pages_str = f"pages {min(pages)}-{max(pages)}" if pages else "document"
        else:
            pages_str = "document"
        self.context_info_label.config(text=f"Context: {tokens} tokens from {pages_str}", foreground=color)
        
        self.add_system_message(f"Document context loaded: {tokens} tokens", attachment=True)
        self.update_token_display()
        
    def clear_context(self):
        self.chat_context = ""
        self.context_pages = []
        
        self.context_display.delete(1.0, tk.END)
        
        self.context_info_label.config(text="Context: 0 tokens", foreground='gray')
        self.add_system_message("Chat context cleared", attachment=True)
        self.update_token_display()
        
    def save_context(self):
        context_text = self.context_display.get("1.0", tk.END).strip()
        if not context_text:
            messagebox.showwarning("No Context", "No context to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Context",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(context_text)
                self.add_system_message(f"Context saved to: {os.path.basename(file_path)}", attachment=True)
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save context: {str(e)}")
                
    def load_context(self):
        file_path = filedialog.askopenfilename(
            title="Load Context",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.context_display.delete(1.0, tk.END)
                self.context_display.insert(tk.END, content)
                
                tokens = self.token_manager.count_text_tokens(content)
                color = self.token_manager.get_color_for_usage(tokens)
                self.context_info_label.config(text=f"Context: {tokens} tokens from {os.path.basename(file_path)}", foreground=color)
                self.add_system_message(f"Context loaded from: {os.path.basename(file_path)} ({tokens} tokens)", attachment=True)
                self.update_token_display()
            except Exception as e:
                messagebox.showerror("Load Error", f"Could not load context: {str(e)}")
            
    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.audio_btn.config(text="Stop Recording")
            self.audio_data = []
            self.audio_stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=44100
            )
            self.audio_stream.start()
            self.status_label.config(text="Recording audio...", foreground='red')
        else:
            self.recording = False
            self.audio_btn.config(text="Record Audio")
            self.audio_stream.stop()
            self.audio_stream.close()
            self.save_audio()
            self.status_label.config(text="Ready", foreground='green')
            
    def audio_callback(self, indata, frames, time, status):
        self.audio_data.append(indata.copy())
        
    def save_audio(self):
        if self.audio_data:
            audio_array = np.concatenate(self.audio_data, axis=0)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                sf.write(tmp.name, audio_array, 44100)
                duration = len(audio_array) / 44100
                audio_tokens = int(duration * self.token_manager.audio_tokens_per_second)
                self.attachments.append({"type": "audio", "audio": tmp.name})
                self.add_system_message(f"Audio recorded: {duration:.1f}s ({audio_tokens} tokens)", attachment=True)
                self.update_attachment_label()
                self.update_token_display()
                
    def capture_screen(self):
        self.root.withdraw()
        self.root.after(200, self.show_capture_widget)
        
    def show_capture_widget(self):
        ScreenCapture(self.on_screen_captured)
        
    def on_screen_captured(self, image):
        self.root.deiconify()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            self.attachments.append({"type": "image", "image": tmp.name})
            self.add_system_message(f"Screen capture added: {image.width}x{image.height} (256 tokens)", attachment=True)
            self.update_attachment_label()
            self.update_token_display()
            
    def update_attachment_label(self):
        count = len(self.attachments)
        if count > 0:
            types = {}
            total_tokens = 0
            for a in self.attachments:
                atype = a['type']
                types[atype] = types.get(atype, 0) + 1
                if atype == 'image':
                    total_tokens += 256
                elif atype == 'audio':
                    try:
                        audio_info = sf.info(a['audio'])
                        total_tokens += int(audio_info.duration * self.token_manager.audio_tokens_per_second)
                    except:
                        total_tokens += 100
            
            parts = []
            for t, c in types.items():
                if c > 1:
                    parts.append(f"{c} {t}s")
                else:
                    parts.append(f"1 {t}")
            
            self.attachment_label.config(text=f"Attachments: {', '.join(parts)} ({total_tokens} tokens)")
        else:
            self.attachment_label.config(text="")
            
    def clear_chat(self):
        if self.is_generating:
            self.stop_generation()
            
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_history.clear()
        self.attachments = []
        self.current_generation_thread = None
        self.update_attachment_label()
        self.update_token_display()
        self.update_history_context()
        self.update_ui_for_generation(False)
        self.refresh_mini_progress()
        self.add_system_message("Chat cleared")
    
    def refresh_mini_progress(self):
        self.mini_stats_display.config(state=tk.NORMAL)
        self.mini_stats_display.delete(1.0, tk.END)
        
        context_text = self.context_display.get("1.0", tk.END).strip()
        chat_count = len(self.chat_history)
        
        if self.use_general_mode.get():
            subject = self.subject_var.get()
            grade = self.grade_var.get()
            difficulty = self.difficulty_var.get()
            
            stats_text = f"▲ CURRENT STUDY MODE\n\n"
            stats_text += f"Mode: General Subject Settings\n"
            stats_text += f"Subject: {subject}\n"
            stats_text += f"Grade Level: {grade}\n"
            stats_text += f"Difficulty: {difficulty}\n\n"
        else:
            stats_text = f"▲ CURRENT STUDY MODE\n\n"
            stats_text += f"Mode: Study Materials Only\n"
            
        if context_text:
            context_tokens = self.token_manager.count_text_tokens(context_text)
            stats_text += f"Study Materials: {context_tokens:,} tokens loaded\n"
        else:
            stats_text += f"Study Materials: None uploaded\n"
            
        stats_text += f"Chat Messages: {chat_count}\n"
        
        if chat_count > 0:
            stats_text += f"\n📚 ACTIVITY SUMMARY\n"
            user_messages = sum(1 for msg in self.chat_history if msg.get('role') == 'user')
            assistant_messages = sum(1 for msg in self.chat_history if msg.get('role') == 'assistant')
            total_tokens = sum(msg.get('tokens', 0) for msg in self.chat_history)
            
            stats_text += f"Questions Asked: {user_messages}\n"
            stats_text += f"Responses Received: {assistant_messages}\n"
            stats_text += f"Total Conversation: {total_tokens:,} tokens\n"
        else:
            stats_text += f"\nStart chatting to see activity statistics!"
        
        self.mini_stats_display.insert(1.0, stats_text)
        self.mini_stats_display.config(state=tk.DISABLED)
    
    def explain_concept(self):
        context_text = self.context_display.get("1.0", tk.END).strip()
        
        if self.use_general_mode.get():
            subject = self.subject_var.get()
            grade = self.grade_var.get()
            difficulty = self.difficulty_var.get()
            
            prompt = f"Explain key concepts in {subject} for {grade} level at {difficulty} difficulty. Please:\n"
            prompt += "1. Identify important concepts in this subject area\n"
            prompt += "2. Provide clear, simple definitions\n"
            prompt += "3. Give real-world examples\n"
            prompt += "4. Break down complex parts step-by-step\n"
            prompt += "5. Include common misconceptions to avoid\n"
            prompt += "6. Suggest ways to remember these concepts\n"
            prompt += "7. Provide practice questions\n\n"
            
            if context_text:
                prompt += "Also consider any uploaded study materials in your explanation."
        else:
            if not context_text:
                messagebox.showwarning("No Study Materials", "Please upload study materials or enable general subject settings.")
                return
                
            prompt = f"Based on the uploaded study materials and our conversation, explain the key concepts covered. Please:\n"
            prompt += "1. Identify the main concepts from the materials\n"
            prompt += "2. Provide clear, simple definitions\n"
            prompt += "3. Give real-world examples\n"
            prompt += "4. Break down complex parts step-by-step\n"
            prompt += "5. Include common misconceptions to avoid\n"
            prompt += "6. Suggest ways to remember these concepts\n"
            prompt += "7. Provide practice questions based on the material"
        
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", prompt)
        self.send_message()
    
    def create_concept_map(self):
        context_text = self.context_display.get("1.0", tk.END).strip()
        
        if self.use_general_mode.get():
            subject = self.subject_var.get()
            
            prompt = f"Create a comprehensive concept map for key topics in {subject}. Structure it as:\n\n"
            prompt += "MAIN CONCEPT: [Core subject concept]\n"
            prompt += "├── Core Subtopic 1\n│   ├── Key Detail A\n│   ├── Key Detail B\n│   └── connects to → [Related Concept]\n"
            prompt += "├── Core Subtopic 2\n│   ├── Important Point X\n│   └── Important Point Y\n"
            prompt += "└── Core Subtopic 3\n    └── leads to → [Next Level Topic]\n\n"
            prompt += "Include:\n1. Main concept at center\n"
            prompt += "2. 3-5 major subtopics\n3. Key details under each subtopic\n"
            prompt += "4. Clear relationships between concepts\n5. Real-world examples where relevant\n"
            prompt += "6. Prerequisites and follow-up topics\n\n"
            
            if context_text:
                prompt += "Also incorporate concepts from the uploaded study materials."
        else:
            if not context_text:
                messagebox.showwarning("No Study Materials", "Please upload study materials or enable general subject settings.")
                return
                
            prompt = f"Create a comprehensive concept map based on the uploaded study materials and our conversation. Structure it as:\n\n"
            prompt += "MAIN CONCEPT: [Identify from materials]\n"
            prompt += "├── Core Subtopic 1\n│   ├── Key Detail A\n│   ├── Key Detail B\n│   └── connects to → [Related Concept]\n"
            prompt += "├── Core Subtopic 2\n│   ├── Important Point X\n│   └── Important Point Y\n"
            prompt += "└── Core Subtopic 3\n    └── leads to → [Next Level Topic]\n\n"
            prompt += "Base the concept map entirely on the uploaded study materials and our conversation."
        
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", prompt)
        self.send_message()
    
    def homework_hints(self):
        context_text = self.context_display.get("1.0", tk.END).strip()
        
        if self.use_general_mode.get():
            subject = self.subject_var.get()
            
            prompt = f"Provide homework hints for common {subject} problems by giving HINTS, not direct answers:\n\n"
            prompt += f"For typical {subject} homework problems and concepts:\n\n"
            
            if context_text:
                prompt += "Consider both general subject concepts and the uploaded study materials.\n\n"
            
            prompt += "For each topic/problem type, provide:\n"
            prompt += "1. Guiding questions to help think through problems\n"
            prompt += "2. Relevant concepts to review\n3. Similar example problems\n"
            prompt += "4. Step-by-step approach without giving away answers\n"
            prompt += "5. Common mistakes to avoid\n\n"
            prompt += "Focus on helping learn and understand the process, not just getting answers."
        else:
            if not context_text:
                messagebox.showwarning("No Study Materials", "Please upload study materials or enable general subject settings.")
                return
                
            prompt = f"Based on the uploaded study materials and our conversation, provide homework hints by giving HINTS, not direct answers:\n\n"
            prompt += "Look at the study materials and our discussion to identify potential homework problems or concepts that need practice.\n\n"
            prompt += "For each identified topic/problem type, provide:\n"
            prompt += "1. Guiding questions to help think through problems\n"
            prompt += "2. Relevant concepts to review\n3. Similar example problems\n"
            prompt += "4. Step-by-step approach without giving away answers\n"
            prompt += "5. Common mistakes to avoid\n\n"
            prompt += "Focus on helping learn and understand the process from the materials."
        
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", prompt)
        self.send_message()
    
    def step_by_step_help(self):
        context_text = self.context_display.get("1.0", tk.END).strip()
        
        if self.use_general_mode.get():
            subject = self.subject_var.get()
            
            prompt = f"Provide step-by-step guidance for common {subject} problems:\n\n"
            
            if context_text:
                prompt += "Consider both general subject problem types and any specific problems from the uploaded study materials.\n\n"
            else:
                prompt += f"Focus on typical {subject} problem types that students commonly encounter.\n\n"
        else:
            if not context_text:
                messagebox.showwarning("No Study Materials", "Please upload study materials or enable general subject settings.")
                return
                
            prompt = f"Provide step-by-step guidance based on the uploaded study materials and our conversation:\n\n"
            prompt += "Based on the study materials, identify key problem types and provide detailed step-by-step solutions.\n\n"
        
        prompt += "For each problem type, break it down into clear steps:\n"
        prompt += "Step 1: [What to do and why]\nStep 2: [Next action and reasoning]\nStep 3: [Continue...]\n\n"
        prompt += "For each step, explain:\n- What to do\n- Why we do it\n"
        prompt += "- What to look out for\n- How it connects to the next step\n\n"
        prompt += "Include the reasoning behind each step to promote understanding and provide multiple examples."
        
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", prompt)
        self.send_message()
    
    def find_similar_problems(self):
        context_text = self.context_display.get("1.0", tk.END).strip()
        
        if self.use_general_mode.get():
            subject = self.subject_var.get()
            difficulty = self.difficulty_var.get()
            
            prompt = f"Create 3-5 similar practice problems for {subject} at {difficulty} level:\n\n"
            
            if context_text:
                prompt += "Use both typical subject problem types and any specific problems from the uploaded study materials.\n\n"
            else:
                prompt += f"Focus on common {subject} problem types at {difficulty} level.\n\n"
        else:
            if not context_text:
                messagebox.showwarning("No Study Materials", "Please upload study materials or enable general subject settings.")
                return
                
            prompt = f"Based on the uploaded study materials and our conversation, create 3-5 similar practice problems:\n\n"
            prompt += "Use the study materials to identify key problem types, then create similar practice problems.\n\n"
        
        prompt += "For each similar problem:\n1. State the problem clearly\n"
        prompt += "2. Vary the numbers/context while keeping the same concept\n"
        prompt += "3. Include the solution approach\n4. Indicate the difficulty level\n"
        prompt += "5. Note what concept it reinforces\n\n"
        prompt += "Arrange them from easier to harder to build confidence and cover different variations of the same concept types."
        
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", prompt)
        self.send_message()
    
    def generate_quiz(self):
        num_questions = int(self.quiz_count_var.get())
        quiz_type = self.quiz_type_var.get()
        context_text = self.context_display.get("1.0", tk.END).strip()
        
        if self.use_general_mode.get():
            subject = self.subject_var.get()
            difficulty = self.difficulty_var.get()
            
            prompt = f"Create a {num_questions}-question {quiz_type} quiz for {subject} at {difficulty} level"
            
            if context_text:
                prompt += " incorporating both general subject concepts and the uploaded study materials.\n\n"
            else:
                prompt += f" covering key {subject} concepts at {difficulty} level.\n\n"
        else:
            if not context_text:
                messagebox.showwarning("No Study Materials", "Please upload study materials or enable general subject settings.")
                return
                
            prompt = f"Create a {num_questions}-question {quiz_type} quiz based entirely on the uploaded study materials and our conversation.\n\n"
        
        prompt += "For each question:\n1. Create a clear, well-written question\n"
        prompt += "2. Provide multiple choice options (A, B, C, D) if applicable\n"
        prompt += "3. Indicate the correct answer\n4. Explain why the correct answer is right\n"
        prompt += "5. Explain why wrong answers are incorrect\n6. Include difficulty level for each question\n\n"
        prompt += "Make questions progressively challenging and cover different aspects of the material."
        
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", prompt)
        self.send_message()
    
    def adaptive_practice(self):
        context_text = self.context_display.get("1.0", tk.END).strip()
        
        if self.use_general_mode.get():
            subject = self.subject_var.get()
            difficulty = self.difficulty_var.get()
            
            prompt = f"Create an adaptive practice session for {subject} at {difficulty} level. "
            
            if context_text:
                prompt += "Use both general subject concepts and the uploaded study materials to design the practice session.\n\n"
            else:
                prompt += f"Focus on core {subject} concepts and skills.\n\n"
        else:
            if not context_text:
                messagebox.showwarning("No Study Materials", "Please upload study materials or enable general subject settings.")
                return
                
            prompt = "Create an adaptive practice session based entirely on the uploaded study materials and our conversation. "
        
        prompt += "Please:\n1. Start with easier questions to build confidence\n"
        prompt += "2. Gradually increase difficulty based on the material\n"
        prompt += "3. Focus on key concepts that need practice\n"
        prompt += "4. Provide immediate feedback after each question\n"
        prompt += "5. Create questions that build on each other\n\n"
        prompt += "Begin with 3 warm-up questions, then progressively challenge with more complex problems."
        
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", prompt)
        self.send_message()
    
    def create_practice_test(self):
        context_text = self.context_display.get("1.0", tk.END).strip()
        
        if self.use_general_mode.get():
            subject = self.subject_var.get()
            difficulty = self.difficulty_var.get()
            
            prompt = f"Create a comprehensive practice test for {subject} at {difficulty} level"
            
            if context_text:
                prompt += " incorporating both general subject concepts and the uploaded study materials.\n\n"
            else:
                prompt += f" covering key {subject} concepts.\n\n"
        else:
            if not context_text:
                messagebox.showwarning("No Study Materials", "Please upload study materials or enable general subject settings.")
                return
                
            prompt = f"Create a comprehensive practice test based entirely on the uploaded study materials and our conversation.\n\n"
        
        prompt += "Test Structure:\n1. Mix of question types (multiple choice, short answer, essay)\n"
        prompt += "2. Cover all major topics from the material\n"
        prompt += "3. Include easy, medium, and hard questions\n4. Provide detailed answer key\n"
        prompt += "5. Include point values and time suggestions\n6. Add study tips for difficult concepts\n\n"
        prompt += "Format as a real exam with clear instructions. Make it 10-15 questions covering the breadth of topics."
        
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", prompt)
        self.send_message()
    
    def timed_assessment(self):
        context_text = self.context_display.get("1.0", tk.END).strip()
        
        if self.use_general_mode.get():
            subject = self.subject_var.get()
            difficulty = self.difficulty_var.get()
            
            prompt = f"Create a 20-minute timed assessment for {subject} at {difficulty} level"
            
            if context_text:
                prompt += " incorporating both general subject concepts and the uploaded study materials.\n\n"
            else:
                prompt += f" covering core {subject} concepts.\n\n"
        else:
            if not context_text:
                messagebox.showwarning("No Study Materials", "Please upload study materials or enable general subject settings.")
                return
                
            prompt = f"Create a 20-minute timed assessment based entirely on the uploaded study materials and our conversation.\n\n"
        
        prompt += "ASSESSMENT DESIGN:\n"
        prompt += "1. QUESTIONS (for 20 minutes):\n"
        prompt += "   - 60% quick recall/recognition (1-2 min each)\n"
        prompt += "   - 30% problem-solving (3-5 min each)\n"
        prompt += "   - 10% analysis/synthesis (5-7 min each)\n\n"
        prompt += "2. TIME MANAGEMENT GUIDE:\n"
        prompt += "   - Suggest time allocation per section\n"
        prompt += "   - Include pacing strategies\n"
        prompt += "   - Mark point values clearly\n\n"
        prompt += "3. INSTRUCTIONS:\n"
        prompt += "   - Clear format and expectations\n"
        prompt += "   - Tips for managing time pressure\n"
        prompt += "   - Scoring rubric\n\n"
        prompt += "Base questions on the relevant topics and concepts."
        
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", prompt)
        self.send_message()
    
    def skill_assessment(self):
        context_text = self.context_display.get("1.0", tk.END).strip()
        
        if self.use_general_mode.get():
            subject = self.subject_var.get()
            grade = self.grade_var.get()
            
            prompt = f"Create a comprehensive skill assessment for {subject} at {grade} level"
            
            if context_text:
                prompt += " incorporating both general subject skills and the uploaded study materials.\n\n"
            else:
                prompt += f" covering core {subject} skills.\n\n"
        else:
            if not context_text:
                messagebox.showwarning("No Study Materials", "Please upload study materials or enable general subject settings.")
                return
                
            prompt = f"Create a comprehensive skill assessment based entirely on the uploaded study materials and our conversation.\n\n"
        
        prompt += "ASSESSMENT FRAMEWORK:\n"
        prompt += "1. FOUNDATIONAL KNOWLEDGE (25%)\n   - Basic concepts and terminology\n   - Essential facts and principles\n\n"
        prompt += "2. APPLICATION SKILLS (25%)\n   - Using knowledge to solve problems\n   - Applying concepts to new situations\n\n"
        prompt += "3. ANALYTICAL THINKING (25%)\n   - Breaking down complex problems\n   - Identifying patterns and relationships\n\n"
        prompt += "4. SYNTHESIS & EVALUATION (25%)\n   - Combining ideas creatively\n   - Making judgments and assessments\n\n"
        prompt += "For each area, provide:\n- 2-3 diagnostic questions\n"
        prompt += "- Performance indicators (Beginner/Intermediate/Advanced)\n"
        prompt += "- Specific feedback based on responses\n- Targeted improvement recommendations\n"
        prompt += "- Next steps for skill development"
        
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", prompt)
        self.send_message()
    
    def improvement_suggestions(self):
        context_text = self.context_display.get("1.0", tk.END).strip()
        
        if self.use_general_mode.get():
            subject = self.subject_var.get()
            
            prompt = f"Provide personalized improvement suggestions for learning {subject}.\n\n"
            
            if context_text:
                prompt += "Consider both general study strategies and the uploaded study materials to suggest:\n"
            else:
                prompt += f"Based on effective {subject} learning strategies, suggest:\n"
        else:
            if not context_text:
                messagebox.showwarning("No Study Materials", "Please upload study materials or enable general subject settings.")
                return
                
            prompt = f"Based on the uploaded study materials and our conversation, provide personalized improvement suggestions.\n\n"
            prompt += "Consider the study materials and our conversation to suggest:\n"
        
        prompt += "1. Areas that might need more focus\n2. Study techniques that could help\n"
        prompt += "3. Ways to deepen understanding\n4. Next topics to explore\n"
        prompt += "5. Methods to retain information better\n6. Specific practice exercises\n"
        prompt += "7. Resources for further learning"
        
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", prompt)
        self.send_message()
    
    def identify_strengths(self):
        context_text = self.context_display.get("1.0", tk.END).strip()
        
        if self.use_general_mode.get():
            subject = self.subject_var.get()
            
            prompt = f"Help me identify my learning strengths in {subject} based on our conversation"
            
            if context_text:
                prompt += " and the uploaded study materials.\n\n"
            else:
                prompt += ".\n\n"
        else:
            if not context_text:
                messagebox.showwarning("No Study Materials", "Please upload study materials or enable general subject settings.")
                return
                
            prompt = f"Based on the uploaded study materials and our conversation, help me identify my learning strengths.\n\n"
        
        chat_count = len(self.chat_history)
        if chat_count > 0:
            user_messages = sum(1 for msg in self.chat_history if msg.get('role') == 'user')
            prompt += f"Conversation Analysis:\n- Total messages exchanged: {chat_count}\n"
            prompt += f"- Questions asked: {user_messages}\n\n"
        
        prompt += "Based on our conversation"
        if context_text:
            prompt += ", the uploaded study materials,"
        prompt += " and any patterns you can observe, please analyze:\n"
        
        prompt += "1. Learning style preferences I'm showing\n2. Areas where I seem most engaged\n"
        prompt += "3. Types of questions I ask most\n4. Problem-solving approaches I prefer\n"
        prompt += "5. Consistent patterns in my learning\n6. Strengths to build upon\n"
        prompt += "7. Natural talents or inclinations you've noticed"
        
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", prompt)
        self.send_message()

def main():
    root = tk.Tk()
    
    root.resizable(True, True)
    root.wm_resizable(True, True)
    
    try:
        root.wm_attributes('-type', 'normal')
    except:
        pass
    
    app = ChatbotApp(root)
    
    root.resizable(True, True)
    root.wm_resizable(True, True)
    
    print("Main window should now be resizable")
    print("Try dragging the window edges/corners")
    
    try:
        root.iconbitmap(default='icon.ico')
    except:
        pass
    
    root.mainloop()

if __name__ == "__main__":
    print("Educational AI Tutor - Multimodal Learning Assistant (Offline Ready)")
    print("===================================================================")
    print("🌐 OFFLINE FEATURES: Complete privacy • No internet required after download • Local AI processing")
    print("🎓 MULTIMODAL: Text, Image, Audio (Record + Import), Screen Capture, Smart Clipboard, Documents")
    print("🤖 AI MODELS: Gemma 3n (2B/4B/5B/8B) - Download once, use forever offline")
    print("📊 MANAGEMENT: 32K context limit, real-time tracking, history management, model management")
    print("📄 FORMATS: PDF, DOCX, TXT, MD, RTF, HTML, CSS, JS, PY, XML, JSON")
    print("🎵 AUDIO: MP3, WAV, M4A, AAC, OGG, FLAC, WMA")
    print("📖 Press F1 or click 'Help & Info' for comprehensive usage guide")
    print("🤖 Click 'Model Manager' to download AI models for offline use")
    print("")
    main()
