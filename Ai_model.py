import os
import io
from typing import List, Dict, Tuple
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import tempfile
import cv2
import numpy as np
import easyocr
import streamlit as st
from PIL import Image, ImageOps,ImageDraw
import hashlib
from paddleocr import PaddleOCR
from spellchecker import SpellChecker
import ollama
import re
import logging
import requests
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import shutil
import datetime
import json
import atexit
import uuid
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize models
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
spell = SpellChecker()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
DATASET_PATH2 = "./dataset2"

# Global variables to store chat history
chat_history = []
history_faiss_index = faiss.IndexFlatL2(384)  # Match embedding dimension
history_texts = []

def save_image_to_temp(image):
    """Save the PIL image to a temporary file and return its path."""
    temp_dir = tempfile.gettempdir()
    image_name = f"chat_image_{uuid.uuid4()}.png"
    image_path = os.path.join(temp_dir, image_name)
    image.save(image_path, format="PNG")
    return image_path

# Load chat history function
def load_chat_history():
    """Loads chat history and initializes required keys."""
    try:
        if os.path.exists("chats.json"):
            with open("chats.json", "r") as f:
                chats = json.load(f)

            for idx, chat in enumerate(chats):
                # Convert embeddings back to numpy arrays
                if "embeddings" in chat:
                    chat["embeddings"] = [np.array(embedding) for embedding in chat["embeddings"]]

                # Load FAISS index if it exists
                faiss_path = f"faiss_index_{idx}.index"
                chat["faiss_index"] = (
                    faiss.read_index(faiss_path) if os.path.exists(faiss_path) else faiss.IndexFlatL2(384)
                )
                chat.setdefault("history", [])
                chat.setdefault("texts", [])

            return chats
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.warning(f"Failed to load chat history: {e}")

    return [
        {"name": "Chat 1", "history": [], "texts": [], "embeddings": [], "faiss_index": faiss.IndexFlatL2(384)}
    ]


# Save all chats function
def save_all_chats():
    """Saves chat history and FAISS indices to disk."""
    chats_to_save = []
    for idx, chat in enumerate(st.session_state.chat_tabs):
        # Create a copy to avoid modifying the original chat object
        chat_copy = chat.copy()
        
        # Save FAISS index to a file and remove it from the copy
        if "faiss_index" in chat_copy:
            faiss.write_index(chat_copy["faiss_index"], f"faiss_index_{idx}.index")
            del chat_copy["faiss_index"]

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            """Recursively convert numpy arrays to lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj

        chat_copy = convert_numpy(chat_copy)

        # Handle uploaded file serialization
        if "uploaded_file" in chat_copy and chat_copy["uploaded_file"]:
            uploaded_file = chat_copy["uploaded_file"]
            chat_copy["uploaded_file"] = {"name": uploaded_file.name}

        chats_to_save.append(chat_copy)

    # Save the sanitized chat data (without FAISS index) to a JSON file
    with open("chats.json", "w") as f:
        json.dump(chats_to_save, f)


# Add message to chat history
def add_to_chat_history(user_input, ai_response):
    """Adds user and AI messages to the chat and updates embeddings."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Ensure ai_response is a dictionary
    if isinstance(ai_response, str):
        ai_response = {"ai_response_text": ai_response}

    # Extract fields
    images = ai_response.get("images", [])
    ai_response_text = ai_response.get("ai_response_text", "Error")
    pdf_content = ai_response.get("pdf_content", None)

    # Convert PIL.Images in 'images' to file paths
    new_images = []
    for im in images:
        if isinstance(im, Image.Image):
            image_path = save_image_to_temp(im)
            new_images.append(image_path)
        else:
            # If it's already a path or another serializable form, just append it
            new_images.append(im)
    ai_response["images"] = new_images

    # Structure the chat entry
    chat_entry = {
        "timestamp": timestamp,
        "user_input": user_input,
        "ai_response": {
            "ai_response_text": ai_response_text,
            "images": ai_response["images"],  # updated images list
            "pdf_content": pdf_content,
        },
    }

    current_chat = st.session_state.current_chat

    # Ensure all keys are present
    current_chat.setdefault("history", [])
    current_chat.setdefault("texts", [])
    current_chat.setdefault("embeddings", [])
    if "faiss_index" not in current_chat:
        current_chat["faiss_index"] = faiss.IndexFlatL2(384)

    # Prevent duplicates by checking the last history entry
    if current_chat["history"] and current_chat["history"][-1].get("user_input") == user_input:
        print("Duplicate user input detected; skipping addition to history.")
        return

    # Save the chat entry to history
    current_chat["history"].append(chat_entry)

    # Generate embeddings for textual data
    conversation_text = f"User: {user_input}\nAI: {ai_response_text}"
    embedding = embedding_model.encode(conversation_text).flatten()
    embedding /= np.linalg.norm(embedding)

    current_chat["embeddings"].append(embedding)
    current_chat["texts"].append(conversation_text)
    current_chat["faiss_index"].add(np.array([embedding]))

    # Log saved data (optional for debugging)
    print(f"Chat entry saved at {timestamp}. AI response text: {ai_response_text}")
    if new_images:
        print(f"Saved {len(new_images)} image(s) in chat history.")
    if pdf_content:
        print("PDF content saved in chat history.")

# Get relevant context
def get_relevant_context(query, top_k=3):
    """Fetches relevant chat context using FAISS."""
    current_chat = st.session_state.current_chat

    # Ensure texts and faiss_index are properly initialized
    if not current_chat.get("texts") or current_chat["faiss_index"].ntotal == 0:
        return []

    # Encode and normalize the query embedding
    query_embedding = embedding_model.encode(query).flatten()
    query_embedding /= np.linalg.norm(query_embedding)

    # Perform the search
    distances, indices = current_chat["faiss_index"].search(np.array([query_embedding]), top_k)

    # Collect the relevant chat context
    relevant_context = []
    for i in indices[0]:
        if i < len(current_chat["texts"]):
            # For textual content, directly fetch the text
            relevant_context.append(current_chat["texts"][i])

        # Handle non-text content such as images or PDF content if present
        if i < len(current_chat["history"]):
            chat_entry = current_chat["history"][i]
            if "images" in chat_entry["ai_response"]:  # If it's an image
                # Convert numpy arrays to strings for joining
                image_paths = [
                    img if isinstance(img, str) else f"<Image data at {hex(id(img))}>"
                    for img in chat_entry["ai_response"]["images"]
                ]
                relevant_context.append(f"Image(s) from chat {i + 1}: {', '.join(image_paths)}")
            if "pdf_content" in chat_entry["ai_response"]:  # If it contains PDF content
                relevant_context.append(f"PDF Content from chat {i + 1}: {chat_entry['ai_response']['pdf_content']}")
            if "ai_response_text" in chat_entry["ai_response"]:  # If it contains an AI response
                relevant_context.append(f"AI Response from chat {i + 1}: {chat_entry['ai_response']['ai_response_text']}")

    return relevant_context


# Delete saved chat history on application closure
def cleanup_chat_history():
    """Delete all saved chat history and FAISS indices."""
    try:
        if os.path.exists("chats.json"):
            os.remove("chats.json")
        # Remove all FAISS index files
        for file in os.listdir("."):
            if file.startswith("faiss_index_") and file.endswith(".index"):
                os.remove(file)
        logging.info("Chat history and FAISS indices cleaned up on application closure.")
    except Exception as e:
        logging.error(f"Failed to clean up chat history: {e}")

# Register cleanup function
atexit.register(cleanup_chat_history)


# Function to find the folder containing the uploaded image
def find_image_folder(uploaded_image_path, dataset_path):
    # Load the uploaded image
    uploaded_image = cv2.imread(uploaded_image_path, cv2.IMREAD_GRAYSCALE)
    if uploaded_image is None:
        logging.error("Uploaded image could not be loaded.")
        return None

    orb = cv2.ORB_create()  # Initialize ORB detector
    kp1, des1 = orb.detectAndCompute(uploaded_image, None)
    if des1 is None:
        logging.error("No descriptors found in the uploaded image.")
        return None

    best_match_folder = None
    max_good_matches = 0

    for project_folder in os.listdir(dataset_path):
        project_folder_path = os.path.join(dataset_path, project_folder)
        if os.path.isdir(project_folder_path):
            for root, _, files in os.walk(project_folder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        dataset_image_path = os.path.join(root, file)
                        dataset_image = cv2.imread(dataset_image_path, cv2.IMREAD_GRAYSCALE)
                        if dataset_image is None:
                            continue

                        kp2, des2 = orb.detectAndCompute(dataset_image, None)
                        if des2 is None:
                            continue

                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(des1, des2)
                        matches = sorted(matches, key=lambda x: x.distance)
                        good_matches = [m for m in matches if m.distance < 30]  # Stricter threshold

                        if len(good_matches) > max_good_matches:
                            max_good_matches = len(good_matches)
                            best_match_folder = project_folder_path

    if best_match_folder:
        logging.info(f"Best match folder: {best_match_folder} with {max_good_matches} good matches.")
    else:
        logging.warning("No suitable match found.")
    return best_match_folder



# Function to process the PDF in the project folder
def process_project_pdf(project_folder):
    pdf_files = []  # List to store found PDFs
    logging.info(f"Checking for PDFs in: {project_folder}")

    # Check in the project folder
    for file in os.listdir(project_folder):
        if file.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(project_folder, file))

    # Optionally, check in a 'pdf' subfolder
    pdf_folder = os.path.join(project_folder, "pdf")
    if os.path.exists(pdf_folder):
        logging.info(f"Checking for PDFs in subfolder: {pdf_folder}")
        for file in os.listdir(pdf_folder):
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(pdf_folder, file))

    if not pdf_files:
        logging.error(f"No PDFs found in project folder: {project_folder}")
        return "No PDFs found in the project."

    # Process found PDFs
    pdf_texts = []
    for pdf_path in pdf_files:
        logging.info(f"Processing PDF: {pdf_path}")
        try:
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    pdf_texts.append(text.strip())
                else:
                    logging.warning(f"No text extracted from page in: {pdf_path}")
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")

    return "\n".join(pdf_texts) if pdf_texts else "No readable text found in the PDFs."



# Paths
FAISS_INDEX_PATH = "faiss_index.pkl"
DATASET_PATH = "./dataset"

# Cache Resource for Processing PDFs
@st.cache_resource
def process_and_cache_pdfs(dataset_path=DATASET_PATH):
    """Processes PDFs and returns texts with embeddings."""
    texts = []
    embeddings = []
    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(dataset_path, file_name)
            logging.info(f"Processing {file_path}")
            try:
                pdf_reader = PdfReader(file_path)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        texts.append(text.strip())
                        embedding = embedding_model.encode(text.strip()).astype("float32")
                        embeddings.append(embedding)
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
    if embeddings:
        embeddings = np.vstack(embeddings)
    return texts, embeddings

# Load or Create FAISS Index
@st.cache_resource
def load_or_create_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        with open(FAISS_INDEX_PATH, "rb") as f:
            index_data = pickle.load(f)
            logging.info("FAISS index loaded from cache.")
            return index_data["index"], index_data["texts"]
    else:
        logging.info("No FAISS index found. Creating new index...")
        texts, embeddings = process_and_cache_pdfs()
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        with open(FAISS_INDEX_PATH, "wb") as f:
            pickle.dump({"index": index, "texts": texts}, f)
        logging.info("FAISS index created and saved to cache.")
        return index, texts

# Load FAISS index and texts once at the start
if "faiss_index" not in st.session_state or "pdf_texts" not in st.session_state:
    st.session_state["faiss_index"], st.session_state["pdf_texts"] = load_or_create_faiss_index()

# Query FAISS Index
def query_faiss_index(question, faiss_index, pdf_texts, top_k=5):
    question_embedding = embedding_model.encode(question).astype("float32").reshape(1, -1)
    distances, indices = faiss_index.search(question_embedding, top_k)
    results = [pdf_texts[i] for i in indices[0] if 0 <= i < len(pdf_texts)]
    return results

def classify_question(question):
    keywords_blueprint = ['blueprint', 'floorplan', 'room', 'layout', 'size']
    keywords_architecture = ['design principles', 'architecture', 'building', 'construction']
    
    if any(keyword in question.lower() for keyword in keywords_blueprint):
        return 'blueprint'
    elif any(keyword in question.lower() for keyword in keywords_architecture):
        return 'architecture'
    else:
        return 'general'

# Function to generate prompts for Llama
def generate_prompt_with_memory(question, matched_rooms):
    # Access from st.session_state
    faiss_index = st.session_state["faiss_index"]
    pdf_texts = st.session_state["pdf_texts"]
    
    # Get relevant context from chat history
    chat_context = "\n\n".join(get_relevant_context(question))
    context_type = classify_question(question)

    if context_type in ['blueprint', 'floorplan']:
        context = """
        You are a friendly assistant helping a user analyze blueprint details. Please provide feedback about the room layout 
        and any information from the architectural PDFs that the user might find useful based on the blueprint. Respond conversationally and warmly.
        End responses with inviting language like "Let me know if you'd like more help!"
        """
        room_summary = "\n".join(
            [f"Room: {room['name']}, Sizes: {', '.join(room['sizes']) if room['sizes'] else 'Size not mentioned'}"
             for rooms in matched_rooms.values() for room in rooms]
        )
        relevant_texts = query_faiss_index(question, faiss_index, pdf_texts)
        pdf_context = "\n\n".join([str(text) for text in relevant_texts])
        prompt = f"{context}\nRelevant chat context:\n{chat_context}\n\nThe user asked: '{question}'.\nHere are the detected rooms:\n{room_summary}\n\nRelevant information from architectural PDFs:\n{pdf_context}"
    else:
        context = "You are a helpful assistant."
        relevant_texts = query_faiss_index(question, faiss_index, pdf_texts)
        pdf_context = "\n\n".join([str(text) for text in relevant_texts])
        prompt = f"{context} Answer based on the following:\n{pdf_context}\n\nThe user asked: '{question}'."
        print(prompt)

    return prompt

def ask_llama(question, matched_rooms):
    # Access from st.session_state
    faiss_index = st.session_state["faiss_index"]
    pdf_texts = st.session_state["pdf_texts"]
    
    full_prompt = generate_prompt_with_memory(question, matched_rooms)
    try:
        response = ollama.generate(model="llama3.2", prompt=full_prompt)
        logging.info(f"Ollama response: {response}")
        return response.get('response', "Oops! I didn't quite catch that. Could you try rephrasing?")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return "Sorry, something went wrong while processing. Let me know if you'd like to try again."

    

# Replace with your Hugging Face API token
HUGGING_FACE_API_TOKEN = "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"

headers = {
    "Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}"
}

# Function to enhance the entire floorplan with furnishings for all rooms
def generate_floorplan(description):
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    detailed_description = (
        f".Generate a floorplan that is in 2D and is realistic with furniture and make the design clean, organized floorplan with clearly defined "
        "rooms.Follow this description: {description}"
    )
    
    payload = {"inputs": detailed_description}
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        
        # Check if the response is valid and save the image if successful
        if response.status_code == 200:
            image_path = "improved_floorplan.png"
            with open(image_path, "wb") as f:
                f.write(response.content)
            return image_path
        else:
            logging.error(f"Failed to generate image: {response.text}")
            return None
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return None

# Function to display the generated floorplan
def display_generated_floorplan(image_path):
    st.image(image_path, caption="Improved Floorplan Generated by Stable Diffusion", use_column_width=True)


# Function to preprocess the image for edge detection
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray)
    edges = cv2.Canny(enhanced_img, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return morphed

# Function to clean and validate detected text
def clean_text(text):
    if text is None:
        return None
    text = text.strip().upper()  # Convert to uppercase and remove leading/trailing spaces
    corrected_text = spell.correction(text) or text  # Spell correction fallback to original

    # Allow letters, spaces, and ensure it doesn't match empty strings
    if re.match(r'^[A-Z][A-Z\s]+$', corrected_text) and len(corrected_text) > 1:
        return corrected_text
    return None


# Function to extract text using PaddleOCR
def extract_text_paddleocr(image):
    try:
        result = ocr_model.ocr(image, cls=True)
        if not result or not isinstance(result, list):
            logging.error("OCR model returned None or unexpected format.")
            return []  # Return an empty list to avoid further errors

        text_output = []
        for line in result:
            if not isinstance(line, list):
                logging.warning("Unexpected format in OCR line result.")
                continue

            for box, (text, confidence) in line:
                if confidence > 0.6 and text:
                    text_output.append((box, text))
                else:
                    logging.warning("Detected text is None or confidence is low.")

        return text_output
    except Exception as e:
        logging.error(f"Error in extract_text_paddleocr: {e}")
        return []  # Return an empty list in case of error


# Function to detect room boundaries
def detect_room_boundaries(image):
    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    room_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000:
            x, y, w, h = cv2.boundingRect(contour)
            room_contours.append((x, y, w, h))
    return room_contours

# Function to match detected text to room sizes
def match_rooms_to_sizes(room_contours, text_output):
    rooms = {}
    room_instance_count = {}

    # Updated regex pattern to detect various size formats
    size_patterns = [
        r'\d+\s?x\s?\d+',                 # Matches "11x11", "12 x 13"
        r'\d+\s?(?:m²|ft²)',              # Matches "12m²", "12ft²"
        r'\d+\s?(?:m\^2|ft\^2)',          # Matches "12m^2", "12ft^2"
        r'\d+(\.\d+)?\s?sq\s?ft',         # Matches "90.27 sq ft", "90 sq ft"
    ]
    size_regex = re.compile("|".join(size_patterns), re.IGNORECASE)

    for i, (bbox, text) in enumerate(text_output):
        clean_name = clean_text(text)  # Process the room name

        if clean_name:
            room_x, room_y = bbox[0][0], bbox[0][1]
            if clean_name not in room_instance_count:
                room_instance_count[clean_name] = 0
            else:
                room_instance_count[clean_name] += 1

            room_key = f"{clean_name}_{room_instance_count[clean_name]}"
            rooms[room_key] = {
                "name": clean_name,
                "sizes": [],
                "bbox": bbox,
                "text_elements": [(bbox, text)],
                "coordinates": (room_x, room_y),
                "color": (0, 255, 0),
                "font_size": 0.9
            }

            # Look for the next text that could represent the room size
            for j in range(i + 1, len(text_output)):
                next_bbox, next_text = text_output[j]
                next_y = next_bbox[0][1]

                # Check for text proximity and size patterns
                if next_y > room_y and (next_y - room_y) < 50:
                    if size_regex.search(next_text):  # Check if text matches any size format
                        rooms[room_key]["sizes"].append(next_text)
                        rooms[room_key]["text_elements"].append((next_bbox, next_text))
                        break

    for room_key, details in rooms.items():
        if not details["sizes"]:
            details["sizes"].append(f"Size not mentioned, Coordinates: {details['coordinates']}")

    named_rooms = {}
    for room_key, details in rooms.items():
        name = details["name"]
        if name not in named_rooms:
            named_rooms[name] = []
        named_rooms[name].append(details)

    return named_rooms



# Function to rename room sizes based on the command
def rename_room_size(command, rooms, original_image):
    updated_image = original_image.copy()
    # Enhanced regex to support multiple size formats: '11x11', '90.27 sq ft', '12m²', etc.
    size_match = re.search(
        r"rename\s+(\w+(?:\s+\w+)*)\s+size\s+(\d+\s?x\s?\d+|\d+(\.\d+)?\s?sq\s?ft|\d+\s?(m²|ft²|m\^2|ft\^2))\s+to\s+(\d+\s?x\s?\d+|\d+(\.\d+)?\s?sq\s?ft|\d+\s?(m²|ft²|m\^2|ft\^2))",
        command, re.IGNORECASE
    )

    if size_match:
        room_name, old_size, _, _, new_size, _, _ = size_match.groups()
        room_name, old_size, new_size = room_name.upper(), old_size.upper(), new_size.upper()

        if room_name in rooms:
            # Find the room entry that contains both the specified room name and old size
            room_info = next((room for room in rooms[room_name] if old_size in [s.upper() for s in room["sizes"]]), None)
            if room_info:
                # Locate the bounding box of the old size text
                size_bbox = None
                for bbox, text in room_info["text_elements"]:
                    if text.upper() == old_size:  # Match size text case-insensitively
                        size_bbox = bbox
                        break

                if size_bbox:
                    # Calculate dimensions of bounding box and background color to overlay
                    x, y, w, h = cv2.boundingRect(np.array(size_bbox, dtype=np.int32))
                    margin = 5
                    bg_area = updated_image[max(y - margin, 0): y + h + margin, x - margin: x + w + margin]
                    bg_color = tuple(map(int, np.median(bg_area, axis=(0, 1)))) if bg_area.size > 0 else (255, 255, 255)

                    # Erase the old size text
                    cv2.rectangle(updated_image, (x, y), (x + w, y + h), bg_color, -1)

                    # Add the new size text at the same location
                    font_scale = room_info.get("font_size", 0.9)
                    cv2.putText(updated_image, new_size, (x, y + h - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

                    # Update the room sizes in the room's details
                    room_info["sizes"] = [new_size if s.upper() == old_size else s for s in room_info["sizes"]]
                    response = f"Updated size of '{room_name}' from '{old_size}' to '{new_size}' in the blueprint."
                else:
                    response = f"Could not locate bounding box for size '{old_size}' in room '{room_name}'."
            else:
                response = f"Room '{room_name}' with size '{old_size}' not found. Ensure the room and size are correctly detected in the blueprint."
        else:
            response = f"Room '{room_name}' not found. Please check if it's detected in the blueprint."
    else:
        response = "Could not understand the size rename command. Use the format: 'rename <room name> size <old size> to <new size>' where sizes can be '11x11', '90.27 sq ft', '12m²', etc."

    return response, rooms, updated_image


def rename_room(command, rooms, original_image):
    updated_image = original_image.copy()

    # Utility function to erase text
    def erase_text(bbox, image):
        x, y, w, h = cv2.boundingRect(np.array([list(point) for point in bbox], dtype=np.int32))
        margin = 5
        bg_area = image[max(y - margin, 0): y + h + margin, x - margin: x + w + margin]
        bg_color = tuple(map(int, np.median(bg_area, axis=(0, 1)))) if bg_area.size > 0 else (255, 255, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), bg_color, -1)
        return x, y, w, h

    # Utility function to draw text
    def draw_text(image, text, x, y, w, h, font_scale, font_color=(0, 0, 0), thickness=2):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = x + (w - text_size[0]) // 2  # Center text horizontally
        text_y = y + (h + text_size[1]) // 2  # Center text vertically
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)

    # Case 1: Renaming room by coordinates
    coord_match = re.search(r"rename\s+(\w+)\s+coordinates:\s*\((\d+\.?\d*),\s*(\d+\.?\d*)\)\s+to\s+(\w+)", command, re.IGNORECASE)
    if coord_match:
        room_name, x, y, new_name = coord_match.groups()
        x, y = float(x), float(y)
        room_name = room_name.upper()
        new_name = new_name.upper()
        renamed = False

        if room_name in rooms:
            for room_info in rooms[room_name]:
                if room_info["coordinates"] == (x, y):
                    bbox = room_info['bbox']
                    x, y, w, h = erase_text(bbox, updated_image)

                    # Add new text with consistent font and size
                    font_scale = room_info.get("font_size", 0.9)
                    draw_text(updated_image, new_name, x, y, w, h, font_scale)

                    # Update metadata
                    room_info["name"] = new_name
                    renamed = True
                    response = f"Renamed room at coordinates ({x}, {y}) from '{room_name}' to '{new_name}'."
                    break

        if not renamed:
            response = f"No room found at coordinates ({x}, {y}) named '{room_name}'."

    # Case 2: Renaming room by size
    else:
        size_match = re.search(r"rename\s+(\w+(?:\s+\w+)*)\s+(\d+\s?x\s?\d+|\d+(\.\d+)?\s?sq\s?ft|\d+\s?(m²|ft²|m\^2|ft\^2))\s+to\s+(\w+(?:\s+\w+)*)", command, re.IGNORECASE)
        if size_match:
            old_name, old_size, _, _, new_name = size_match.groups()
            old_name, old_size, new_name = old_name.upper(), old_size.upper(), new_name.upper()

            if old_name in rooms:
                room_info = next((room for room in rooms[old_name] if old_size in [s.upper() for s in room["sizes"]]), None)
                if room_info:
                    bbox = room_info['bbox']
                    x, y, w, h = erase_text(bbox, updated_image)

                    # Add new text with consistent font and size
                    font_scale = room_info.get("font_size", 0.9)
                    draw_text(updated_image, new_name, x, y, w, h, font_scale)

                    # Update metadata
                    room_info["name"] = new_name
                    response = f"Renamed room '{old_name}' with size '{old_size}' to '{new_name}' in the blueprint."
                else:
                    response = f"Could not locate room '{old_name}' with size '{old_size}' in the blueprint."
            else:
                response = f"Room '{old_name}' not found. Ensure the room name is correct."
        else:
            response = "Could not understand the rename command. Use the format: 'rename <room name> <size> to <new name>' where size can be '11x11', '90.27 sq ft', or '12m²'."

    return response, rooms, updated_image

# Function to display detected rooms and sizes in the desired format
def display_detected_rooms(rooms):
    st.write("### Detected Room Names and Sizes")
    for room_name, room_details in rooms.items():
        for room in room_details:
            size_text = ", ".join(room['sizes']) if room['sizes'] else "Size not mentioned"
            st.write(f"Room Name: {room['name']}, Size: {size_text}")
            

# -------------------------
# Plagiarism Detection 
# -------------------------

DATASET_PATH3 = "./dataset3"

# --------------------------------------
# Build ORB Feature Database
# --------------------------------------
def build_orb_database(dataset_path):
    """Build a database of ORB features for all images in the dataset."""
    orb = cv2.ORB_create(nfeatures=2000)
    database = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    kp, des = orb.detectAndCompute(img, None)
                    if des is not None and len(des) > 0:
                        database.append((img_path, kp, des))
    return database

def load_image_for_cv(image_path: str):
    img_pil = Image.open(image_path).convert("RGB")
    img_pil = ImageOps.exif_transpose(img_pil)
    img = np.array(img_pil)
    # Convert to BGR for consistency with OpenCV
    img = img[:, :, ::-1]
    return img

def orb_similarity(uploaded_img_path, database, ratio_threshold=0.75, similarity_threshold=0.5):
    """
    Compare the uploaded image's ORB features against the database.
    Returns a list of (img_path, similarity_score, matched_count, min_keypoints) 
    for suspicious matches, where:
    - similarity_score = matched_count / min_keypoints
    """
    orb = cv2.ORB_create(nfeatures=2000)
    uploaded_img = load_image_for_cv(uploaded_img_path)
    kp_u, des_u = orb.detectAndCompute(uploaded_img, None)
    if des_u is None or len(des_u) == 0:
        # No features found in uploaded image, can't detect plagiarism meaningfully
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    suspicious_matches = []

    for (img_path, kp_d, des_d) in database:
        # KNN match
        matches = bf.knnMatch(des_u, des_d, k=2)
        good = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good.append(m)

        min_keypoints = min(len(des_u), len(des_d))
        if min_keypoints == 0:
            continue
        score = len(good) / min_keypoints
        if score >= similarity_threshold:
            suspicious_matches.append((img_path, score, len(good), min_keypoints))

    return suspicious_matches

# Build the ORB database once
if "orb_database" not in st.session_state:
    st.session_state["orb_database"] = build_orb_database(DATASET_PATH3)

def check_plagiarism_advanced(uploaded_image_path):
    matches = orb_similarity(uploaded_image_path, st.session_state["orb_database"], ratio_threshold=0.75, similarity_threshold=0.5)
    report = {
        "plagiarism_detected": len(matches) > 0,
        "matches": matches
    }
    return report



# Initialize or load chat memory
if "chat_tabs" not in st.session_state:
    st.session_state["chat_tabs"] = []

if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = {
        "name": "Chat 1",
        "history": [],
        "texts": [],
        "embeddings": [],
        "uploaded_file": None,
    }
    st.session_state["chat_tabs"].append(st.session_state["current_chat"])

# -------------------------
# Streamlit UI
# -------------------------
st.title("Architecture AI Assistant")
st.markdown("### Upload your blueprint to analyze room details and get instant feedback or improvements!")

uploaded_file = st.file_uploader("Upload a blueprint (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.session_state.current_chat["uploaded_file"] = uploaded_file

st.sidebar.title("Chat Sessions")
if st.sidebar.button("Create New Chat"):
    new_chat = {
        "name": f"Chat {len(st.session_state.chat_tabs) + 1}",
        "history": [],
        "texts": [],
        "embeddings": [],
        "faiss_index": faiss.IndexFlatL2(384),
        "uploaded_file": None,
        "pdf_texts": []
    }
    st.session_state.chat_tabs.append(new_chat)
    st.session_state.current_chat = new_chat

for idx, chat in enumerate(st.session_state.chat_tabs):
    if st.sidebar.button(chat["name"]):
        st.session_state.current_chat = chat

st.sidebar.write(f"Current Chat: {st.session_state.current_chat['name']}")

if uploaded_file:
    st.write("### Original Blueprint")
    image_pil = Image.open(uploaded_file).convert("RGB")
    img = np.array(image_pil)
    st.image(img, caption="Uploaded Blueprint", use_column_width=True)

    preprocessed_image = preprocess_image(img)
    room_contours = detect_room_boundaries(preprocessed_image)
    extracted_text = extract_text_paddleocr(img)
    matched_rooms = match_rooms_to_sizes(room_contours, extracted_text)
    display_detected_rooms(matched_rooms)

    user_question = st.chat_input(
        "Examples: 'Rename Living Room 12x12 to Lounge', 'Provide feedback on blueprint', 'Suggest an improved layout', or 'Check plagiarism'."
    )
    
    if user_question:
        uploaded_image_path = "./uploaded_image.jpg"
        with open(uploaded_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if "plagiarism" in user_question.lower():
            report = check_plagiarism_advanced(uploaded_image_path)
            if report["plagiarism_detected"]:
                # Sort matches by highest similarity first
                sorted_matches = sorted(report["matches"], key=lambda x: x[1], reverse=True)
                top_match_path, top_match_score, top_matched_count, top_min_keypoints = sorted_matches[0]

                similarity_percentage = top_match_score * 100
                matched_feature_ratio = f"{top_matched_count}/{top_min_keypoints}"

                if top_match_score >= 0.8:
                    improvement_suggestion = (
                        "Consider altering distinctive features, such as unique room outlines, door/window placements, or decorative elements. "
                        "Reduce repetitive patterns or distinctive markers that appear frequently, so fewer key features match known projects."
                    )
                elif top_match_score >= 0.6:
                    improvement_suggestion = (
                        "Try adjusting the floorplan's scale or orientation, introduce unique design elements, "
                        "and remove or alter repeated patterns that closely match existing projects."
                    )
                else:
                    improvement_suggestion = (
                        "Introduce more unique elements or variation in room shapes, furniture placement, or structural details. "
                        "Avoid using standard components that appear frequently in existing designs."
                    )

                response_text = "Plagiarism Detected!\n\n"
                response_text += f"Reason: The uploaded blueprint shares {similarity_percentage:.1f}% of its key architectural features with a known project.\n"
                response_text += f"A total of {matched_feature_ratio} distinctive features overlapped after applying ratio tests (Lowe's test with ratio={0.75}).\n\n"
                response_text += "This high overlap suggests the uploaded blueprint may be derived from an existing design.\n\n"
                response_text += "How to Improve:\n"
                response_text += improvement_suggestion + "\n\n"
                response_text += "Similar Images Found:\n"
                response_text += f"- {os.path.basename(top_match_path)}: approximately {similarity_percentage:.1f}% of key features match.\n"

                similar_img = Image.open(top_match_path).convert("RGB")

                # Add images to the chat history as well
                add_to_chat_history(
                    user_question,
                    {
                        "ai_response_text": response_text,
                        "images": [similar_img]  # Including the similar image in the chat history
                    }
                )

            else:
                response_text = ("No plagiarism detected.\n\n"
                                 "Reason: The uploaded image does not share enough distinctive local architectural features "
                                 "with any known projects in our dataset. Consider this as a unique or sufficiently distinct design.")
                add_to_chat_history(user_question, {"ai_response_text": response_text})

        elif user_question.lower().startswith("i want") or "where is this" in user_question.lower():
            project_folder = find_image_folder(uploaded_image_path, DATASET_PATH2)
            if project_folder:
                st.success(f"Image found in project: {os.path.basename(project_folder)}")
                st.markdown("### Images in the Project Folder:")
                image_list = []
                for root, _, files in os.walk(project_folder):
                    for file in files:
                        if file.lower().endswith((".png", ".jpg", ".jpeg")):
                            image_path = os.path.join(root, file)
                            image_list.append(image_path)  # Save image paths for history

                pdf_content = process_project_pdf(project_folder)

                prompt_with_context = (
                    f"User uploaded an image from project '{os.path.basename(project_folder)}'. {user_question}\n\n"
                    f"PDF Content:\n{pdf_content}"
                )
                try:
                    response = ollama.generate(model="llama3.2", prompt=prompt_with_context)
                    ai_response_text = response.get('response', "Error generating response.")
                except Exception as e:
                    st.error("Failed to generate response from Llama.")
                    ai_response_text = "Failed to generate AI response."

                combined_response = {
                    "images": image_list,
                    "pdf_content": pdf_content,
                    "ai_response_text": ai_response_text
                }

                # Add interaction to chat history
                add_to_chat_history(user_question, combined_response)
            else:
                st.error("Image not found in the dataset.")
                add_to_chat_history(user_question, {"error": "Image not found in the dataset."})

        elif user_question.lower().startswith("rename"):
            if "size" in user_question.lower():
                response, matched_rooms, updated_image = rename_room_size(user_question, matched_rooms, img)
            else:
                response, matched_rooms, updated_image = rename_room(user_question, matched_rooms, img)

            ai_response_text = response
            images = [updated_image] if updated_image is not None else []
            combined_response = {
                "ai_response_text": ai_response_text,
                "images": images,
                "pdf_content": None
            }
            add_to_chat_history(user_question, combined_response)

        elif "improve" in user_question.lower():
            response = ask_llama(user_question, matched_rooms)
            improved_floorplan_path = generate_floorplan(response)

            if improved_floorplan_path:
                add_to_chat_history(
                    user_question,
                    {
                        "images": [improved_floorplan_path],
                        "ai_response_text": "Generated a enhanced layout suggestion based on the original blueprint."
                    }
                )
            else:
                st.write("Failed to enhance the layout. Please try again with a different description.")
                add_to_chat_history(
                    user_question,
                    {"ai_response_text": "Failed to enhance the layout."}
                )

        else:
            response = ask_llama(user_question, matched_rooms)
            response_dict = {"ai_response_text": response} if isinstance(response, str) else response
            add_to_chat_history(user_question, response_dict)
            



 

# Chat history section
st.write("### Chat")
for chat in st.session_state.current_chat["history"]:
    user_input = chat.get("user_input", "Unknown")
    ai_response = chat.get("ai_response", {})

    with st.chat_message("user"):
        st.markdown(user_input)

    if isinstance(ai_response, dict):
        pdf_content = ai_response.get("pdf_content", "")
        ai_text = ai_response.get("ai_response_text", "")
        images = ai_response.get("images", [])

        with st.chat_message("assistant"):
            if ai_text:
                st.markdown(ai_text)
            if pdf_content:
                st.markdown("**Extracted PDF Content:**")
                st.write(pdf_content)
            for im in images:
                if isinstance(im, str) and os.path.exists(im):
                    st.image(im, caption="Project Image", use_column_width=True)
                else:
                    st.image(im, caption="Project Image", use_column_width=True)
    else:
        with st.chat_message("assistant"):
            st.markdown(ai_response)

save_all_chats()
