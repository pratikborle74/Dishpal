import streamlit as st
import google.generativeai as genai
import os
import bcrypt
import json
from pymongo import MongoClient
from dotenv import load_dotenv
import datetime
import uuid
import re
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import time
from PIL import Image
import requests
import base64
import io
import random
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np

# Load environment variables
load_dotenv()

# Initialize MongoDB
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client.recipe_app
users = db.users
recipes = db.recipes  # Collection for recipes
fridge_collection = db.fridge  # Collection for fridge ingredients
reviews = db.reviews  # Collection for customer reviews

# Load API Keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize APIs
genai.configure(api_key=GEMINI_API_KEY)

# Initialize MobileNetV2 model for ingredient recognition
@st.cache_resource
def load_model():
    """Initialize MobileNetV2 model for ingredient recognition"""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.eval()
    return model

# Load ImageNet class labels
@st.cache_data
def load_labels():
    """Load ImageNet class labels"""
    labels_path = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(labels_path)
    return [line.strip() for line in response.text.split('\n')]

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'current_recipe' not in st.session_state:
    st.session_state.current_recipe = None
if 'generated_recipes' not in st.session_state:
    st.session_state.generated_recipes = []
if 'recipe_saved' not in st.session_state:
    st.session_state.recipe_saved = False

def save_review(username, star_rating, review_text):
    """Save a customer review to the database."""
    try:
        reviews.insert_one({
            'username': username,
            'star_rating': star_rating,
            'review': review_text,
            'created_at': datetime.datetime.now()
        })
        return True
    except Exception as e:
        st.error(f"Error saving review: {str(e)}")
        return False
    
def display_reviews():
    """Display all customer reviews."""
    st.subheader("Customer Reviews")
    all_reviews = list(reviews.find().sort('created_at', -1))  # Fetch reviews sorted by date
    if all_reviews:
        for review in all_reviews:
            st.markdown(f"**{review['username']}**: {'⭐' * review['star_rating']}")  # Display star rating
            st.markdown(f"*{review['review']}*")  # Display the written review
            st.markdown(f"*{review['created_at'].strftime('%Y-%m-%d %H:%M:%S')}*")
            st.markdown("---")
    else:
        st.info("No reviews yet. Be the first to share your feedback!")

def calculate_average_rating():
    """Calculate the average rating from the reviews."""
    all_reviews = list(reviews.find())
    if all_reviews:
        total_rating = sum(review['star_rating'] for review in all_reviews)
        average_rating = total_rating / len(all_reviews)
        return average_rating
    return None


def signup():
    st.subheader("Create New Account")
    new_username = st.text_input("Username", key="signup_username")
    new_password = st.text_input("Password", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Sign Up"):
        if new_password != confirm_password:
            st.error("Passwords do not match!")
            return
        
        if users.find_one({"username": new_username}):
            st.error("Username already exists!")
            return
        
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        users.insert_one({
            "username": new_username,
            "password": hashed_password
        })
        st.success("Account created successfully! Please log in.")

def login():
    st.subheader("Login to Your Account")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login"):
        user = users.find_one({"username": username})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.page = 'home'
    st.rerun()

def navigate_to_recipe(recipe_id):
    """Navigate to recipe page"""
    st.session_state.page = 'recipe'
    st.session_state.current_recipe = recipe_id
    st.rerun()

def navigate_to_home():
    """Navigate back to home page"""
    st.session_state.page = 'home'
    st.session_state.current_recipe = None
    st.rerun()

def parse_recipe(recipe_text):
    """Parse recipe text to extract title, ingredients, and instructions"""
    # Clean up the text
    recipe_text = recipe_text.strip()
    
    # Split into lines and clean them
    lines = [line.strip() for line in recipe_text.split('\n') if line.strip()]
    
    # Initialize sections
    title = None
    ingredients = []
    instructions = []
    current_section = None
    calories = None
    
    # Try to find title based on "Total Calories" first (for surprise me recipes)
    for i, line in enumerate(lines):
        # Look for Total Calories line
        if 'total calories:' in line.lower():
            # Title is the line before Total Calories
            if i > 0:  # Make sure we have a previous line
                title = lines[i-1].strip()
                # Clean up the title
                title = re.sub(r'^\d+\.\s*', '', title)  # Remove leading numbers
                title = re.sub(r'[*,.!:;"\']', '', title)  # Remove special characters
                title = re.sub(r'\s*[-–—]\s*.*$', '', title)  # Remove everything after any dash
                title = re.sub(r'\s*\(.*\)', '', title)  # Remove parenthetical notes
                title = title.strip()
                
                # Extract calories from this line
                calorie_match = re.search(r'total calories:\s*(\d+)', line.lower())
                if calorie_match:
                    calories = calorie_match.group(1)
                break
    
    # If title wasn't found using Total Calories, try the original method
    if not title:
        # Find the ingredients section marker
        for i, line in enumerate(lines):
            if 'ingredients:' in line.lower() or line.lower().startswith('**ingredients**'):
                # Title is the line before ingredients
                if i > 0:  # Make sure we have a previous line
                    title = lines[i-1].strip()
                    # Clean up the title
                    title = re.sub(r'^\d+\.\s*', '', title)  # Remove leading numbers
                    title = re.sub(r'[*,.!:;"\']', '', title)  # Remove special characters
                    title = re.sub(r'\s*[-–—]\s*.*$', '', title)  # Remove everything after any dash
                    title = re.sub(r'\s*\(.*\)', '', title)  # Remove parenthetical notes
                    title = title.strip()
                start_idx = i
                break
    
    # Find the start index for processing sections if not set
    if 'start_idx' not in locals():
        for i, line in enumerate(lines):
            if 'ingredients:' in line.lower() or line.lower().startswith('**ingredients**'):
                start_idx = i
                break
    
    # Process remaining lines for ingredients and instructions
    in_ingredients = False
    in_instructions = False
    
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        
        line_lower = line.lower()
        
        # Check for section headers
        if any(header in line_lower for header in ['ingredients:', '**ingredients**']):
            in_ingredients = True
            in_instructions = False
            continue
        elif 'instruct' in line_lower:  # Changed to catch any variation of 'instructions'
            in_ingredients = False
            in_instructions = True
            continue
        
        # Add content to appropriate section
        if in_ingredients:
            # Only add if it's not a header
            if not any(header in line_lower for header in ['ingredients:', '**ingredients**']):
                ingredients.append(line)
        elif in_instructions:
            # Only add if it's not a header
            if not 'instruct' in line_lower:  # Changed to match the header check
                # Clean up numbered steps if present
                step = re.sub(r'^\d+\.\s*', '', line)  # Remove leading numbers
                instructions.append(step.strip())
        elif line.startswith(('•', '*', '-')) or re.search(r'\d+\s*(?:cup|tbsp|tsp|g|kg|ml|l|oz|lb|pound)', line_lower):
            # If line looks like an ingredient but we haven't seen the header yet
            ingredients.append(line)
    
    # Format sections
    ingredients_text = '\n'.join(ingredients).strip()
    instructions_text = '\n'.join(instructions).strip()
    
    return {
        'title': title if title else 'Untitled Recipe',
        'ingredients': ingredients_text,
        'instructions': instructions_text,
        'full_text': recipe_text,
        'calories': calories
    }

def split_multiple_recipes(full_text, num_recipes=None):
    """Split text containing multiple recipes into individual recipes"""
    # First, clean up the text
    full_text = full_text.strip()
    
    # Try splitting by numbered sections first
    recipe_starts = [0]  # Start of first recipe
    
    # Find all potential recipe start positions
    for match in re.finditer(r'\n(?=\d+\.\s+\*\*.*?\*\*)', full_text):
        if match.start() > 0:  # Don't add the first match as it's already covered by 0
            recipe_starts.append(match.start())
    
    # If we found recipe breaks
    if len(recipe_starts) > 1:
        recipes = []
        # Add end of text for the last slice
        recipe_starts.append(len(full_text))
        
        # Extract each recipe using the start positions
        for i in range(len(recipe_starts) - 1):
            recipe = full_text[recipe_starts[i]:recipe_starts[i + 1]].strip()
            if recipe:
                recipes.append(recipe)
    else:
        # If no numbered sections found, try splitting by double newlines
        recipes = [r.strip() for r in full_text.split('\n\n\n') if r.strip()]
    
    # Clean up each recipe
    cleaned_recipes = []
    for recipe in recipes:
        # Remove any leading/trailing whitespace
        recipe = recipe.strip()
 # Skip if recipe is too short (likely a section of another recipe)
        if len(recipe.split('\n')) < 3:
            continue
        
        # Ensure recipe has both ingredients and instructions
        if 'ingredients' in recipe.lower() and ('instructions' in recipe.lower() or 'cooking instructions' in recipe.lower() or 'method' in recipe.lower() or 'steps' in recipe.lower() or 'directions' in recipe.lower()):
            cleaned_recipes.append(recipe)
    
    # If we still don't have enough recipes, try one last method
    if len(cleaned_recipes) < (num_recipes or 1):
        # Look for clear section markers
        recipe_blocks = re.split(r'\n(?=\s*(?:Ingredients:|INGREDIENTS:|**Ingredients**))', full_text)
        if len(recipe_blocks) > len(cleaned_recipes):
            cleaned_recipes = [block.strip() for block in recipe_blocks if block.strip()]
    
    # Return the requested number of recipes
    return cleaned_recipes[:num_recipes] if num_recipes is not None else cleaned_recipes

def save_recipe(recipe_text, user, diet, meal, cuisine, calories, image=None):
    """Save recipe to database"""
    parsed_recipe = parse_recipe(recipe_text)
    recipe_id = str(uuid.uuid4())
    
    recipe_data = {
        'id': recipe_id,
        'user': user,
        'title': parsed_recipe['title'],
        'ingredients': parsed_recipe['ingredients'],
        'instructions': parsed_recipe['instructions'],
        'full_text': recipe_text,
        'diet': diet,
        'meal': meal,
        'cuisine': cuisine,
        'calories': calories,
        'created_at': datetime.datetime.now()
    }
    
    # Add image if available
    if image:
        recipe_data['image'] = image
    
    try:
        recipes.insert_one(recipe_data)
        return recipe_id
    except Exception as e:
        st.error(f"⚠️ Error saving recipe: {str(e)}")
        return None

def save_individual_recipe(index):
    """Save a single recipe from the generated recipes list"""
    if 0 <= index < len(st.session_state.generated_recipes):
        recipe_info = st.session_state.generated_recipes[index]
        
        # Add image to the recipe object if it exists
        recipe_data = {
            'text': recipe_info['text'],
            'diet': recipe_info['diet'],
            'meal': recipe_info['meal'],
            'cuisine': recipe_info['cuisine'],
            'calories': recipe_info['calories']
        }
        
        # Include the image if it exists
        if 'image' in recipe_info and recipe_info['image']:
            recipe_data['image'] = recipe_info['image']
            
        recipe_id = save_recipe(
            recipe_data['text'],
            st.session_state.username,
            recipe_data['diet'],
            recipe_data['meal'],
            recipe_data['cuisine'],
            recipe_data['calories'],
            recipe_data.get('image')  # Pass image if available
        )
        
        if recipe_id:
            # Mark this recipe as saved
            st.session_state.generated_recipes[index]['saved'] = True
            st.session_state.recipe_saved = True  # Set a flag to indicate successful save
        else:
            st.error("Failed to save recipe. Please try again.")
    else:
        st.warning("Invalid recipe index.")

def generate_recipe_pdf(recipe):
    """Generate PDF for recipe using ReportLab"""
    try:
        # Create a file-like buffer to receive PDF data
        buffer = BytesIO()
        
        # Create the PDF object using the buffer as its "file"
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=72)
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Title'],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=12
        )
        heading_style = ParagraphStyle(
            'HeadingStyle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6
        )
        normal_style = styles["Normal"]
        normal_style.fontSize = 10
        
        # Recipe metadata style
        info_style = ParagraphStyle(
            'InfoStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.brown,
            alignment=TA_CENTER
        )
        
        # Footer style
        footer_style = ParagraphStyle(
            'FooterStyle',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            alignment=TA_CENTER
        )
        
        # Helper function to sanitize text for PDF
        def sanitize_text(text):
            if not text:
                return ""
            # Replace problematic sequences
            text = text.replace('<', '&lt;').replace('>', '&gt;')
            # Replace any other problematic sequences
            text = text.replace('<para>', '').replace('</para>', '')
            text = text.replace('<br>', '<br/>').replace('</br>', '')
            return text
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Add logo at the top
        logo_path = "assets/logo.png"
        if os.path.exists(logo_path):
            try:
                # Open and resize logo
                logo_img = Image.open(logo_path)
                # Calculate aspect ratio to maintain proportions
                aspect = logo_img.width / logo_img.height
                target_width = 111  # pixels (increased 3x from 37)
                target_height = int(target_width / aspect)
                logo_img = logo_img.resize((target_width, target_height))
                
                # Convert PIL Image to bytes
                logo_buffer = BytesIO()
                logo_img.save(logo_buffer, format='PNG')
                logo_buffer.seek(0)
                
                # Add logo to PDF
                logo = ReportLabImage(logo_buffer, width=1.35*inch, height=(1.35*inch/aspect))  # increased 3x from 0.45*inch
                elements.append(logo)
                elements.append(Spacer(1, 0.25 * inch))
            except Exception as e:
                # If logo fails, continue without it
                pass
        
        # Add the title (with error handling)
        title = recipe.get('title', 'Untitled Recipe')
        if not isinstance(title, str):
            title = str(title)
        title = sanitize_text(title)
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 0.25 * inch))
        
        # Add recipe image if available
        if 'image' in recipe and recipe['image']:
            try:
                # Create a temporary file for the image
                img_buffer = BytesIO(recipe['image'])
                img = ReportLabImage(img_buffer, width=5*inch, height=3.75*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.2 * inch))
            except Exception as e:
                # If image fails, just continue without it
                pass
                
        # Add recipe metadata as a table (with error handling)
        data = [
            ["Diet", "Meal Type", "Cuisine", "Calories"],
            [
                sanitize_text(str(recipe.get('diet', 'N/A'))),
                sanitize_text(str(recipe.get('meal', 'N/A'))),
                sanitize_text(str(recipe.get('cuisine', 'N/A'))),
                sanitize_text(str(recipe.get('calories', 'N/A')))
            ]
        ]
        
        t = Table(data, colWidths=[1.1 * inch] * 4)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (3, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (3, 0), colors.brown),
            ('ALIGN', (0, 0), (3, 0), 'CENTER'),
            ('ALIGN', (0, 1), (3, 1), 'CENTER'),
            ('FONTNAME', (0, 0), (3, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (3, 0), 12),
            ('TOPPADDING', (0, 1), (3, 1), 12),
            ('GRID', (0, 0), (3, 1), 1, colors.black),
            ('BOX', (0, 0), (3, 1), 1, colors.black),
        ]))
        
        elements.append(t)
        elements.append(Spacer(1, 0.25 * inch))
        
        # Add Nutritional Information section
        elements.append(Paragraph("Nutritional Information", heading_style))
        
        # Extract macros using regex with error handling
        full_text = recipe.get('full_text', '')
        if not isinstance(full_text, str):
            full_text = str(full_text)
        
        protein_match = re.search(r'protein:?\s*(\d+)\s*g', full_text, re.IGNORECASE)
        carbs_match = re.search(r'carbs?:?\s*(\d+)\s*g', full_text, re.IGNORECASE)
        fat_match = re.search(r'fat:?\s*(\d+)\s*g', full_text, re.IGNORECASE)
        
        # Create macros table with better error handling
        macros_data = [
            ["Calories", "Protein", "Carbs", "Fat"],
            [
                sanitize_text(str(recipe.get('calories', 'N/A'))),
                sanitize_text(f"{protein_match.group(1)}g" if protein_match else 'N/A'),
                sanitize_text(f"{carbs_match.group(1)}g" if carbs_match else 'N/A'),
                sanitize_text(f"{fat_match.group(1)}g" if fat_match else 'N/A')
            ]
        ]
        
        macros_table = Table(macros_data, colWidths=[1.1 * inch] * 4)
        macros_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (3, 0), colors.chocolate),
            ('TEXTCOLOR', (0, 0), (3, 0), colors.white),
            ('ALIGN', (0, 0), (3, 1), 'CENTER'),
            ('FONTNAME', (0, 0), (3, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (3, 1), 10),
            ('BOTTOMPADDING', (0, 0), (3, 0), 8),
            ('TOPPADDING', (0, 1), (3, 1), 8),
            ('GRID', (0, 0), (3, 1), 1, colors.black),
            ('BOX', (0, 0), (3, 1), 1, colors.black),
        ]))
        
        elements.append(macros_table)
        elements.append(Spacer(1, 0.25 * inch))
        
        # Ingredients section (with error handling)
        elements.append(Paragraph("Ingredients", heading_style))
        ingredients = recipe.get('ingredients', '')
        if not isinstance(ingredients, str):
            ingredients = str(ingredients)
        
        ingredients_lines = ingredients.split('\n')
        for line in ingredients_lines:
            if line.strip():
                # Skip section headers and macro information
                if not any(x in line.lower() for x in ['ingredients:', '**ingredients**', '#', 'calories:', 'protein:', 'carbs:', 'fat:', 'total calories']):
                    # Sanitize the line
                    clean_line = sanitize_text(line.strip())
                    if clean_line:  # Only add if line is not empty after cleaning
                        elements.append(Paragraph(clean_line, normal_style))
                        elements.append(Spacer(1, 0.05 * inch))
        
        elements.append(Spacer(1, 0.15 * inch))
        
        # Instructions section (with error handling)
        elements.append(Paragraph("Instructions", heading_style))
        instructions = recipe.get('instructions', '')
        if not isinstance(instructions, str):
            instructions = str(instructions)
        
        instructions_lines = instructions.split('\n')
        for line in instructions_lines:
            if line.strip():
                if not line.lower().startswith(('instructions:', 'instruction', '**instruction', '#')):
                    # Sanitize the line
                    clean_line = sanitize_text(line.strip())
                    if clean_line:  # Only add if line is not empty after cleaning
                        elements.append(Paragraph(clean_line, normal_style))
                        elements.append(Spacer(1, 0.05 * inch))
        
        # Add footer
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(f"Generated by Dishpal", footer_style))
        elements.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", footer_style))
        
        # Build the PDF
        doc.build(elements)
        
        # Get the PDF value from the buffer
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
    
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        # Return a simple fallback PDF with error information
        try:
            fallback_buffer = BytesIO()
            fallback_doc = SimpleDocTemplate(fallback_buffer, pagesize=letter)
            
            styles = getSampleStyleSheet()
            elements = []
            
            elements.append(Paragraph("Recipe PDF Generation Error", styles['Title']))
            elements.append(Spacer(1, 0.5 * inch))
            elements.append(Paragraph(f"There was an error generating the PDF for recipe '{recipe.get('title', 'Unknown Recipe')}':", styles['Normal']))
            elements.append(Paragraph(str(e), styles['Normal']))
            elements.append(Spacer(1, 0.5 * inch))
            elements.append(Paragraph("Please try again or contact support.", styles['Normal']))
            
            fallback_doc.build(elements)
            fallback_data = fallback_buffer.getvalue()
            fallback_buffer.close()
            
            return fallback_data
        except:
            # If even the fallback fails, return None
            return None

def parse_ingredient_amount(ingredient_text):
    """Parse ingredient text to extract amount, unit, and ingredient name"""
    # Regular expression to match amount and unit
    pattern = r'^(?:(\d+(?:\.\d+)?)\s*([a-zA-Z]+|\"|\')\s+)?(.+)$'
    match = re.match(pattern, ingredient_text.strip())
    
    if match:
        amount = float(match.group(1)) if match.group(1) else 1
        unit = match.group(2) if match.group(2) else ''
        name = match.group(3).lower()
        return amount, unit, name
    return 1, '', ingredient_text.lower()

def check_recipe_ingredients(recipe, username):
    """Compare recipe ingredients with fridge contents and return available and missing ingredients"""
    fridge_contents = get_fridge_contents(username)
    fridge_dict = {item['ingredient'].lower(): {'quantity': item['quantity'], 'unit': item.get('unit', '')} 
                  for item in fridge_contents}
    
    available = []
    missing = []
    ingredients_list = recipe['ingredients'].split('\n')
    
    for ingredient in ingredients_list:
        if ingredient.strip() and not any(x in ingredient.lower() for x in ['ingredients:', '**ingredients**']):
            amount, unit, name = parse_ingredient_amount(ingredient)
            
            if name in fridge_dict:
                fridge_item = fridge_dict[name]
                if fridge_item['unit'] == unit:
                    if fridge_item['quantity'] >= amount:
                        available.append({
                            'name': name,
                            'amount': amount,
                            'unit': unit,
                            'available': fridge_item['quantity']
                        })
                    else:
                        missing.append({
                            'name': name,
                            'amount': amount,
                            'unit': unit,
                            'available': fridge_item['quantity']
                        })
                else:
                    missing.append({
                        'name': name,
                        'amount': amount,
                        'unit': unit,
                        'available': f"{fridge_item['quantity']} {fridge_item['unit']}"
                    })
            else:
                missing.append({
                    'name': name,
                    'amount': amount,
                    'unit': unit,
                    'available': 0
                })
    
    return available, missing

def make_recipe(recipe, username):
    """Subtract used ingredients from fridge"""
    available, missing = check_recipe_ingredients(recipe, username)
    
    if missing:
        return False, "Missing ingredients"
    
    # Update fridge quantities
    fridge_contents = get_fridge_contents(username)
    for ingredient in available:
        for item in fridge_contents:
            if item['ingredient'].lower() == ingredient['name']:
                new_quantity = item['quantity'] - ingredient['amount']
                if new_quantity > 0:
                    fridge_collection.update_one(
                        {'username': username, 'ingredient': item['ingredient']},
                        {'$set': {'quantity': new_quantity}}
                    )
                else:
                    fridge_collection.delete_one({'username': username, 'ingredient': item['ingredient']})
                break
    
    return True, "Recipe made successfully!"

def recipe_page():
    """Display individual recipe page"""
    if st.session_state.current_recipe:
        recipe = recipes.find_one({'id': st.session_state.current_recipe})
        
        if recipe:
            col1, col2 = st.columns([1, 20])
            with col1:
                st.button("←", on_click=navigate_to_home)
            with col2:
                st.title(recipe['title'])
            
            # Recipe details in two columns - info and image
            details_col, image_col = st.columns([3, 1])
            
            with details_col:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**Diet:** {recipe['diet']}")
                with col2:
                    st.write(f"**Meal Type:** {recipe['meal']}")
                with col3:
                    st.write(f"**Cuisine:** {recipe['cuisine']}")
                with col4:
                    st.write(f"**Calories:** {recipe['calories']}")
            
            # Display image in side column
            with image_col:
                if 'image' in recipe and recipe['image']:
                    try:
                        # Center the image using columns
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            image = Image.open(io.BytesIO(recipe['image']))
                            st.image(image, use_column_width=True)
                    except Exception:
                        if st.button("Generate Image", key=f"gen_img_{recipe['id']}", help="Generate a new image for this recipe"):
                            with st.spinner("Generating image..."):
                                image_bytes = generate_food_image(recipe['title'], recipe['ingredients'])
                                if image_bytes:
                                    recipes.update_one(
                                        {'id': recipe['id']},
                                        {'$set': {'image': image_bytes}}
                                    )
                                    st.rerun()
                else:
                    if st.button("Generate image"):
                        with st.spinner("Generating image..."):
                            image_bytes = generate_food_image(recipe['title'], recipe['ingredients'])
                            if image_bytes:
                                recipes.update_one(
                                    {'id': recipe['id']},
                                    {'$set': {'image': image_bytes}}
                                )
                                st.rerun()
            
            # Check recipe ingredients against fridge
            available, missing = check_recipe_ingredients(recipe, st.session_state.username)
            
            # Display ingredient availability
            st.subheader("Ingredient Availability")
            if available:
                st.write("✅ Available ingredients:")
                for ing in available:
                    st.write(f"- {ing['name'].title()}: {ing['amount']} {ing['unit']} (You have: {ing['available']} {ing['unit']})")
            
            if missing:
                st.write("❌ Missing or insufficient ingredients:")
                for ing in missing:
                    st.write(f"- {ing['name'].title()}: Need {ing['amount']} {ing['unit']} (You have: {ing['available']})")
            
            # Make Recipe button
            if not missing:
                if st.button("🥘 Make Recipe", help="This will subtract the used ingredients from your fridge"):
                    success, message = make_recipe(recipe, st.session_state.username)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.warning("⚠️ Add missing ingredients to your fridge before making this recipe")
            
            # Ingredients and Instructions
            st.subheader("Ingredients")
            st.markdown(recipe['ingredients'])
            
            st.subheader("Instructions")
            st.markdown(recipe['instructions'])
            
            # Download PDF and Delete buttons in columns
            col1, col2 = st.columns([1, 1])
            with col1:
                # Download PDF button
                try:
                    pdf_data = generate_recipe_pdf(recipe)
                    st.download_button(
                        label="📄 Download Recipe as PDF",
                        data=pdf_data,
                        file_name=f"{recipe['title'].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        help="Download a beautifully formatted PDF of this recipe"
                    )
                except Exception as e:
                    st.error(f"Could not generate PDF: {str(e)}")
                    st.info("You can still view the recipe on this page.")
            
            with col2:
                # Delete button
                if st.button("🗑️ Delete Recipe"):
                    st.session_state.confirm_full_delete = True
            
            # Show delete confirmation
            if getattr(st.session_state, 'confirm_full_delete', False):
                st.warning(f"Are you sure you want to delete '{recipe['title']}'?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Delete Recipe"):
                        if delete_saved_recipe(recipe['id']):
                            st.session_state.confirm_full_delete = False
                            navigate_to_home()
                with col2:
                    if st.button("Cancel Delete"):
                        st.session_state.confirm_full_delete = False
                        st.rerun()
        else:
            st.error("Recipe not found")
            st.button("Back to Home", on_click=navigate_to_home)
    else:
        st.error("No recipe selected")
        st.button("Back to Home", on_click=navigate_to_home)

def display_recipe_card(recipe):
    """Display a card for a recipe with title and ingredients only"""
    # Title and metadata with white background
    st.markdown(f"""
    <div style="background-color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #ffbcd1;">
        <h3 style="color: #5b3c24; margin-bottom: 10px; text-align: center;">{recipe['title']}</h3>
        <p class="recipe-metadata" style="text-align: center;"><strong>Diet:</strong> {recipe['diet']} | <strong>Cuisine:</strong> {recipe['cuisine']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display image if available with centering
    if 'image' in recipe and recipe['image']:
        try:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                image = Image.open(io.BytesIO(recipe['image']))
                st.image(image, use_column_width=True)
        except Exception:
            if st.button("Generate Image", key=f"gen_img_{recipe['id']}", help="Generate a new image for this recipe"):
                with st.spinner("Generating image..."):
                    image_bytes = generate_food_image(recipe['title'], recipe['ingredients'])
                    if image_bytes:
                        recipes.update_one(
                            {'id': recipe['id']},
                            {'$set': {'image': image_bytes}}
                        )
                        st.rerun()
    else:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Generate Image", key=f"gen_img_{recipe['id']}", help="Generate a new image for this recipe"):
                with st.spinner("Generating image..."):
                    image_bytes = generate_food_image(recipe['title'], recipe['ingredients'])
                    if image_bytes:
                        recipes.update_one(
                            {'id': recipe['id']},
                            {'$set': {'image': image_bytes}}
                        )
                        st.rerun()
    
    # Ingredients section with styling
    st.markdown("""
    <h4 class="recipe-section-header" style="text-align: center;">Ingredients</h4>
    """, unsafe_allow_html=True)
    
    # Extract ingredients
    ingredients_text = recipe['ingredients']
    ingredients_list = [ing.strip() for ing in ingredients_text.split('\n') if ing.strip()]
    
    # Filter out section headers and empty lines
    filtered_ingredients = []
    for ing in ingredients_list:
        if not ing.lower().startswith(('ingredient', '**ingredient', '## ingredient')):
            filtered_ingredients.append(ing)
    
    # Show 5 ingredients max on the card with styling
    max_ingredients = min(5, len(filtered_ingredients))
    ingredients_html = '<ul style="list-style-type: none; padding-left: 0; margin: 10px 0;">'
    for i in range(max_ingredients):
        ingredients_html += f'<li class="recipe-content" style="text-align: center;">• {filtered_ingredients[i]}</li>'
    ingredients_html += '</ul>'
    st.markdown(ingredients_html, unsafe_allow_html=True)
    
    if len(filtered_ingredients) > max_ingredients:
        st.markdown(f'<p class="recipe-content" style="text-align: center; color: #666;">...and {len(filtered_ingredients) - max_ingredients} more</p>', unsafe_allow_html=True)
    
    # Check recipe ingredients against fridge
    available, missing = check_recipe_ingredients(recipe, st.session_state.username)
    
    # Buttons section - now with 4 columns
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.button("👁️ View Recipe", key=f"view_{recipe['id']}", 
                 on_click=navigate_to_recipe, args=(recipe['id'],))
    
    with col2:
        # Add PDF download button
        try:
            pdf_data = generate_recipe_pdf(recipe)
            st.download_button(
                label="📄 Download PDF",
                data=pdf_data,
                file_name=f"{recipe['title'].replace(' ', '_')}.pdf",
                mime="application/pdf",
                help="Download a beautifully formatted PDF of this recipe",
                key=f"pdf_{recipe['id']}"
            )
        except Exception as e:
            st.error("Could not generate PDF")
    
    with col3:
        # Make Recipe button
        if not missing:
            if st.button("🥘 Make Recipe", key=f"make_{recipe['id']}", help="This will subtract the used ingredients from your fridge"):
                success, message = make_recipe(recipe, st.session_state.username)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.button("🥘 Make Recipe", key=f"make_{recipe['id']}", disabled=True, help="Add missing ingredients to your fridge first")
    
    with col4:
        # Delete button
        if st.button("🗑️ Delete", key=f"delete_{recipe['id']}"):
            st.session_state.confirm_delete_id = recipe['id']
            st.session_state.confirm_delete_title = recipe['title']
    
    # Show missing ingredients warning if any
    if missing:
        st.warning("⚠️ Missing ingredients: " + ", ".join(f"{ing['name']} ({ing['amount']} {ing['unit']})" for ing in missing))
    
    # Show confirmation if this recipe is selected for deletion
    if getattr(st.session_state, 'confirm_delete_id', None) == recipe['id']:
        st.warning(f"Are you sure you want to delete '{st.session_state.confirm_delete_title}'?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Delete", key=f"confirm_delete_{recipe['id']}"):
                if delete_saved_recipe(recipe['id']):
                    st.session_state.confirm_delete_id = None
                    st.rerun()
        with col2:
            if st.button("Cancel", key=f"cancel_delete_{recipe['id']}"):
                st.session_state.confirm_delete_id = None
                st.rerun()
    
    st.markdown("---")

def display_generated_recipe(recipe_info, index):
    """Display a generated recipe with a save button"""
    recipe_text = recipe_info['text']
    
    parsed = parse_recipe(recipe_text)
    
    # Record title for easier reference
    if 'title' not in recipe_info:
        recipe_info['title'] = parsed['title']
    
    # Extract protein from text
    protein_match = re.search(r'Protein:\s*(\d+)\s*g', recipe_text)
    protein = protein_match.group(1) if protein_match else "N/A"
    
    # Create a container for the entire recipe card
    with st.container():
        # Start the white background container for the entire recipe card
        st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
        
        # Title and metadata with white background
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #ffbcd1;">
            <h3 style="color: #5b3c24; margin-bottom: 10px; text-align: center;">{parsed['title']}</h3>
            <p class="recipe-metadata" style="text-align: center;"><strong>Diet:</strong> {recipe_info['diet']} | <strong>Meal:</strong> {recipe_info['meal']} | <strong>Cuisine:</strong> {recipe_info['cuisine']} | <strong>Calories:</strong> {parsed['calories'] if parsed['calories'] else 'N/A'} kcal | <strong>Protein:</strong> {protein}g</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate and display image
        if 'image' not in recipe_info:
            with st.spinner("Generating image..."):
                # Generate image
                image_bytes = generate_food_image(parsed['title'], parsed['ingredients'])
                if image_bytes:
                    recipe_info['image'] = image_bytes
        
        # Display image if available with centering
        if 'image' in recipe_info and recipe_info['image']:
            try:
                # Center the image using columns
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    image = Image.open(io.BytesIO(recipe_info['image']))
                    st.image(image, use_column_width=True)
            except Exception:
                # If image fails to load, don't show anything
                pass
        
        # Tabs for ingredients and instructions - full width
        ing_tab, inst_tab = st.tabs(["📝 Ingredients", "👩‍🍳 Instructions"])
        
        with ing_tab:
            # Clean and format ingredients
            ingredients = parsed['ingredients'].strip()
            if ingredients:
                # Format ingredients as a bulleted list
                ingredients_list = [ing.strip() for ing in ingredients.split('\n') if ing.strip()]
                ingredients_html = '<ul style="list-style-type: none; padding-left: 0;">'
                for ing in ingredients_list:
                    # Skip lines with nutritional info or section headers
                    if not ing.lower().startswith(('ingredients:', '**ingredients**')):
                        # Clean up asterisks from ingredient text
                        clean_ing = re.sub(r'^\s*\*\s*', '', ing)  # Remove leading asterisks
                        clean_ing = re.sub(r'\*+', '', clean_ing)  # Remove any remaining asterisks
                        ingredients_html += f'<li class="recipe-content" style="margin-bottom: 8px;">• {clean_ing}</li>'
                ingredients_html += '</ul>'
                st.markdown(ingredients_html, unsafe_allow_html=True)
            else:
                st.info("No ingredients listed")
        
        with inst_tab:
            # Clean and format instructions
            instructions = parsed['instructions'].strip()
            if instructions:
                # Format instructions as numbered steps
                instructions_list = [inst.strip() for inst in instructions.split('\n') if inst.strip()]
                instructions_html = '<ol style="padding-left: 20px;">'
                for inst in instructions_list:
                    if not inst.lower().startswith(('instructions:', 'instruction', '**instruction', '#')):
                        # Remove any existing numbers at the start
                        inst = re.sub(r'^\d+\.\s*', '', inst)
                        instructions_html += f'<li class="recipe-content" style="margin-bottom: 12px;">{inst}</li>'
                instructions_html += '</ol>'
                st.markdown(instructions_html, unsafe_allow_html=True)
            else:
                st.info("No instructions listed")
        
        # Save and Download buttons with spinners
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if not recipe_info.get('saved', False):
                st.button(f"💾 Save Recipe", key=f"save_recipe_{index}", 
                         on_click=save_individual_recipe, args=(index,))
            else:
                st.success("✅ Saved!")
        
        with col2:
            # Create a temporary recipe dict with the required format for PDF generation
            temp_recipe = {
                'title': parsed['title'],
                'ingredients': parsed['ingredients'],
                'instructions': parsed['instructions'],
                'diet': recipe_info['diet'],
                'meal': recipe_info['meal'],
                'cuisine': recipe_info['cuisine'],
                'calories': parsed['calories'] if parsed['calories'] else 'N/A',
                'full_text': recipe_info['text']  # Add the full text for macro extraction
            }
            
            try:
                pdf_data = generate_recipe_pdf(temp_recipe)
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_data,
                    file_name=f"{parsed['title'].replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    help="Download a beautifully formatted PDF of this recipe",
                    key=f"download_pdf_{index}"
                )
            except Exception as e:
                st.error(f"Could not generate PDF")
        
        # Close the recipe card div at the end of all content
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")

def home_page():
    """Display home page with recipe generator and saved recipes"""
    # Display logo using Streamlit's image function
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.markdown("""
            <div style="display: flex; justify-content: center; width: 100%; margin: 0 auto;">
                <div style="width: 155px;">
                    <img src="data:image/png;base64,{}" width="155">
                </div>
            </div>
        """.format(base64.b64encode(open(logo_path, "rb").read()).decode()), unsafe_allow_html=True)

    # Check if a recipe was just saved and display a success message
    if st.session_state.get('recipe_saved', False):
        st.success("✅ Recipe saved successfully!")
        st.session_state.recipe_saved = False  # Reset the flag
    
    # Check if a recipe was just deleted and display a success message
    if st.session_state.get('recipe_deleted', False):
        st.success("✅ Recipe deleted successfully!")
        st.session_state.recipe_deleted = False  # Reset the flag
    
    if not st.session_state.logged_in:
        # Replace tabs with radio buttons to avoid setIn error
        auth_option = st.radio("", ["Login", "Sign Up"], horizontal=True)
        if auth_option == "Login":
            login()
        else:
            signup()
    else:
        st.write(f"Welcome, {st.session_state.username}! 👋")
        if st.button("Logout", key="logout"):
            logout()

        # Checkbox to toggle the review section
        show_reviews = st.checkbox("Show Customer Reviews", value=st.session_state.get('show_reviews', False))
        st.session_state.show_reviews = show_reviews  # Update session state

        # Show the review section if toggled
        if st.session_state.show_reviews:
            # Review Form
            st.subheader("Customer Review")
            
            # Star Rating
            star_rating = st.select_slider("Rate us (1 to 5 stars)", options=[1, 2, 3, 4, 5], value=3)
            
            # Optional Text Review
            review_text = st.text_area("Share your feedback about our webpage (optional):")
            
            if st.button("Submit Review"):
                if review_text or star_rating:  # Allow submission if either field is filled
                    if save_review(st.session_state.username, star_rating, review_text):
                        st.success(f"Thank you for your feedback! You rated us {star_rating} star(s).")
                        # Clear the inputs after submission
                        star_rating = 3  # Reset to default
                        review_text = ""  # Clear text area
                    else:
                        st.error("Failed to submit your review. Please try again.")
                else:
                    st.warning("Please provide at least a star rating or a written review.")

            if st.button("Show Previous Reviews"):
                # Calculate and display average rating
                average_rating = calculate_average_rating()
                if average_rating is not None:
                    st.markdown(f"**Average Rating:** {'⭐' * int(average_rating)} ({average_rating:.1f}/5)")
                else:
                    st.info("No reviews yet to calculate an average rating.")
                # Display previous reviews
                display_reviews()

            # Calculate and display average rating
            average_rating = calculate_average_rating()
            if average_rating is not None:
                st.markdown(f"**Average Rating:** {'⭐' * int(average_rating)} ({average_rating:.1f}/5)")

            
        # Replace tabs with radio buttons to avoid setIn error
        page_option = st.radio("", ["Generate New Recipe", "My Recipes", "My Fridge"], horizontal=True)
        
        if page_option == "Generate New Recipe":
            # Recipe generator inputs
            diet = st.selectbox("Dietary Preference", ["Veg", "Non-Veg", "Vegan"])
            meal = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snacks"])
            cuisine = st.selectbox("Cuisine", ["Indian", "Chinese", "Italian", "Mexican", "American", "Other", "Any"])
            calories = st.slider("Max Calories", 100, 1000, 500)
            allergies = st.text_input("Allergies (comma-separated)")
            
            # Get fridge ingredients for potential import
            fridge_contents = get_fridge_contents(st.session_state.username)
            fridge_ingredients = [f"{item['ingredient']} ({item['quantity']} {item.get('unit', '')})" for item in fridge_contents] if fridge_contents else []
            
            # Initialize ingredients in session state if not present
            if 'ingredients_input' not in st.session_state:
                st.session_state.ingredients_input = ""

            # Initialize selection state if not present
            if 'fridge_selections' not in st.session_state:
                st.session_state.fridge_selections = {}
                
            # Function to import fridge ingredients
            def import_from_fridge():
                if fridge_contents:
                    # Create a comma-separated list with selected ingredients
                    selected_ingredients = [item['ingredient'] for item in fridge_contents 
                                          if st.session_state.fridge_selections.get(item['ingredient'], False)]
                    
                    if selected_ingredients:
                        imported_ingredients = ", ".join(selected_ingredients)
                        
                        # Update the session state, appending to existing ingredients if any
                        current = st.session_state.ingredients_input
                        if current and not current.endswith(", "):
                            st.session_state.ingredients_input = current + ", " + imported_ingredients
                        else:
                            st.session_state.ingredients_input = current + imported_ingredients
            
            # Ingredients input with fridge import option
            ingredients = st.text_area("Ingredients You Have (comma-separated)", value=st.session_state.ingredients_input)
            if st.button("Select from Fridge", key="select_from_fridge_btn", disabled=not fridge_ingredients):
                st.session_state.show_fridge_select = True
                    
            # Show ingredient selection interface if button was clicked
            if fridge_ingredients and getattr(st.session_state, 'show_fridge_select', False):
                st.write("#### Select ingredients to import:")
                
                # Add search box for filtering ingredients
                search_term = st.text_input("Search ingredients", key="fridge_search", placeholder="Type to filter...")
                
                # Filter ingredients based on search term
                filtered_contents = fridge_contents
                if search_term:
                    search_lower = search_term.lower()
                    filtered_contents = [item for item in fridge_contents 
                                        if search_lower in item['ingredient'].lower()]
                    
                    if not filtered_contents:
                        st.info(f"No ingredients match '{search_term}'")
                
                # Create columns for a more compact layout
                cols = st.columns(3)
                
                # Display checkboxes for each ingredient
                for i, item in enumerate(filtered_contents):
                    ing_name = item['ingredient']
                    col_idx = i % 3
                    
                    # Initialize this ingredient's selection state if not present
                    if ing_name not in st.session_state.fridge_selections:
                        st.session_state.fridge_selections[ing_name] = False
                        
                    # Display checkbox in the appropriate column
                    with cols[col_idx]:
                        unit_display = f" ({item['quantity']} {item.get('unit', '')})"
                        st.session_state.fridge_selections[ing_name] = st.checkbox(
                            f"{ing_name}{unit_display}", 
                            value=st.session_state.fridge_selections.get(ing_name, False),
                            key=f"select_{ing_name}"
                        )
                
                # Import button
                if st.button("Import Selected Ingredients"):
                    import_from_fridge()
                    st.session_state.show_fridge_select = False
                    st.rerun()
                    
                # Cancel button    
                if st.button("Cancel"):
                    st.session_state.show_fridge_select = False
                    st.rerun()
                
            elif not fridge_ingredients:
                st.info("Your fridge is empty. Add ingredients in the 'My Fridge' tab.")
                
            options = st.slider("Number of Recipes", 1, 5, 3)

            col1, col2 = st.columns(2)
            with col1:
                surprise_button = st.button("Surprise me!")
            with col2:
                generate_button = st.button("Generate Recipe")
            
            # Reset recipes when generating new ones
            if surprise_button or generate_button:
                st.session_state.generated_recipes = []

            if surprise_button:
                # Create a placeholder for the loading animation
                loading_placeholder = st.empty()
                
                # Display a CSS-animated emoji spinner
                loading_placeholder.markdown("""
                <style>
                    @keyframes fade {
                        0% { opacity: 1; }
                        50% { opacity: 0.3; }
                        100% { opacity: 1; }
                    }
                    .emoji-container {
                        text-align: center;
                        padding: 30px;
                        animation: fade 1.5s infinite;
                        background-color: #ffffff;
                        border-radius: 12px;
                        border: 1px solid #ffbcd1;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                        margin: 20px 0;
                    }
                    .emoji-row {
                        font-size: 2.5rem;
                        margin-bottom: 15px;
                    }
                    .loading-text {
                        color: #5b3c24;
                        font-size: 18px;
                        font-weight: 500;
                    }
                    .loading-subtext {
                        color: #888;
                        font-size: 14px;
                        margin-top: 10px;
                    }
                </style>
                <div class="emoji-container">
                    <div class="emoji-row">🍳 🥘 👩‍🍳 🍲 🌮 🥗</div>
                    <p class="loading-text">Cooking up something special for you...</p>
                    <p class="loading-subtext">Please wait! Good food takes time! 🕒</p>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    prompt = f"""
                    Generate 1 random recipe using this exact structure:
                    1. Recipe name in bold with a number (use "1. **Recipe Name**")
                    2. Include nutritional information right after the title:
                       - Total Calories: X kcal
                       - Protein: X g
                       - Carbs: X g
                       - Fat: X g
                    3. "Ingredients:" section with:
                       - Clear measurements
                       - One ingredient per line
                       - Use bullet points (*)
                    4. "Instructions:" section with:
                       - Numbered steps (1., 2., 3., etc.)
                       - Each step should be detailed and clear
                       - Include cooking times and temperatures
                       - Explain techniques and methods thoroughly
                       - Break complex steps into sub-steps
                    
                    IMPORTANT: 
                    - Recipe MUST have both "Ingredients:" and "Instructions:" section headers
                    - Recipe number MUST be "1." at the start
                    - MUST include nutritional information with all macros
                    """
                    
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(prompt)

                    # Clear the loading animation after generation is complete
                    loading_placeholder.empty()

                    recipe_text = response.text if response.text else "No recipe found. Try different inputs."
                    
                    # Store the generated recipes in session state
                    st.session_state.generated_recipes = [
                        {
                            'text': recipe_text,
                            'diet': diet,
                            'meal': meal,
                            'cuisine': cuisine,
                            'calories': calories,
                            'saved': False
                        }
                    ]

                except Exception as e:
                    # Clear the loading animation on error too
                    loading_placeholder.empty()
                    st.error(f"⚠️ Error generating recipe: {str(e)}")

                # Display generated recipes if available
                if st.session_state.generated_recipes:
                    st.write("---")
                    st.subheader("🍳 Generated Recipes")
                    
                    for i, recipe_info in enumerate(st.session_state.generated_recipes):
                        display_generated_recipe(recipe_info, i)

            if generate_button:
                # Create a placeholder for the loading animation
                loading_placeholder = st.empty()
                
                # Display loading animation
                loading_placeholder.markdown("""
                <style>
                    @keyframes bounce {
                        0%, 100% { transform: translateY(0); }
                        50% { transform: translateY(-10px); }
                    }
                    .emoji-container {
                        text-align: center;
                        padding: 30px;
                        background-color: #ffffff;
                        border-radius: 12px;
                        border: 1px solid #ffbcd1;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                        margin: 20px 0;
                    }
                    .emoji-row {
                        font-size: 2.5rem;
                        margin-bottom: 15px;
                    }
                    .emoji-row span {
                        display: inline-block;
                        animation: bounce 1s infinite;
                        animation-delay: calc(var(--delay) * 0.1s);
                    }
                    .loading-text {
                        color: #5b3c24;
                        font-size: 18px;
                        font-weight: 500;
                    }
                    .loading-subtext {
                        color: #888;
                        font-size: 14px;
                        margin-top: 10px;
                    }
                </style>
                <div class="emoji-container">
                    <div class="emoji-row">
                        <span style="--delay: 1">👨‍🍳</span>
                        <span style="--delay: 2">🔪</span>
                        <span style="--delay: 3">🍽️</span>
                        <span style="--delay: 4">🥄</span>
                        <span style="--delay: 5">🧄</span>
                        <span style="--delay: 6">🧅</span>
                    </div>
                    <p class="loading-text">Preparing your custom recipes...</p>
                    <p class="loading-subtext">Please wait while our virtual chef works their magic!</p>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    # Clean and format ingredients
                    ingredients_prompt = ""
                    if ingredients and ingredients.strip():
                        # Remove any trailing commas and clean the input
                        cleaned_ingredients = ingredients.strip().rstrip(',').strip()
                        if cleaned_ingredients:
                            ingredients_prompt = f"Use these ingredients: {cleaned_ingredients}."
                    
                    prompt = f"""
                    Generate {options} {diet} {meal} recipe{'s' if options > 1 else ''} from {cuisine} cuisine within {calories} calories.
                    {ingredients_prompt}
                    {'Exclude ingredients causing these allergies: ' + allergies + '.' if allergies.strip() else ''}
                    Assume kitchen staples like spices, water, herbs are available by default.
                    
                    For each recipe, use this exact structure:
                    1. Recipe name in bold with a number from 1 to {options} (e.g., if generating 3 recipes, use "1. **Recipe Name**", "2. **Recipe Name**", "3. **Recipe Name**")
                    2. Include nutritional information right after the title:
                       - Total Calories: X kcal
                       - Protein: X g
                       - Carbs: X g
                       - Fat: X g
                    3. "Ingredients:" section with:
                       - Clear measurements
                       - One ingredient per line
                       - Use bullet points (*)
                    4. "Instructions:" section with:
                       - Numbered steps (1., 2., 3., etc.)
                       - Each step should be detailed and clear
                       - Include cooking times and temperatures
                       - Explain techniques and methods thoroughly
                       - Break complex steps into sub-steps
                    
                    Each recipe must have all sections clearly marked.
                    Separate recipes with blank lines.
                    Do not include any text between recipes.
                    Create exactly {options} distinct recipes, each with complete instructions.
                    
                    IMPORTANT: 
                    - Each recipe MUST have both "Ingredients:" and "Instructions:" section headers
                    - Recipe numbers MUST start at 1 and go up to {options} in sequence
                    - MUST include nutritional information with all macros
                    """
                    
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(prompt)

                    # Clear the loading animation after generation is complete
                    loading_placeholder.empty()

                    recipe_text = response.text if response.text else "No recipe found. Try different inputs."
                    
                    # Split into individual recipes
                    recipe_texts = split_multiple_recipes(recipe_text, num_recipes=options)
                    
                    # Store the generated recipes in session state
                    st.session_state.generated_recipes = [
                        {
                            'text': text,
                            'diet': diet,
                            'meal': meal,
                            'cuisine': cuisine,
                            'calories': calories,
                            'saved': False
                        }
                        for text in recipe_texts
                    ]
                
                except Exception as e:
                    # Clear the loading animation on error too
                    loading_placeholder.empty()
                    st.error(f"⚠️ Error generating recipe: {str(e)}")
                
            # Display generated recipes if available
            if st.session_state.generated_recipes:
                st.write("---")
                st.subheader("🍳 Generated Recipes")
                
                for i, recipe_info in enumerate(st.session_state.generated_recipes):
                    display_generated_recipe(recipe_info, i)

        elif page_option == "My Recipes":
            # st.subheader("My Saved Recipes")  # Removed redundant subheader
            
            # Get user's saved recipes
            try:
                user_recipes = list(recipes.find({'user': st.session_state.username}).sort('created_at', -1))
                
                if not user_recipes:
                    st.info("You haven't saved any recipes yet. Generate and save some recipes to see them here!")
                else:
                    st.write(f"Found {len(user_recipes)} recipes")
                    
                    # Display each recipe in a card
                    for recipe in user_recipes:
                        display_recipe_card(recipe)
            except Exception as e:
                st.error(f"⚠️ Error fetching recipes: {str(e)}")
        
        elif page_option == "My Fridge":
            fridge_page()

# Fridge Management Functions
def get_fridge_contents(user):
    """Retrieve all ingredients in the user's fridge."""
    return list(fridge_collection.find({'user': user}))

def delete_ingredient(user, ingredient):
    """Remove an ingredient from the user's fridge."""
    fridge_collection.delete_one({'user': user, 'ingredient': ingredient})

def recognize_ingredients_from_image(image_array):
    """Use MobileNetV2 to recognize ingredients in an image"""
    try:
        # Load model and labels if not already loaded
        model = load_model()
        labels = load_labels()
        
        # Preprocess image with optimized transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert numpy array to tensor
        image_tensor = transform(image_array).unsqueeze(0)
        
        # Get predictions with torch.no_grad() for faster inference
        with torch.no_grad():
            outputs = model(image_tensor)
            _, indices = torch.topk(outputs, 100)  # Increased from 50 to 100 predictions
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probs = probabilities[0]
        
        # Filter for food-related items with specific ingredients
        food_related = []
        specific_ingredients = [
            # Fruits
            'apple', 'banana', 'orange', 'lemon', 'lime', 'grape', 'strawberry', 'blueberry', 'mango', 'peach',
            'pear', 'plum', 'cherry', 'watermelon', 'pineapple', 'kiwi', 'pomegranate', 'fig', 'date', 'coconut',
            'raspberry', 'blackberry', 'cranberry', 'apricot', 'nectarine', 'guava', 'papaya', 'dragon fruit',
            'lychee', 'passion fruit', 'mulberry', 'gooseberry', 'quince', 'persimmon', 'star fruit',
            # Vegetables
            'carrot', 'broccoli', 'cauliflower', 'cabbage', 'lettuce', 'spinach', 'tomato', 'potato', 'onion', 'garlic',
            'cucumber', 'bell pepper', 'zucchini', 'eggplant', 'mushroom', 'asparagus', 'green bean', 'pea', 'corn',
            'celery', 'radish', 'beet', 'sweet potato', 'yam', 'ginger', 'leek', 'shallot', 'scallion', 'chive',
            'kale', 'arugula', 'brussels sprout', 'artichoke', 'fennel', 'turnip', 'rutabaga', 'parsnip', 'jicama',
            'bok choy', 'chinese cabbage', 'daikon', 'okra', 'squash', 'pumpkin', 'butternut', 'acorn squash',
            # Meats
            'chicken', 'beef', 'pork', 'lamb', 'fish', 'shrimp', 'crab', 'lobster', 'turkey', 'duck', 'goose',
            'salmon', 'tuna', 'cod', 'tilapia', 'sardine', 'anchovy', 'squid', 'octopus', 'clam', 'mussel',
            'bacon', 'ham', 'sausage', 'hot dog', 'brisket', 'ribs', 'steak', 'ground beef', 'chicken breast',
            'chicken thigh', 'chicken wing', 'drumstick', 'pork chop', 'pork tenderloin', 'lamb chop',
            # Dairy
            'milk', 'cheese', 'yogurt', 'butter', 'cream', 'cottage cheese', 'sour cream', 'whipped cream',
            'cream cheese', 'mozzarella', 'cheddar', 'parmesan', 'ricotta', 'feta', 'greek yogurt',
            'provolone', 'swiss', 'brie', 'camembert', 'blue cheese', 'goat cheese', 'halloumi', 'queso',
            'mascarpone', 'heavy cream', 'half and half', 'evaporated milk', 'condensed milk',
            # Grains
            'rice', 'pasta', 'bread', 'flour', 'oats', 'quinoa', 'barley', 'millet', 'buckwheat', 'rye',
            'cornmeal', 'couscous', 'bulgur', 'farro', 'spelt', 'wheat', 'semolina', 'noodle',
            'brown rice', 'white rice', 'wild rice', 'basmati', 'jasmine', 'arborio', 'risotto',
            'spaghetti', 'penne', 'fettuccine', 'linguine', 'macaroni', 'lasagna', 'ravioli',
            # Nuts and Seeds
            'almond', 'walnut', 'peanut', 'cashew', 'sunflower seed', 'chia seed', 'pistachio', 'pecan',
            'macadamia', 'hazelnut', 'pine nut', 'sesame seed', 'pumpkin seed', 'flax seed', 'hemp seed',
            'brazil nut', 'chestnut', 'coconut', 'peanut butter', 'almond butter', 'tahini',
            # Herbs and Spices
            'basil', 'oregano', 'thyme', 'rosemary', 'cinnamon', 'pepper', 'salt', 'parsley', 'mint',
            'sage', 'dill', 'coriander', 'cumin', 'turmeric', 'ginger', 'garlic', 'onion powder',
            'paprika', 'nutmeg', 'clove', 'cardamom', 'bay leaf', 'lemongrass', 'chive', 'tarragon',
            'chili', 'cayenne', 'red pepper', 'black pepper', 'white pepper', 'allspice', 'star anise',
            # Other Common Ingredients
            'egg', 'mushroom', 'bell pepper', 'cucumber', 'avocado', 'lemon', 'lime', 'olive oil',
            'soy sauce', 'vinegar', 'honey', 'sugar', 'chocolate', 'coffee', 'tea', 'wine', 'beer',
            'tofu', 'tempeh', 'seitan', 'miso', 'kimchi', 'sauerkraut', 'pickle', 'jam', 'jelly',
            'mayonnaise', 'mustard', 'ketchup', 'salsa', 'guacamole', 'hummus', 'tahini', 'pesto',
            'balsamic', 'worcestershire', 'fish sauce', 'oyster sauce', 'hoisin', 'teriyaki',
            # Cooking Methods and Categories
            'soup', 'salad', 'sauce', 'dressing', 'dip', 'spread', 'smoothie', 'juice', 'beverage',
            'dessert', 'snack', 'appetizer', 'main course', 'side dish', 'breakfast', 'lunch', 'dinner',
            'stew', 'curry', 'stir fry', 'roasted', 'grilled', 'baked', 'fried', 'steamed', 'boiled',
            'sautéed', 'braised', 'poached', 'smoked', 'cured', 'pickled', 'fermented', 'marinated'
        ]
        
        # Lower confidence threshold for better detection
        min_confidence = 0.05  # Reduced from 0.10 to 0.05 (5% minimum confidence)
        
        for idx, prob in zip(indices[0], probs[indices[0]]):
            label = labels[idx].lower()
            # Check if the label contains any of our specific ingredients
            if any(ingredient in label for ingredient in specific_ingredients) and float(prob) >= min_confidence:
                food_related.append({
                    'label': label,
                    'score': float(prob)
                })
        
        # Sort by confidence and return top 5 food items
        food_related.sort(key=lambda x: x['score'], reverse=True)
        
        # If we found specific ingredients, return them
        if food_related:
            return food_related[:5]
        
        # If no specific ingredients found, try with broader categories
        broad_categories = [
            'fruit', 'vegetable', 'meat', 'fish', 'dairy', 'grain', 'nut', 'seed',
            'herb', 'spice', 'rice', 'pasta', 'bread', 'cake', 'soup', 'salad',
            'sauce', 'egg', 'cheese', 'milk', 'yogurt', 'butter', 'oil', 'flour',
            'food', 'dish', 'meal', 'cooking', 'baking', 'eating', 'dining',
            'produce', 'grocery', 'market', 'fresh', 'raw', 'cooked', 'prepared',
            'ingredient', 'kitchen', 'culinary', 'gastronomy', 'cuisine', 'recipe',
            'restaurant', 'cafe', 'bistro', 'diner', 'buffet', 'catering', 'delicacy'
        ]
        
        # Reset food_related list
        food_related = []
        
        for idx, prob in zip(indices[0], probs[indices[0]]):
            label = labels[idx].lower()
            if any(category in label for category in broad_categories) and float(prob) >= min_confidence:
                food_related.append({
                    'label': label,
                    'score': float(prob)
                })
        
        # Sort by confidence and return top 5 food items
        food_related.sort(key=lambda x: x['score'], reverse=True)
        return food_related[:5]

    except Exception as e:
        st.error(f"Error in ingredient recognition: {str(e)}")
        return []

def camera_ingredient_input():
    """Capture and process camera input for ingredient recognition"""
    st.write("### 📸 Add Ingredients via Camera")
    
    # Add camera input
    camera_image = st.camera_input("Take a picture of your ingredients")
    
    if camera_image is not None:
        try:
            # Convert the image to numpy array
            image_array = np.array(Image.open(camera_image))
            
            with st.spinner("Analyzing ingredients..."):
                # Recognize ingredients
                predictions = recognize_ingredients_from_image(image_array)
                
                if predictions:
                    st.write("#### Detected Ingredients:")
                    
                    # Create a form for confirming ingredients
                    with st.form("ingredient_confirmation"):
                        confirmed_ingredients = []
                        
                        # Display detected ingredients in a grid
                        for i in range(0, len(predictions), 2):
                            col1, col2 = st.columns(2)
                            
                            # First ingredient in the row
                            with col1:
                                if i < len(predictions):
                                    pred = predictions[i]
                                    st.markdown(f"""
                                    <div style="font-family: 'Press Start 2P', cursive; color: #5b3c24; font-size: 1em; margin-bottom: 10px;">
                                        {pred['label'].title()}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    confidence_color = "green" if pred['score'] > 0.8 else "orange" if pred['score'] > 0.6 else "red"
                                    st.markdown(f"<p style='color: {confidence_color}; font-weight: bold;'>Confidence: {pred['score']:.1%}</p>", unsafe_allow_html=True)
                                    
                                    col_a, col_b, col_c = st.columns([2, 2, 1])
                                    with col_a:
                                        quantity = st.number_input(
                                            "Amount",
                                            min_value=0.1,
                                            value=1.0,
                                            step=0.1,
                                            key=f"quantity_{i}"
                                        )
                                    with col_b:
                                        unit = st.selectbox(
                                            "Unit",
                                            ["pieces", "g", "kg", "ml", "l", "tbsp", "tsp", "cup", "oz", "lb"],
                                            key=f"unit_{i}"
                                        )
                                    with col_c:
                                        confirm = st.checkbox("Add", key=f"confirm_{i}")
                                    
                                    if confirm:
                                        confirmed_ingredients.append({
                                            'name': pred['label'],
                                            'quantity': quantity,
                                            'unit': unit
                                        })
                            
                            # Second ingredient in the row
                            with col2:
                                if i + 1 < len(predictions):
                                    pred = predictions[i + 1]
                                    st.markdown(f"""
                                    <div style="font-family: 'Press Start 2P', cursive; color: #5b3c24; font-size: 1em; margin-bottom: 10px;">
                                        {pred['label'].title()}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    confidence_color = "green" if pred['score'] > 0.8 else "orange" if pred['score'] > 0.6 else "red"
                                    st.markdown(f"<p style='color: {confidence_color}; font-weight: bold;'>Confidence: {pred['score']:.1%}</p>", unsafe_allow_html=True)
                                    
                                    col_a, col_b, col_c = st.columns([2, 2, 1])
                                    with col_a:
                                        quantity = st.number_input(
                                            "Amount",
                                            min_value=0.1,
                                            value=1.0,
                                            step=0.1,
                                            key=f"quantity_{i+1}"
                                        )
                                    with col_b:
                                        unit = st.selectbox(
                                            "Unit",
                                            ["pieces", "g", "kg", "ml", "l", "tbsp", "tsp", "cup", "oz", "lb"],
                                            key=f"unit_{i+1}"
                                        )
                                    with col_c:
                                        confirm = st.checkbox("Add", key=f"confirm_{i+1}")
                                    
                                    if confirm:
                                        confirmed_ingredients.append({
                                            'name': pred['label'],
                                            'quantity': quantity,
                                            'unit': unit
                                        })
                        
                        st.markdown("---")
                        
                        # Add a text input for manual ingredient entry
                        st.write("#### Add Additional Ingredient")
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        with col1:
                            new_ing = st.text_input("Ingredient name", key="new_ing_name")
                        with col2:
                            new_qty = st.number_input(
                                "Amount",
                                min_value=0.1,
                                value=1.0,
                                step=0.1,
                                key="new_ing_qty"
                            )
                        with col3:
                            new_unit = st.selectbox(
                                "Unit",
                                ["pieces", "g", "kg", "ml", "l", "tbsp", "tsp", "cup", "oz", "lb"],
                                key="new_ing_unit"
                            )
                        with col4:
                            new_confirm = st.checkbox("Add", key="new_ing_confirm")
                        
                        if new_confirm and new_ing:
                            confirmed_ingredients.append({
                                'name': new_ing.lower(),
                                'quantity': new_qty,
                                'unit': new_unit
                            })
                        
                        # Submit button with summary
                        if confirmed_ingredients:
                            st.markdown("#### Selected Ingredients:")
                            for ing in confirmed_ingredients:
                                st.markdown(f"""
                                <div style="font-family: 'Press Start 2P', cursive; color: #5b3c24; font-size: 0.9em;">
                                    • {ing['name'].title()}: {ing['quantity']} {ing['unit']}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        if st.form_submit_button("Add Selected Ingredients to Fridge"):
                            if confirmed_ingredients:
                                for ingredient in confirmed_ingredients:
                                    add_ingredient(
                                        st.session_state.username,
                                        ingredient['name'],
                                        ingredient['quantity'],
                                        ingredient['unit']
                                    )
                                st.success(f"✅ Added {len(confirmed_ingredients)} ingredients to your fridge!")
                                st.rerun()
                            else:
                                st.warning("Please select at least one ingredient to add.")
                else:
                    st.warning("No ingredients detected in the image. Try taking another picture with better lighting and focus.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("You can still add ingredients manually below.")

def fridge_page():
    """Display fridge management page"""
    # Add tabs for different input methods
    input_method = st.radio("Add Ingredients By:", ["Manual Input", "Camera", "Upload Image"], horizontal=True)
    
    if input_method == "Camera":
        camera_ingredient_input()
        st.markdown("---")
    elif input_method == "Upload Image":
        st.write("### 📸 Add Ingredients via Image Upload")
        
        # Add file uploader
        uploaded_file = st.file_uploader("Upload an image of your ingredients", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            try:
                # Convert the uploaded file to numpy array
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                with st.spinner("Analyzing ingredients..."):
                    # Recognize ingredients
                    predictions = recognize_ingredients_from_image(image_array)
                    
                    if predictions:
                        st.write("#### Detected Ingredients:")
                        
                        # Create a form for confirming ingredients
                        with st.form("uploaded_ingredient_confirmation"):
                            confirmed_ingredients = []
                            
                            # Display detected ingredients in a grid
                            for i in range(0, len(predictions), 2):
                                col1, col2 = st.columns(2)
                                
                                # First ingredient in the row
                                with col1:
                                    if i < len(predictions):
                                        pred = predictions[i]
                                        st.markdown(f"""
                                        <div style="font-family: 'Press Start 2P', cursive; color: #5b3c24; font-size: 1em; margin-bottom: 10px;">
                                            {pred['label'].title()}
                                        </div>
                                        """, unsafe_allow_html=True)
                                        confidence_color = "green" if pred['score'] > 0.8 else "orange" if pred['score'] > 0.6 else "red"
                                        st.markdown(f"<p style='color: {confidence_color}; font-weight: bold;'>Confidence: {pred['score']:.1%}</p>", unsafe_allow_html=True)
                                        
                                        col_a, col_b, col_c = st.columns([2, 2, 1])
                                        with col_a:
                                            quantity = st.number_input(
                                                "Amount",
                                                min_value=0.1,
                                                value=1.0,
                                                step=0.1,
                                                key=f"upload_quantity_{i}"
                                            )
                                        with col_b:
                                            unit = st.selectbox(
                                                "Unit",
                                                ["pieces", "g", "kg", "ml", "l", "tbsp", "tsp", "cup", "oz", "lb"],
                                                key=f"upload_unit_{i}"
                                            )
                                        with col_c:
                                            confirm = st.checkbox("Add", key=f"upload_confirm_{i}")
                                        
                                        if confirm:
                                            confirmed_ingredients.append({
                                                'name': pred['label'],
                                                'quantity': quantity,
                                                'unit': unit
                                            })
                            
                            # Second ingredient in the row
                            with col2:
                                if i + 1 < len(predictions):
                                    pred = predictions[i + 1]
                                    st.markdown(f"""
                                    <div style="font-family: 'Press Start 2P', cursive; color: #5b3c24; font-size: 1em; margin-bottom: 10px;">
                                        {pred['label'].title()}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    confidence_color = "green" if pred['score'] > 0.8 else "orange" if pred['score'] > 0.6 else "red"
                                    st.markdown(f"<p style='color: {confidence_color}; font-weight: bold;'>Confidence: {pred['score']:.1%}</p>", unsafe_allow_html=True)
                                    
                                    col_a, col_b, col_c = st.columns([2, 2, 1])
                                    with col_a:
                                        quantity = st.number_input(
                                            "Amount",
                                            min_value=0.1,
                                            value=1.0,
                                            step=0.1,
                                            key=f"upload_quantity_{i+1}"
                                        )
                                    with col_b:
                                        unit = st.selectbox(
                                            "Unit",
                                            ["pieces", "g", "kg", "ml", "l", "tbsp", "tsp", "cup", "oz", "lb"],
                                            key=f"upload_unit_{i+1}"
                                        )
                                    with col_c:
                                        confirm = st.checkbox("Add", key=f"upload_confirm_{i+1}")
                                        
                                        if confirm:
                                            confirmed_ingredients.append({
                                                'name': pred['label'],
                                                'quantity': quantity,
                                                'unit': unit
                                            })
                            
                            st.markdown("---")
                            
                            # Add a text input for manual ingredient entry
                            st.write("#### Add Additional Ingredient")
                            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                            with col1:
                                new_ing = st.text_input("Ingredient name", key="upload_new_ing_name")
                            with col2:
                                new_qty = st.number_input(
                                    "Amount",
                                    min_value=0.1,
                                    value=1.0,
                                    step=0.1,
                                    key="upload_new_ing_qty"
                                )
                            with col3:
                                new_unit = st.selectbox(
                                    "Unit",
                                    ["pieces", "g", "kg", "ml", "l", "tbsp", "tsp", "cup", "oz", "lb"],
                                    key="upload_new_ing_unit"
                                )
                            with col4:
                                new_confirm = st.checkbox("Add", key="upload_new_ing_confirm")
                            
                            if new_confirm and new_ing:
                                confirmed_ingredients.append({
                                    'name': new_ing.lower(),
                                    'quantity': new_qty,
                                    'unit': new_unit
                                })
                            
                            # Submit button with summary
                            if confirmed_ingredients:
                                st.markdown("#### Selected Ingredients:")
                                for ing in confirmed_ingredients:
                                    st.markdown(f"""
                                    <div style="font-family: 'Press Start 2P', cursive; color: #5b3c24; font-size: 0.9em;">
                                        • {ing['name'].title()}: {ing['quantity']} {ing['unit']}
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            if st.form_submit_button("Add Selected Ingredients to Fridge"):
                                if confirmed_ingredients:
                                    for ingredient in confirmed_ingredients:
                                        add_ingredient(
                                            st.session_state.username,
                                            ingredient['name'],
                                            ingredient['quantity'],
                                            ingredient['unit']
                                        )
                                    st.success(f"✅ Added {len(confirmed_ingredients)} ingredients to your fridge!")
                                    st.rerun()
                                else:
                                    st.warning("Please select at least one ingredient to add.")
                    else:
                        st.warning("No ingredients detected in the image. Try uploading a different image with better lighting and focus.")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("You can still add ingredients manually below.")
        
        st.markdown("---")
    
    # Display current fridge contents
    fridge_contents = get_fridge_contents(st.session_state.username)
    
    # Show current ingredients
    if fridge_contents:
        st.write("### Current Ingredients:")
        for item in fridge_contents:
            col1, col2 = st.columns([3, 1])
            quantity = item.get('quantity', 1)
            unit = item.get('unit', '')
            with col1:
                if unit:
                    st.write(f"- {item['ingredient']} ({quantity} {unit})")
                else:
                    st.write(f"- {item['ingredient']} (Quantity: {quantity})")
            with col2:
                if st.button("Delete", key=f"delete_{item['ingredient']}_{item['_id']}", on_click=delete_ingredient, args=(st.session_state.username, item['ingredient'])):
                    st.success(f"Deleted {item['ingredient']} from fridge.")
                    st.rerun()
    else:
        st.write("Your fridge is empty. Add some ingredients!")

    if input_method == "Manual Input":
        # Manual ingredient input
        st.markdown("### ✍️ Manual Input")
        new_ingredient = st.text_input("Ingredient")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            quantity = st.number_input("Amount", min_value=0.1, step=0.1)
        with col2:
            unit_type = st.selectbox("Unit", ["pieces", "g", "kg", "ml", "l", "tbsp", "tsp", "cup", "oz", "lb"])
        
        if st.button("Add Ingredient"):
            if new_ingredient:
                add_ingredient(st.session_state.username, new_ingredient, quantity, unit_type)
                st.session_state.new_ingredient = ""
                st.success(f"Added {new_ingredient} ({quantity} {unit_type}) to fridge.")
                st.rerun()
            else:
                st.error("Please enter an ingredient.")

def add_ingredient(user, ingredient, quantity, unit=''):
    """Add an ingredient to the user's fridge or update the quantity if it already exists."""
    existing_item = fridge_collection.find_one({'user': user, 'ingredient': ingredient})
    
    if existing_item:
        # If units match or existing item has no unit, update the quantity
        if existing_item.get('unit', '') == unit or not existing_item.get('unit'):
            new_quantity = existing_item.get('quantity', 0) + quantity
            fridge_collection.update_one(
                {'_id': existing_item['_id']},
                {'$set': {'quantity': new_quantity, 'unit': unit}}
            )
        else:
            # If units don't match, add as a new entry with a modified name
            fridge_collection.insert_one({
                'user': user,
                'ingredient': f"{ingredient} ({unit})",  # Add unit to name to differentiate
                'quantity': quantity,
                'unit': unit,
                'added_at': datetime.datetime.now()
            })
    else:
        # Add new ingredient if it doesn't exist
        fridge_collection.insert_one({
            'user': user,
            'ingredient': ingredient,
            'quantity': quantity,
            'unit': unit,
            'added_at': datetime.datetime.now()
        })
    
def delete_saved_recipe(recipe_id):
    """Delete a saved recipe from the database"""
    try:
        result = recipes.delete_one({'id': recipe_id, 'user': st.session_state.username})
        if result.deleted_count > 0:
            st.session_state.recipe_deleted = True
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting recipe: {str(e)}")
        return False

def generate_food_image(recipe_title, recipe_description=None):
    """Generate an image of food based on the recipe title and description using Gemini"""
    try:
        # Create a detailed food description for more accurate image generation
        image_description_prompt = f"""
        I need a detailed description of what {recipe_title} looks like as a finished dish.
        
        Include these details in your description:
        - Colors and visual appearance 
        - Texture and consistency
        - How it's plated or presented
        - Garnishes or toppings
        - Any distinctive visual features of this dish
        
        Keep your description under 100 words and focus only on visual appearance.
        
        Additional context about the dish:
        {recipe_description if recipe_description else ""}
        """
        
        # Get a detailed food description from Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(image_description_prompt)
        detailed_description = response.text if response.text else recipe_title
        
        # Now use the detailed description to request an image from Stability AI or Dall-E
        # For demo, we'll use an image keyword approach to bridge the gap
        refined_keywords = f"professional food photography of {recipe_title}, {detailed_description}, restaurant quality, high resolution food photo, top view, on a beautiful plate, food styling, culinary magazine quality"
        
        # URL encode the keywords
        encoded_keywords = refined_keywords.replace(' ', '+')
        image_api_url = f"https://image.pollinations.ai/prompt/{encoded_keywords}?width=800&height=600&nologo=true"
        
        # Get image from the generative API
        response = requests.get(image_api_url, stream=True)
        if response.status_code == 200:
            return response.content
            
        # Fallback to Unsplash if the generative API fails
        fallback_url = f"https://source.unsplash.com/featured/?{recipe_title.replace(' ', ',')},food,dish,cuisine"
        fallback_response = requests.get(fallback_url, stream=True)
        if fallback_response.status_code == 200:
            return fallback_response.content
            
        return None
    except Exception as e:
        st.warning(f"Could not generate image: {str(e)}")
        return None

def set_page_config():
    st.set_page_config(
        page_title="DishPal - AI Recipe Generator",
        page_icon="🍽️",
        layout="wide",
    )
    
    # Load and encode the background image
    try:
        with open("assets/bg.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading background image: {str(e)}")
        encoded_string = ""
    
    # Apply custom theme with food background
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
        
        .stApp {{
            background-image: url('data:image/jpeg;base64,{encoded_string}');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            min-height: 100vh;
            position: relative;
            background-repeat: no-repeat;
            background-color: #ffeaf0  /* Fallback color */
        }}

        .stApp::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 234, 240, 0.85);  /* Changed to a softer pink with less opacity */
            z-index: 0;
        }}

        .stApp > * {{
            position: relative;
            z-index: 1;
        }}
        
        /* Recipe metadata styling */
        .recipe-metadata {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #666;
            margin-bottom: 15px;
        }}
        
        /* Recipe section headers */
        .recipe-section-header {{
            color: #7c5e40;
            margin-bottom: 10px;
        }}
        
        /* Recipe content text */
        .recipe-content {{
            color: #333333;
            margin-bottom: 8px;
        }}
        
        h1, h2, h3, h4 {{
            font-family: 'Press Start 2P', cursive !important;
            color: #5b3c24;  /* Warm brown for headings */
            line-height: 1.6 !important;
            letter-spacing: 1px !important;
        }}
        
        /* Make headings a bit smaller to accommodate pixel font */
        h1 {{
            font-size: 1.8em !important;
        }}
        h2 {{
            font-size: 1.5em !important;
        }}
        h3 {{
            font-size: 1.2em !important;
        }}
        h4 {{
            font-size: 1em !important;
        }}
        
        /* Keep body text readable */
        p, li, div {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #333333;  /* Dark gray for better readability */
        }}
        
        /* Recipe card title styling */
        .recipe-card h3 {{
            font-family: 'Press Start 2P', cursive !important;
            font-size: 1.1em !important;
            line-height: 1.6 !important;
            margin-bottom: 15px !important;
        }}
        
        /* Button text in regular font */
        .stButton button {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #e68a5c;  /* Warm orange for buttons */
            color: white;
            padding: 0.8em 1.2em !important;
        }}
        .stButton button:hover {{
            background-color: #d17a4c;  /* Darker orange on hover */
        }}

        /* Dropdown styles - more specific selectors */
        .stSelectbox,
        .stSelectbox div,
        .stSelectbox span,
        div[data-baseweb="select"],
        div[data-baseweb="select"] *,
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] * {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: rgba(255, 203, 164, 0.95) !important;
            color: #5b3c24 !important;
        }}

        /* Target the menu container specifically */
        div[role="menu"],
        div[role="menu"] *,
        div[role="listbox"],
        div[role="listbox"] *,
        ul[role="listbox"],
        ul[role="listbox"] * {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: rgba(255, 203, 164, 0.95) !important;
            color: #5b3c24 !important;
        }}

        /* Base style for options */
        div[role="option"],
        li[role="option"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: rgba(255, 203, 164, 0.95) !important;
            color: #5b3c24 !important;
            border-left: 4px solid transparent !important;
            padding-left: 12px !important;
            transition: all 0.2s ease-in-out !important;
        }}

        /* Style hover and selected states for individual options */
        div[role="option"]:hover,
        li[role="option"]:hover,
        div[aria-selected="true"],
        li[aria-selected="true"] {{
            background-color: rgba(255, 219, 193, 0.95) !important;
            color: #5b3c24 !important;
            border-left: 4px solid #e68a5c !important;
        }}
        
        /* Override any default styles */
        .stSelectbox::before,
        .stSelectbox::after,
        div[data-baseweb="select"]::before,
        div[data-baseweb="select"]::after,
        div[data-baseweb="popover"]::before,
        div[data-baseweb="popover"]::after {{
            background-color: rgba(255, 203, 164, 0.95) !important;
        }}

        /* Target the select input */
        .stSelectbox input,
        div[data-baseweb="select"] input {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #5b3c24 !important;
        }}

        /* Target the dropdown arrow */
        .stSelectbox [role="presentation"] svg {{
            color: #5b3c24 !important;
        }}

        /* Text input styles */
        .stTextInput input, 
        .stTextArea textarea {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: rgba(248, 171, 197, 0.95) !important;
            color: #5b3c24 !important;
        }}

        /* Number input styles */
        .stNumberInput input {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: rgba(255, 203, 164, 0.95) !important;
            color: #5b3c24 !important;
        }}

        /* Download button styling */
        .stDownloadButton button {{
            background-color: #e68a5c !important;
            color: white !important;
        }}
        .stDownloadButton button:hover {{
            background-color: #d17a4c !important;
        }}

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: rgba(255, 234, 240, 0.95) !important;
            padding: 10px !important;
            border-radius: 10px !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            height: auto !important;
            padding: 10px 16px !important;
            background-color: rgba(255, 203, 164, 0.95) !important;
            border: 2px solid #e68a5c !important;
            border-radius: 8px !important;
            color: #5b3c24 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: rgba(255, 219, 193, 0.95) !important;
            border-color: #d17a4c !important;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: #e68a5c !important;
            border-color: #d17a4c !important;
            color: white !important;
        }}

        /* Tab content area */
        .stTabs [data-baseweb="tab-panel"] {{
            padding: 20px !important;
            background-color: rgba(255, 255, 255, 0.95) !important;
            border: 2px solid #ffbcd1 !important;
            border-radius: 10px !important;
            margin-top: 10px !important;
        }}

        /* Recipe cards and content boxes */
        .recipe-card, 
        .stAlert,
        .stSuccess,
        .stError,
        .stWarning,
        .stInfo {{
            background-color: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
        }}
        </style>
    """, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    # Set page config
    set_page_config()
    
    # Display the appropriate page based on session state
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'recipe':
        recipe_page()
    elif st.session_state.page == 'fridge':
        fridge_page()

if __name__ == "__main__":
    main()
