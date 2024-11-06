import streamlit as st
from bs4 import BeautifulSoup
import requests
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import re

# Load environment variables
load_dotenv()

# Check for API key at startup
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("Please ensure OPENROUTER_API_KEY is set in your .env file")
    st.stop()

# Initialize OpenAI client with OpenRouter
try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
except Exception as e:
    st.error(f"Error initializing OpenAI client: {str(e)}")
    client = None

def fetch_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Error fetching content: {str(e)}")
        return None

def extract_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract title
    title = soup.title.string.strip() if soup.title else ""
    
    # Extract meta description
    meta_desc = ""
    meta_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
    if meta_tag:
        meta_desc = meta_tag.get('content', '').strip()
    
    # Extract H1
    h1 = ""
    h1_tag = soup.find('h1')
    if h1_tag:
        h1 = h1_tag.get_text().strip()
    
    # Extract content specifically from article content area
    content_structure = []
    main_content = soup.find(class_=["article-single__content", "category_contents_details"])
    
    if main_content:
        # Remove unwanted sections before processing
        exclude_classes = [
            "single-post-new__tags",
            "single-post-new__author-top",
            "follow-button",
            "single-post__recommended"
        ]
        
        for class_name in exclude_classes:
            for element in main_content.find_all(class_=class_name):
                element.decompose()
        
        current_section = {'h2': 'Introduction', 'content': []}
        
        for element in main_content.find_all(['h2', 'p', 'h3', 'ul', 'ol', 'table']):
            text_content = element.get_text().strip()
            if not text_content:
                continue
                
            if element.name == 'h2':
                if current_section['content']:
                    content_structure.append(current_section)
                current_section = {'h2': text_content, 'content': []}
            else:
                current_section['content'].append({
                    'type': element.name,
                    'text': text_content
                })
        
        # Add the last section if it has content
        if current_section['content']:
            content_structure.append(current_section)
    
    return {
        'title': title,
        'meta_description': meta_desc,
        'h1': h1,
        'sections': content_structure
    }

def translate_content(content):
    try:
        prompt = f"""Translate this crypto news article from English to Thai. 
        Write for Thai audiences with basic crypto knowledge.
        Use plain but professional Thai language.
        Use transliteration for crypto terms where appropriate.
        
        Original Content:
        Title: {content['title']}
        Meta Description: {content['meta_description']}
        H1: {content['h1']}
        
        Sections:
        {json.dumps(content['sections'], indent=2)}
        
        Please provide the translation in this format:
        Title:
        [Thai translation]
        
        Meta Description:
        [Thai translation]
        
        H1:
        [Thai translation]
        
        [For each H2 section]:
        [Thai translation of H2]
        [Thai translation of corresponding content]
        """

        response = client.chat.completions.create(
            model="openai/o1-mini-2024-09-12",
            messages=[
                {"role": "system", "content": "You are a professional crypto news translator specializing in English to Thai translation. You understand both crypto news terminology and Thai language conventions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3000
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None

# Add these near the top of the file, after imports
def init_session_state():
    if 'extracted_content' not in st.session_state:
        st.session_state.extracted_content = None
    if 'translated_content' not in st.session_state:
        st.session_state.translated_content = None
    if 'url' not in st.session_state:
        st.session_state.url = ""

# Streamlit UI
st.set_page_config(layout="wide")
init_session_state()

st.title("Crypto News Translator")
st.subheader("English to Thai News Translation")

# URL Input and buttons on the same row
col1, col2, col3 = st.columns([4, 1, 1])  # Adjust ratio for URL input and two buttons
with col1:
    url = st.text_input("Enter the English news article URL:", 
                       value=st.session_state.url,
                       label_visibility="collapsed",
                       key="url_input")
with col2:
    extract_button = st.button(
        "Extract and Translate", 
        type="primary",  # Makes the button green
        use_container_width=True
    )
with col3:
    st.button(
        "Clear Session",  # Shortened text
        use_container_width=True,
        help="Click to clear all previous translations",
        type="secondary",
        key="clear_button",
        on_click=lambda: (st.session_state.clear(), st.rerun())
    )

if extract_button:
    if url:
        with st.spinner("Fetching and analysing content..."):
            html_content = fetch_content(url)
            if html_content:  # Check if content was successfully fetched
                # Extract content
                extracted_content = extract_content(html_content)
                if extracted_content and any(extracted_content.values()):  # Check if extraction was successful
                    st.session_state.extracted_content = extracted_content
                    # Translate content
                    translated_content = translate_content(extracted_content)
                    if translated_content:  # Check if translation was successful
                        st.session_state.translated_content = translated_content
                        st.session_state.url = url
                    else:
                        st.error("Translation failed. Please try again.")
                else:
                    st.error("Could not extract content from the webpage. Please check the URL.")
            else:
                st.error("Could not fetch content from the URL. Please check if the URL is accessible.")

# Display content if available in session state
if st.session_state.extracted_content and st.session_state.translated_content:
    # Split translated content into sections
    sections = st.session_state.translated_content.split('\n\n')
    
    # Display side-by-side content
    st.subheader("Original vs Translated Content")
    
    # Create two columns for side-by-side display with more width
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### English Content")
        
        # Combine title, meta, and H1 in a more compact way
        with st.expander("Title, Meta & H1", expanded=True):
            st.text_area("Title", 
                        value=st.session_state.extracted_content['title'],
                        height=50,
                        key="orig_title",
                        disabled=True)
            
            st.text_area("Meta Description", 
                        value=st.session_state.extracted_content['meta_description'],
                        height=50,
                        key="orig_meta",
                        disabled=True)
            
            st.text_area("H1", 
                        value=st.session_state.extracted_content['h1'],
                        height=50,
                        key="orig_h1",
                        disabled=True)
        
        st.markdown("**Main Content**")
        original_content = ""
        for section in st.session_state.extracted_content['sections']:
            original_content += f"\n## {section['h2']}\n"
            for content in section['content']:
                original_content += f"{content['text']}\n\n"
        
        st.text_area("Original content:",
                    value=original_content,
                    height=600,
                    key="orig_content",
                    disabled=True)
    
    with col2:
        st.markdown("### Thai Content")
        
        # Combine title, meta, and H1 in a more compact way
        with st.expander("Title, Meta & H1", expanded=True):
            title_section = st.text_area("Title", 
                                       value=sections[0].replace("Title:", "").strip(),
                                       height=50,
                                       key="title_editor")
            
            meta_section = st.text_area("Meta Description", 
                                      value=sections[1].replace("Meta Description:", "").strip(),
                                      height=50,
                                      key="meta_editor")
            
            h1_section = st.text_area("H1", 
                                    value=sections[2].replace("H1:", "").strip(),
                                    height=50,
                                    key="h1_editor")
        
        st.markdown("**Main Content**")
        main_content = st.text_area("Edit content:", 
                                  value="\n\n".join(sections[3:]),
                                  height=600,
                                  key="content_editor")
    
    # Download button in a more prominent position
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        combined_content = f"""Title:
{title_section}

Meta Description:
{meta_section}

H1:
{h1_section}

Content:
{main_content}"""
        
        st.download_button(
            label="📥 Download Translated Content",
            data=combined_content,
            file_name=f"translated_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
elif url:
    st.warning("Please click 'Extract and Translate' to process the URL.")
else:
    st.warning("Please enter a URL to translate.")