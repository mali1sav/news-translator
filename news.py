import streamlit as st
from dotenv import load_dotenv
import os
from urllib.parse import urlparse
import requests
import random
import time
from typing import Optional, List, Dict
from bs4 import BeautifulSoup
import asyncio
import httpx
from docx import Document
from io import BytesIO

# Load environment variables
load_dotenv()

# Initialize OpenAI client with OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in environment variables.")
    st.stop()

client = httpx.AsyncClient(
    base_url="https://openrouter.ai/api/v1",
    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
)

# Default URL
DEFAULT_URL = "https://cryptonews.com/news/ripple-ceo-brad-garlinghouse-celebrates-crypto-candidates-victory-in-u-s-election-2024/"

def get_random_user_agent() -> str:
    """Return a random user agent string."""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)\
         Chrome/58.0.3029.110 Safari/537.3',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)\
         Version/14.0.3 Safari/605.1.15',
        # Add more user agents as needed
    ]
    return random.choice(user_agents)

def fetch_webpage(url: str, max_retries: int = 3) -> Optional[str]:
    """Fetch webpage content with retry logic."""
    headers = {
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

    for attempt in range(max_retries):
        try:
            st.write(f"Fetching URL (attempt {attempt + 1})")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            st.error(f"Error fetching webpage (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
    return None

def extract_structured_content(html: str, url: str) -> Dict:
    """Extract structured content from HTML using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove author sections and other unwanted content first
        unwanted_classes = [
            'single-post-new__author-top',  # Author section
            'author', 'byline', 'meta', 'post-meta', 'site-info', 
            'footer', 'nav', 'advertisement', 'ad', 'sidebar', 'newsletter',
            'author-bio', 'author-box', 'author-info'
        ]
        
        for class_name in unwanted_classes:
            for element in soup.find_all(class_=class_name):
                element.decompose()

        content = {
            'title': '',
            'meta_description': '',
            'h1': '',
            'sections': []
        }

        # Extract title
        if soup.title:
            content['title'] = soup.title.string.strip()

        # Extract meta description
        meta_desc = soup.find('meta', {'name': ['description', 'og:description']})
        if meta_desc and meta_desc.get('content'):
            content['meta_description'] = meta_desc['content']

        # Extract H1
        h1_tag = soup.find('h1')
        if h1_tag:
            content['h1'] = h1_tag.text.strip()

        # Find main content area
        main_content = soup.find(['main', 'article']) or soup.find(class_=[
            'content', 'main-content', 'article-content', 'entry-content', 
            'post-content', 'article-body', 'story-content'
        ])

        if main_content:
            current_section = None

            # Remove author information and unwanted elements
            unwanted_classes = [
                'author', 'byline', 'meta', 'post-meta', 'site-info', 
                'footer', 'nav', 'advertisement', 'ad', 'sidebar', 'newsletter'
            ]
            for unwanted_class in unwanted_classes:
                for div in main_content.find_all(class_=unwanted_class):
                    div.decompose()

            # Remove unwanted tags globally within main_content
            unwanted_tags = ['script', 'style', 'aside', 'noscript', 'iframe']
            for tag in unwanted_tags:
                for element in main_content.find_all(tag):
                    element.decompose()

            # Initialize with a default section if no headings are present
            if not main_content.find(['h2', 'h3']):
                current_section = {
                    'heading': 'Main Content',
                    'content': []
                }
                content['sections'].append(current_section)

            # Process content elements
            for element in main_content.find_all(['h2', 'h3', 'p', 'ul', 'ol', 'table', 'blockquote']):
                # Skip elements with unwanted text
                text = element.get_text(strip=True).lower()
                if text in ['main content', 'เนื้อหาหลัก', 'about author', 'author bio']:
                    continue

                # Start new section on headings
                if element.name in ['h2', 'h3']:
                    current_section = {
                        'heading': element.text.strip(),
                        'content': []
                    }
                    content['sections'].append(current_section)
                    continue

                # Create default section if none exists
                if not current_section:
                    current_section = {
                        'heading': 'Main Content',  # Default heading
                        'content': []
                    }
                    content['sections'].append(current_section)

                # Process content
                if element.name == 'p' and len(text) > 20:
                    current_section['content'].append({
                        'type': 'paragraph',
                        'text': element.text.strip()
                    })
                elif element.name in ['ul', 'ol']:
                    items = [li.text.strip() for li in element.find_all('li') if li.text.strip()]
                    if items:
                        current_section['content'].append({
                            'type': 'list',
                            'items': items,
                            'list_type': element.name
                        })
                elif element.name == 'table':
                    table_data = []
                    headers = []

                    # Extract headers
                    header_row = element.find('thead')
                    if header_row:
                        headers = [th.text.strip() for th in header_row.find_all(['th', 'td'])]

                    # Extract data rows
                    for row in element.find_all('tr'):
                        cells = [cell.text.strip() for cell in row.find_all(['th', 'td'])]
                        if any(cells) and cells != headers:  # Skip empty rows and header row
                            table_data.append(cells)

                    if table_data:
                        current_section['content'].append({
                            'type': 'table',
                            'headers': headers,
                            'data': table_data
                        })
                elif element.name == 'blockquote':
                    quote_text = element.text.strip()
                    if quote_text:
                        current_section['content'].append({
                            'type': 'quote',
                            'text': quote_text
                        })

        return content

    except Exception as e:
        st.error(f"Error extracting structured content: {str(e)}")
        return {'title': '', 'meta_description': '', 'h1': '', 'sections': []}

async def translate_text_async(client: httpx.AsyncClient, system_prompt: str, text: str) -> str:
    """Asynchronously translate text using OpenRouter API."""
    try:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate the following text into Thai following system prompt instructions:\n{text}"}
                ]
            },
            timeout=60  # Adjust timeout as needed
        )
        response.raise_for_status()
        data = response.json()
        translated_text = data['choices'][0]['message']['content'].strip()
        return translated_text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

async def translate_content_async(content: Dict, system_prompt: str) -> Dict:
    """Asynchronously translate all relevant fields in the content."""
    translated_content = {
        'title': '',
        'meta_description': '',
        'h1': '',
        'sections': []
    }

    tasks = []

    # Collect all texts to translate
    if content.get('title'):
        tasks.append(translate_text_async(client, system_prompt, content['title']))
    if content.get('meta_description'):
        tasks.append(translate_text_async(client, system_prompt, content['meta_description']))
    if content.get('h1'):
        tasks.append(translate_text_async(client, system_prompt, content['h1']))

    for section in content.get('sections', []):
        if section.get('heading'):
            tasks.append(translate_text_async(client, system_prompt, section['heading']))
        for item in section.get('content', []):
            if item['type'] in ['paragraph', 'quote']:
                tasks.append(translate_text_async(client, system_prompt, item['text']))
            elif item['type'] == 'list':
                for li in item['items']:
                    tasks.append(translate_text_async(client, system_prompt, li))
            elif item['type'] == 'table':
                for row in item['data']:
                    for cell in row:
                        tasks.append(translate_text_async(client, system_prompt, cell))

    # Perform all translations concurrently
    translations = await asyncio.gather(*tasks, return_exceptions=True)

    # Assign translations back to the structured content
    translation_iter = iter(translations)

    # Translate title
    translated_content['title'] = next(translation_iter, content.get('title', ''))

    # Translate meta_description
    translated_content['meta_description'] = next(translation_iter, content.get('meta_description', ''))

    # Translate h1
    translated_content['h1'] = next(translation_iter, content.get('h1', ''))

    # Translate sections
    for section in content.get('sections', []):
        translated_section = {'heading': '', 'content': []}
        # Translate heading
        translated_section['heading'] = next(translation_iter, section.get('heading', ''))
        # Translate content
        for item in section.get('content', []):
            translated_item = {'type': item['type']}
            if item['type'] in ['paragraph', 'quote']:
                translated_item['text'] = next(translation_iter, item.get('text', ''))
            elif item['type'] == 'list':
                translated_items = []
                for _ in item['items']:
                    translated_items.append(next(translation_iter, ''))
                translated_item['items'] = translated_items
                translated_item['list_type'] = item.get('list_type', 'ul')
            elif item['type'] == 'table':
                translated_headers = item.get('headers', [])
                translated_data = []
                for row in item.get('data', []):
                    translated_row = []
                    for _ in row:
                        translated_row.append(next(translation_iter, ''))
                    translated_data.append(translated_row)
                translated_item['headers'] = translated_headers
                translated_item['data'] = translated_data
            translated_section['content'].append(translated_item)
        translated_content['sections'].append(translated_section)

    return translated_content

def create_comparison_doc(url: str, content: Dict, translated_content: Dict) -> BytesIO:
    """Create a .docx document with side-by-side comparison in memory."""
    doc = Document()
    
    # Title
    doc.add_heading('Crypto News Translation Comparison', 0)
    doc.add_paragraph(f"Original URL: {url}")
    
    # Add Title, Meta & H1 section
    doc.add_heading('Title, Meta & H1', level=1)
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Light List Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'English'
    hdr_cells[1].text = 'Thai'
    
    # Add Title
    if content.get('title'):
        row = table.add_row().cells
        row[0].text = 'Title:\n' + content['title']
        row[1].text = 'Title:\n' + translated_content.get('title', '')
    
    # Add Meta Description
    if content.get('meta_description'):
        row = table.add_row().cells
        row[0].text = 'Meta Description:\n' + content['meta_description']
        row[1].text = 'Meta Description:\n' + translated_content.get('meta_description', '')
    
    # Add H1
    if content.get('h1'):
        row = table.add_row().cells
        row[0].text = 'H1:\n' + content['h1']
        row[1].text = 'H1:\n' + translated_content.get('h1', '')

    # Main Content Sections
    for section_idx, (orig_section, trans_section) in enumerate(zip(content.get('sections', []), translated_content.get('sections', []))):
        # Skip sections with heading "Main Content"
        if orig_section.get('heading', '').strip().lower() == "main content":
            continue

        doc.add_heading(f"Section {section_idx + 1}: {orig_section.get('heading', '')}", level=1)
        table = doc.add_table(rows=1, cols=2)
        table.style = 'Light List Accent 1'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'English'
        hdr_cells[1].text = 'Thai'

        # Heading
        row = table.add_row().cells
        row[0].text = orig_section.get('heading', '')
        row[1].text = trans_section.get('heading', '')

        # Process section content
        for orig_item, trans_item in zip(orig_section.get('content', []), trans_section.get('content', [])):
            if orig_item['type'] == 'paragraph':
                row = table.add_row().cells
                row[0].text = orig_item.get('text', '')
                row[1].text = trans_item.get('text', '')
            elif orig_item['type'] == 'list':
                orig_list = '\n'.join([f"- {li}" for li in orig_item.get('items', [])])
                trans_list = '\n'.join([f"- {li}" for li in trans_item.get('items', [])])
                row = table.add_row().cells
                row[0].text = orig_list
                row[1].text = trans_list
            elif orig_item['type'] == 'table':
                orig_table = '\n'.join([' | '.join(row) for row in orig_item.get('data', [])])
                trans_table = '\n'.join([' | '.join(row) for row in trans_item.get('data', [])])
                row = table.add_row().cells
                row[0].text = orig_table
                row[1].text = trans_table
            elif orig_item['type'] == 'quote':
                row = table.add_row().cells
                row[0].text = f"\"{orig_item.get('text', '')}\""
                row[1].text = f"\"{trans_item.get('text', '')}\""

    # Save to in-memory file
    doc_io = BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)
    return doc_io


def display_content_comparison(content: Dict, translated_content: Dict):
    """Display original and translated content side by side with unique keys."""
    # Title, Meta & H1
    with st.expander("Title, Meta & H1", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.text_area(
                label="English Title",
                value=content.get('title', ''),
                height=68,
                key="en_title_main_unique"
            )
            
            if content.get('meta_description'):             
                st.text_area(
                    label="English Meta Description",
                    value=content['meta_description'],
                    height=100,
                    key="en_meta_main_unique"
                )
            
            if content.get('h1'):        
                st.text_area(
                    label="English H1",
                    value=content['h1'],
                    height=68,
                    key="en_h1_main_unique"
                )
        
        with col2:
            st.text_area(
                label="Thai Title",
                value=translated_content.get('title', ''),
                height=68,
                key="th_title_main_unique"
            )
            
            if translated_content.get('meta_description'):
                st.text_area(
                    label="Thai Meta Description",
                    value=translated_content['meta_description'],
                    height=100,
                    key="th_meta_main_unique"
                )
            
            if translated_content.get('h1'):
                st.text_area(
                    label="Thai H1",
                    value=translated_content['h1'],
                    height=68,
                    key="th_h1_main_unique"
                )

    # Main Content
    for section_idx, (orig_section, trans_section) in enumerate(zip(content.get('sections', []), translated_content.get('sections', []))):
        # Skip sections with heading "Main Content"
        if orig_section.get('heading', '').strip().lower() == "main content":
            continue
        
        with st.expander(f"Section {section_idx + 1}: {orig_section.get('heading', '')}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_area(
                    label=f"English Heading {section_idx + 1}",
                    value=orig_section.get('heading', ''),
                    height=68,
                    key=f"en_heading_{section_idx}_unique"
                )
                
                for item_idx, item in enumerate(orig_section.get('content', [])):
                    if item['type'] == 'paragraph':
                        st.text_area(
                            label=f"English Paragraph {section_idx + 1}.{item_idx + 1}",
                            value=item.get('text', ''),
                            height=100,
                            key=f"en_para_{section_idx}_{item_idx}_unique"
                        )
                    elif item['type'] == 'list':
                        for li_idx, li in enumerate(item.get('items', [])):
                            st.text_area(
                                label=f"English List Item {section_idx + 1}.{item_idx + 1}.{li_idx + 1}",
                                value=li,
                                height=68,
                                key=f"en_list_{section_idx}_{item_idx}_{li_idx}_unique"
                            )
                    elif item['type'] == 'table':
                        if item.get('headers'):
                            st.text_area(
                                label=f"English Table Headers {section_idx + 1}.{item_idx + 1}",
                                value=" | ".join(item['headers']),
                                height=68,
                                key=f"en_table_header_{section_idx}_{item_idx}_unique"
                            )
                        for row_idx, row in enumerate(item.get('data', [])):
                            st.text_area(
                                label=f"English Table Row {section_idx + 1}.{item_idx + 1}.{row_idx + 1}",
                                value=" | ".join(row),
                                height=68,
                                key=f"en_table_{section_idx}_{item_idx}_{row_idx}_unique"
                            )
                    elif item['type'] == 'quote':
                        st.text_area(
                            label=f"English Quote {section_idx + 1}.{item_idx + 1}",
                            value=item.get('text', ''),
                            height=120,
                            key=f"en_quote_{section_idx}_{item_idx}_unique"
                        )
            
            with col2:
                if trans_section.get('heading'):
                    st.text_area(
                        label=f"Thai Heading {section_idx + 1}",
                        value=trans_section['heading'],
                        height=68,
                        key=f"th_heading_{section_idx}_unique"
                    )
                
                for item_idx, item in enumerate(trans_section.get('content', [])):
                    if item['type'] == 'paragraph':
                        st.text_area(
                            label=f"Thai Paragraph {section_idx + 1}.{item_idx + 1}",
                            value=item.get('text', ''),
                            height=100,
                            key=f"th_para_{section_idx}_{item_idx}_unique"
                        )
                    elif item['type'] == 'list':
                        for li_idx, li in enumerate(item.get('items', [])):
                            st.text_area(
                                label=f"Thai List Item {section_idx + 1}.{item_idx + 1}.{li_idx + 1}",
                                value=li,
                                height=68,
                                key=f"th_list_{section_idx}_{item_idx}_{li_idx}_unique"
                            )
                    elif item['type'] == 'table':
                        if item.get('headers'):
                            st.text_area(
                                label=f"Thai Table Headers {section_idx + 1}.{item_idx + 1}",
                                value=" | ".join(item['headers']),
                                height=68,
                                key=f"th_table_header_{section_idx}_{item_idx}_unique"
                            )
                        for row_idx, row in enumerate(item.get('data', [])):
                            st.text_area(
                                label=f"Thai Table Row {section_idx + 1}.{item_idx + 1}.{row_idx + 1}",
                                value=" | ".join(row),
                                height=68,
                                key=f"th_table_{section_idx}_{item_idx}_{row_idx}_unique"
                            )
                    elif item['type'] == 'quote':
                        st.text_area(
                            label=f"Thai Quote {section_idx + 1}.{item_idx + 1}",
                            value=item.get('text', ''),
                            height=120,
                            key=f"th_quote_{section_idx}_{item_idx}_unique"
                        )

def get_edited_translated_content(translated_content: Dict) -> Dict:
    """Retrieve edited Thai content from Streamlit session state."""
    edited_translated = {
        'title': st.session_state.get('th_title_main_unique', ''),
        'meta_description': st.session_state.get('th_meta_main_unique', ''),
        'h1': st.session_state.get('th_h1_main_unique', ''),
        'sections': []
    }
    for idx, section in enumerate(translated_content.get('sections', [])):
        edited_section = {
            'heading': st.session_state.get(f"th_heading_{idx}_unique", section.get('heading', '')),
            'content': []
        }
        for item_idx, item in enumerate(section.get('content', [])):
            if item['type'] == 'paragraph':
                edited_text = st.session_state.get(f"th_para_{idx}_{item_idx}_unique", item.get('text', ''))
                edited_section['content'].append({'type': 'paragraph', 'text': edited_text})
            elif item['type'] == 'list':
                edited_items = []
                for li_idx in range(len(item.get('items', []))):
                    edited_li = st.session_state.get(f"th_list_{idx}_{item_idx}_{li_idx}_unique", item['items'][li_idx])
                    edited_items.append(edited_li)
                edited_section['content'].append({'type': 'list', 'items': edited_items, 'list_type': item.get('list_type', 'ul')})
            elif item['type'] == 'table':
                edited_table = []
                for row_idx in range(len(item.get('data', []))):
                    edited_row = []
                    for cell_idx in range(len(item['data'][row_idx])):
                        edited_cell = st.session_state.get(f"th_table_{idx}_{item_idx}_{row_idx}_unique", item['data'][row_idx][cell_idx])
                        edited_row.append(edited_cell)
                    edited_table.append(edited_row)
                edited_section['content'].append({'type': 'table', 'data': edited_table, 'headers': item.get('headers', [])})
            elif item['type'] == 'quote':
                edited_quote = st.session_state.get(f"th_quote_{idx}_{item_idx}_unique", item.get('text', ''))
                edited_section['content'].append({'type': 'quote', 'text': edited_quote})
        edited_translated['sections'].append(edited_section)
    return edited_translated


def main():
    st.set_page_config(layout="wide")
    st.title("Crypto News Translator (EN → TH)")
    
    # Initialize session state for content
    if 'content' not in st.session_state:
        st.session_state.content = None
    if 'translated_content' not in st.session_state:
        st.session_state.translated_content = None
    if 'url' not in st.session_state:
        st.session_state.url = DEFAULT_URL

    # URL input
    url = st.text_input("Enter a webpage URL:", value=st.session_state.url)
    
    # Extract and Translate button
    if st.button("Extract and Translate"):
        st.session_state.url = url
        if url:
            with st.spinner("Processing..."):
                # Fetch webpage
                html = fetch_webpage(url)
                
                if html:
                    # Extract structured content
                    content = extract_structured_content(html, url)
                    st.session_state.content = content  # Assign to session state
                    
                    if content.get('sections'):
                        # Translate content asynchronously
                        system_prompt = """
You are a professional Crypto news translator. Your task is to translate this crypto news article from English to Thai.

Instructions:
1. Write for Thai audiences with basic crypto knowledge, using semi-professional Thai language.
2. Use Thai crypto terms where appropriate.
3. Maintain the original meaning while making it natural in Thai.
4.If headings and content contain names such as people, company, or coin names, DO NOT translate names but ensure to translate the rest. 
5. Do not add explanations nor comments. Focus on translating the content. Return only the translated content.
"""
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            translated_content = loop.run_until_complete(translate_content_async(content, system_prompt))
                            st.session_state.translated_content = translated_content  # Assign to session state
                        except Exception as e:
                            st.error(f"Error during translation: {e}")
                            translated_content = content  # Fallback to original content
                            st.session_state.translated_content = translated_content
                        
                        st.success("Content extracted and translated.")
                        # Removed the direct call to display_content_comparison here
                    else:
                        st.error("No content could be extracted.")
                else:
                    st.error("Failed to fetch webpage.")
        else:
            st.warning("Please enter a valid URL.")
    
    # If we have content in session state, display it
    if st.session_state.content and st.session_state.translated_content:
        # Display content comparison
        display_content_comparison(st.session_state.content, st.session_state.translated_content)
        
        # Retrieve edited content
        edited_content = get_edited_translated_content(st.session_state.translated_content)
        
        # Create and download document
        with st.spinner("Generating document..."):
            edited_doc = create_comparison_doc(st.session_state.url, st.session_state.content, edited_content)
        
        st.download_button(
            label="Download Edited Document",
            data=edited_doc,
            file_name=f"{urlparse(st.session_state.url).path.strip('/').replace('/', '-')}_edited.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )


if __name__ == "__main__":
    main()
