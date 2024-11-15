import streamlit as st
import os
import asyncio
import httpx
from dotenv import load_dotenv
from docx import Document
from io import BytesIO
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Constants
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in environment variables.")
    st.stop()

API_BASE_URL = "https://openrouter.ai/api/v1"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

# Initialize Async Client
client = httpx.AsyncClient(base_url=API_BASE_URL, headers=HEADERS)

# Revised TAGS Structure
TAGS = {
    "Primary Categories": [
        "Bitcoin News",
        "Ethereum News",
        "Altcoin News",
        "DeFi News",
        "NFT News",
        "Industry News"
    ],
    "Topics": [
        "Market Trends",
        "Exchange News",
        "Technical Analysis",
        "Cyber Security",
        "Technology",
        "Regulation",
        "Adoption",
        "Investment",
        "Partnership",
        "SEC",
        "Legal"
    ],
    "Cryptocurrencies": {
        "Layer 1": [
            "Solana",
            "Cardano",
            "Avalanche",
            "Polkadot",
            "Cosmos",
            "NEAR Protocol",
            "Fantom",
            "TRON",
            "Other Layer 1"

        ],
        "Layer 2": [
            "Polygon",
            "Arbitrum",
            "Optimism",
            "Base",
            "Other Layer 2"
        ],
        "Infrastructure": [
            "Chainlink",
            "The Graph",
            "Quant",
            "Ren Protocol",
            "Other Infrastructure"
        ],
        "DeFi": [
            "Protocols",
            "Uniswap",
            "Aave",
            "Maker",
            "Curve",
            "Synthetix",
            "Compound",
            "Other DeFi platforms"
        ],
        "AI": [
            "Render",
            "Fetch.ai",
            "Ocean Protocol",
            "TAO",
            "Other AI coins"
        ],

        "Gaming & Metaverse": [
            "The Sandbox",
            "Decentraland",
            "Axie Infinity",
            "Immutable X",
            "Other Gaming & Metaverse"
        ],
        "Meme": [
            "Dogecoin",
            "Shiba Inu",
            "Pepe",
            "Floki",
            "Bonk",
            "Friend.tech",
            "Memecoin",
            "Other Meme"
        ],
        "Stable Coins": [
            "Tether",
            "USDC",
            "DAI",
            "Other Stable Coins"
        ]
    },
    "Market Infrastructure": {
        "Exchanges": [
            "Coinbase",
            "Binance",
            "Kraken",
            "KuCoin",
            "OKx",
            "Gemini",
            "Bitfinex",
            "Bitstamp",
            "Other Exchanges"
        ],
        "Data & Analytics": [
            "CoinGecko",
            "CoinMarketCap",
            "Glassnode",
            "Messari",
            "Chainalysis",
            "Etherscan",
            "Other Data & Analytics"
        ],
        "Wallets & Custody": [
            "Ledger",
            "Trezor",
            "MetaMask",
            "Trust Wallet",
            "Exodus",
            "Tangem",
            "Other Wallets & Custody"
        ]
    }
}

# Update the ALL_TAGS generation
ALL_TAGS = set(TAGS["Primary Categories"])
ALL_TAGS.update(TAGS["Topics"])
for category in TAGS["Cryptocurrencies"].values():
    ALL_TAGS.update(category)
for category in TAGS["Market Infrastructure"].values():
    ALL_TAGS.update(category)
ALL_TAGS = sorted(ALL_TAGS)  # Sorted list for consistent ordering

async def generate_meta_description(content: str) -> str:
    """
    Generate a Thai meta description using LLM.
    """
    system_prompt = """
You are a professional crypto content writer. Create a meta description in Thai for this crypto news article.
Requirements:
- Keep length no more than 160 characters
- Capture the key points: price movements, market impact, and key events
- Use natural Thai language
- Focus on the main news angle and impact
- Return ONLY the translated text, no explanations
- If contain names such as people, company, or coin names, DO NOT translate names but ensure to translate the rest
"""
    try:
        response = await client.post(
            "/chat/completions",
            json={
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create a Thai meta description for this article:\n{content}"}
                ]
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        thai_meta = data['choices'][0]['message']['content'].strip()
        # Ensure it's not longer than 160 characters
        if len(thai_meta) > 160:
            thai_meta = thai_meta[:157] + "..."
        return thai_meta
    except Exception as e:
        st.error(f"Meta description generation error: {e}")
        return "ไม่มีคำอธิบาย"

def parse_article(content: str) -> Dict[str, Any]:
    """
    Parse the article content into structured elements with content split into chunks.
    Each chunk contains 2-3 paragraphs separated by blank lines.
    """
    try:
        parsed_content = {
            'title': '',
            'meta_description': '',
            'h1': '',
            'sections': []
        }

        # Split content into lines and clean up
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        # Skip unwanted content
        skip_patterns = [
            'Crypto Reporter',
            'Share',
            'Last updated:',
            'Why Trust Cryptonews',
            'GMT'
        ]

        content_lines = [line for line in lines if not any(pattern in line for pattern in skip_patterns)]

        if not content_lines:
            raise ValueError("No valid content found after filtering.")

        # Extract title (first line)
        parsed_content['title'] = content_lines[0]
        parsed_content['h1'] = content_lines[0]

        # Assume the first paragraph after the title is the meta description
        if len(content_lines) > 1:
            parsed_content['meta_description'] = content_lines[1]
        else:
            parsed_content['meta_description'] = "Not provided"

        # Process main content into sections based on headings
        current_section = {'heading': 'Main Content', 'content': []}
        for line in content_lines[1:]:
            if line.isupper() or line.endswith(':'):  # Simple heuristic for headings
                if current_section['content']:
                    current_section['content'] = '\n\n'.join(current_section['content'])
                    parsed_content['sections'].append(current_section)
                current_section = {'heading': line, 'content': []}
            else:
                current_section['content'].append(line)
        # Add the last section
        if current_section['content']:
            current_section['content'] = '\n\n'.join(current_section['content'])
            parsed_content['sections'].append(current_section)

        # Further split main content into chunks of 2-3 paragraphs
        for section in parsed_content['sections']:
            paragraphs = section['content'].split('\n\n')
            chunks = [paragraphs[i:i + 3] for i in range(0, len(paragraphs), 3)]
            chunked_sections = []
            for chunk in chunks:
                chunked_sections.append({
                    'heading': section['heading'],
                    'content': '\n\n'.join(chunk)
                })
            section['chunks'] = chunked_sections
            del section['content']  # Remove original content

        # Validate parsed content
        if not parsed_content['title'] or not parsed_content['sections']:
            raise ValueError("Missing required content elements")

        return parsed_content

    except Exception as e:
        st.error(f"Error parsing article content: {str(e)}")
        return {'title': '', 'meta_description': 'Not provided', 'h1': '', 'sections': []}

async def categorize_with_llm(content: str) -> List[str]:
    """
    Categorize the news article using Gemini and return a list of relevant tags.
    """
    system_prompt = """You are a cryptocurrency news categorization expert. Analyze this article and suggest relevant tags.

Content Analysis Requirements:
1. Primary Category (Must include one):
   - If about Bitcoin: "Bitcoin News"
   - If about Ethereum: "Ethereum News"
   - If about other cryptocurrencies: "Altcoin News"

2. Specific Cryptocurrencies:
   - Identify all mentioned cryptocurrencies
   - Include their ecosystem tags (e.g., "Solana", "Cardano", etc.)

3. Market Infrastructure:
   - Identify mentioned exchanges, wallets, or data providers
   - Include relevant infrastructure tags

4. Topics:
   - Market analysis/trends
   - Technical developments
   - Regulatory news
   - Business developments

5. Special Categories:
   - Check for DeFi, NFT, or Gaming content
   - Identify if content relates to Layer 1/Layer 2 solutions

Rules:
- Maximum 5 most relevant tags
- Only use tags from the provided list
- Prioritize specificity over generality
- Ensure tags accurately reflect the main focus of the article

Return only comma-separated tags from the available list."""

    try:
        response = await client.post(
            "/chat/completions",
            json={
                "model": "google/gemini-flash-1.5-8b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Available tags:\n{', '.join(ALL_TAGS)}\n\nArticle content for analysis:\n{content}"}
                ],
                "temperature": 0.3,  # Lower temperature for more focused responses
                "max_tokens": 100    # Limit response length
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        tags_str = data['choices'][0]['message']['content'].strip()
        
        # Process and validate tags
        suggested_tags = [tag.strip() for tag in tags_str.split(',') if tag.strip() in ALL_TAGS]
        
        # Ensure we have at least one primary category
        has_primary = any(tag in TAGS["Primary Categories"] for tag in suggested_tags)
        if not has_primary:
            # Add most relevant primary category based on content
            if 'bitcoin' in content.lower():
                suggested_tags.insert(0, "Bitcoin News")
            elif 'ethereum' in content.lower():
                suggested_tags.insert(0, "Ethereum News")
            else:
                suggested_tags.insert(0, "Altcoin News")
        
        return suggested_tags[:5]  # Return top 5 tags
        
    except Exception as e:
        st.error(f"Categorization error: {e}")
        return []

def display_tag_section(current_tags: List[str]):
    """
    Display all tags in 6 columns with clear headings.
    """
    st.markdown("### Tag Management")
    st.write("Current Tags:", ", ".join(current_tags))
    
    # Create 6 columns
    cols = st.columns(6)
    
    # Column 1: Primary Categories
    with cols[0]:
        st.markdown("##### Primary Categories")
        for tag in TAGS["Primary Categories"]:
            key = f"primary_{tag.replace(' ', '_')}"
            if st.checkbox(tag, value=tag in current_tags, key=key):
                if tag not in current_tags:
                    current_tags.append(tag)
                elif tag in current_tags:
                    current_tags.remove(tag)
    
    # Column 2: Topics
    with cols[1]:
        st.markdown("##### Topics")
        for tag in TAGS["Topics"]:
            key = f"topic_{tag.replace(' ', '_')}"
            if st.checkbox(tag, value=tag in current_tags, key=key):
                if tag not in current_tags:
                    current_tags.append(tag)
                elif tag in current_tags:
                    current_tags.remove(tag)
    
    # Column 3-4: Cryptocurrencies (split into two columns due to length)
    crypto_categories = list(TAGS["Cryptocurrencies"].items())
    mid = len(crypto_categories) // 2
    
    with cols[2]:
        st.markdown("##### Cryptocurrencies (1/2)")
        for category, coins in crypto_categories[:mid]:
            st.markdown(f"###### {category}")
            for tag in coins:
                key = f"crypto_{category}_{tag.replace(' ', '_')}"
                if st.checkbox(tag, value=tag in current_tags, key=key):
                    if tag not in current_tags:
                        current_tags.append(tag)
                    elif tag in current_tags:
                        current_tags.remove(tag)
    
    with cols[3]:
        st.markdown("##### Cryptocurrencies (2/2)")
        for category, coins in crypto_categories[mid:]:
            st.markdown(f"###### {category}")
            for tag in coins:
                key = f"crypto_{category}_{tag.replace(' ', '_')}"
                if st.checkbox(tag, value=tag in current_tags, key=key):
                    if tag not in current_tags:
                        current_tags.append(tag)
                    elif tag in current_tags:
                        current_tags.remove(tag)
    
    # Column 5: Market Infrastructure - Exchanges & Data
    with cols[4]:
        st.markdown("##### Market Infrastructure (1/2)")
        for category in ["Exchanges", "Data & Analytics"]:
            st.markdown(f"###### {category}")
            for tag in TAGS["Market Infrastructure"][category]:
                key = f"infra_{category}_{tag.replace(' ', '_').replace('.', '_')}"
                if st.checkbox(tag, value=tag in current_tags, key=key):
                    if tag not in current_tags:
                        current_tags.append(tag)
                    elif tag in current_tags:
                        current_tags.remove(tag)
    
    # Column 6: Market Infrastructure - Wallets & Others
    with cols[5]:
        st.markdown("##### Market Infrastructure (2/2)")
        for category in ["Wallets & Custody"]:
            st.markdown(f"###### {category}")
            for tag in TAGS["Market Infrastructure"][category]:
                key = f"infra_{category}_{tag.replace(' ', '_').replace('.', '_')}"
                if st.checkbox(tag, value=tag in current_tags, key=key):
                    if tag not in current_tags:
                        current_tags.append(tag)
                    elif tag in current_tags:
                        current_tags.remove(tag)
    
    # Update session state
    st.session_state['current_tags'] = current_tags

async def translate_text(text: str, is_title: bool = False, is_h1: bool = False) -> str:
    """
    Translate text from English to Thai using OpenRouter's translation service.
    """
    if is_title:
        system_prompt = """You are a Thai crypto news translator. Translate this title to Thai.
Rules:
- Maximum 60 Thai characters
- Keep cryptocurrency names in English
- Maintain key message
- Return ONLY the translated text, no explanations
- If contain names such as people, company, or coin names, DO NOT translate names but ensure to translate the rest
- Do not add quotes or formatting"""
    elif is_h1:
        system_prompt = """You are a Thai crypto news translator. Translate this H1 to Thai.
Rules:
- Match the original English length as closely as possible
- Keep cryptocurrency names in English
- Maintain full meaning
- Return ONLY the translated text, no explanations
- If contain names such as people, company, or coin names, DO NOT translate names but ensure to translate the rest
- Do not add quotes or formatting"""
    else:
        system_prompt = """You are a Thai crypto news translator. Translate this text to Thai.
Rules:
- Use natural Thai language
- Write for Thai audiences with basic crypto knowledge, using semi-professional Thai language.
- Use Thai crypto terms where appropriate.
- Maintain the original meaning while making it natural in Thai.
- Return ONLY the translated text, no explanations
- If contain names such as people, company, or coin names, DO NOT translate names but ensure to translate the rest
- Do not add explanations nor comments. Focus on translating the content. Return only the translated content.
- Do not add quotes nor formatting"""

    try:
        response = await client.post(
            "/chat/completions",
            json={
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        translated_text = data['choices'][0]['message']['content'].strip()
        
        # Additional check for title length
        if is_title and len(translated_text) > 70:
            # Retry with stronger emphasis on length
            return await translate_text(f"Translate this title in under 70 Thai characters: {text}", is_title=True)
            
        return translated_text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

async def translate_section(section: Dict[str, str]) -> Dict[str, str]:
    """
    Translate a single section's heading and content.
    """
    translated_heading = await translate_text(section.get('heading', ''))
    translated_content = await translate_text(section.get('content', ''))
    return {
        'heading': translated_heading,
        'content': translated_content
    }

def determine_primary_category(crypto: str) -> str:
    """
    Determine the primary category based on the cryptocurrency.
    """
    if crypto.lower() == 'bitcoin':
        return 'Bitcoin News'
    elif crypto.lower() == 'ethereum':
        return 'Ethereum News'
    else:
        return 'Altcoin News'

async def translate_content(structured_content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate each element of the structured content and categorize with tags.
    """
    translated = {}
    translated['title'] = await translate_text(structured_content.get('title', ''), is_title=True)
    translated['h1'] = await translate_text(structured_content.get('h1', ''), is_h1=True)
    translated['meta_description'] = await translate_text(structured_content.get('meta_description', ''))
    
    # Translate sections concurrently
    sections = structured_content.get('sections', [])
    translated_sections = []
    for section in sections:
        chunks = section.get('chunks', [])
        translated_chunks = await asyncio.gather(
            *[
                translate_section(chunk)
                for chunk in chunks
            ]
        )
        translated_sections.append({
            'heading': section.get('heading', ''),
            'chunks': translated_chunks
        })
    translated['sections'] = translated_sections
    
    # Combine all content for categorization
    combined_content = f"{structured_content.get('title', '')} {structured_content.get('meta_description', '')} " \
                       + ' '.join([
                           chunk['content'] 
                           for section in structured_content.get('sections', []) 
                           for chunk in section.get('chunks', [])
                       ])
    
    # Get tags from LLM
    tags = await categorize_with_llm(combined_content)
    
    # Add mandatory tags based on content analysis
    mandatory_tags = []
    content_lower = combined_content.lower()
    
    # Check for major cryptocurrencies
    if 'bitcoin' in content_lower or ' btc ' in content_lower:
        mandatory_tags.extend(['Bitcoin News', 'BTC'])
    if 'ethereum' in content_lower or ' eth ' in content_lower:
        mandatory_tags.extend(['Ethereum News', 'ETH'])
    if 'solana' in content_lower or ' sol ' in content_lower:
        mandatory_tags.extend(['Altcoin News', 'SOL'])
    
    # Check for specific entities
    for category, entities in TAGS["Market Infrastructure"].items():
        for entity in entities:
            if entity.lower() in content_lower:
                if entity not in mandatory_tags:
                    mandatory_tags.append(entity)
    
    # Remove duplicates while preserving order
    seen = set()
    all_tags = []
    for tag in (mandatory_tags + tags):
        if tag not in seen and tag in ALL_TAGS:
            seen.add(tag)
            all_tags.append(tag)
    
    translated['tags'] = all_tags[:5]  # Keep top 5 tags
    return translated

def create_docx(original: Dict[str, Any], translated: Dict[str, Any]) -> BytesIO:
    """
    Create a .docx document with side-by-side comparison of original and translated content.
    """
    doc = Document()
    
    # Title
    doc.add_heading('Crypto News Translation Comparison', 0)
    
    # Tags section at the top
    current_tags = st.session_state.get('current_tags', [])
    doc.add_paragraph(f"Tags: {', '.join(current_tags)}")
    doc.add_paragraph(f"Original URL: {original.get('url', 'N/A')}")
    
    # Title Comparison
    doc.add_heading('Title', level=1)
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Light List Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'English'
    hdr_cells[1].text = 'Thai'

    row = table.add_row().cells
    row[0].text = original.get('title', '')
    row[1].text = translated.get('title', '')

    # Meta Description Comparison
    doc.add_heading('Meta Description', level=1)
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Light List Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'English'
    hdr_cells[1].text = 'Thai'

    row = table.add_row().cells
    row[0].text = original.get('meta_description', '')
    row[1].text = translated.get('meta_description', '')

    # Published Time Comparison
    doc.add_heading('Published Time', level=1)
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Light List Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'English'
    hdr_cells[1].text = 'Thai'

    row = table.add_row().cells
    row[0].text = original.get('published_time', '')
    row[1].text = translated.get('published_time', '')  # Assuming time doesn't need translation

    # Main Content Comparison
    doc.add_heading('Main Content', level=1)
    for i, (orig_section, trans_section) in enumerate(zip(
        original.get('sections', []), 
        translated.get('sections', [])
    )):
        doc.add_heading(f"Section {i+1}: {orig_section['heading']}", level=2)
        for j, (orig_chunk, trans_chunk) in enumerate(zip(
            orig_section.get('chunks', []), 
            trans_section.get('chunks', [])
        )):
            table = doc.add_table(rows=1, cols=2)
            table.style = 'Light List Accent 1'
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'English'
            hdr_cells[1].text = 'Thai'

            row = table.add_row().cells
            row[0].text = orig_chunk['content']
            row[1].text = trans_chunk['content']
            
            doc.add_paragraph()  # Blank line between chunks

    # Save to in-memory file
    doc_io = BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)
    return doc_io

def display_comparison(original: Dict[str, Any], translated: Dict[str, Any]):
    """
    Display side-by-side comparison with editable text areas.
    """
    st.markdown("## Content Comparison")

    # Original URL and Tags
    st.text(f"Original URL: {original.get('url', '')}")
    st.text(f"Tags: {', '.join(translated.get('tags', [])) if translated.get('tags') else 'N/A'}")
    st.markdown("---")

    # Title, H1 & Meta Description
    with st.expander("Title, H1 & Meta Description", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("English")
            st.text_area("Title", value=original.get('title', ''), height=100, key="en_title")
            st.text_area("H1", value=original.get('h1', ''), height=100, key="en_h1")
            st.text_area("Meta Description", value=original.get('meta_description', 'Not provided'), height=100, key="en_meta", disabled=True)
        with col2:
            st.subheader("Thai")
            st.text_area("Title", value=translated.get('title', ''), height=100, key="th_title")
            st.text_area("H1", value=translated.get('h1', ''), height=100, key="th_h1")
            st.text_area("Meta Description ภาษาไทย", value=translated.get('meta_description', ''), height=100, key="th_meta")

    # Main Content Sections
    with st.expander("Main Content", expanded=True):
        for i, (orig_section, trans_section) in enumerate(zip(
            original.get('sections', []), 
            translated.get('sections', [])
        )):
            st.markdown(f"### Section {i+1}: {orig_section['heading']}")
            for j, (orig_chunk, trans_chunk) in enumerate(zip(
                orig_section.get('chunks', []), 
                trans_section.get('chunks', [])
            )):
                st.markdown(f"#### Chunk {j+1}")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area(
                        "English",
                        value=orig_chunk['content'],
                        height=200,
                        key=f"en_section_{i}_chunk_{j}"
                    )
                with col2:
                    st.text_area(
                        "Thai",
                        value=trans_chunk['content'],
                        height=200,
                        key=f"th_section_{i}_chunk_{j}"
                    )
                st.markdown("---")  # Separator between chunks

    # Tags Section
    with st.expander("Tags", expanded=True):
        current_tags = st.session_state.get('current_tags', translated.get('tags', []))
        display_tag_section(current_tags)

async def process_and_translate(original_url: str, original_article: str):
    """
    Orchestrate the parsing, translation, and categorization of the article.
    """
    # Parse the article
    structured = parse_article(original_article)

    if structured.get('title') and structured.get('meta_description') and structured.get('sections'):
        # Add URL to structured content
        structured['url'] = original_url.strip()

        # Translate the content
        translated = await translate_content(structured)

        # Store in session state
        st.session_state['original'] = structured
        st.session_state['translated'] = translated

        st.success("Content processed and translated successfully.")
    else:
        st.error("Failed to parse the article. Please ensure the format is correct.")

def main():
    try:
        st.set_page_config(layout="wide")
        st.title("Crypto News Translator (EN → TH)")
        
        # Input Section with default values
        st.markdown("### Paste the Original English Article URL and Content Below")

        default_url = "https://cryptonews.com/news/solana-price-three-year-high-bitcoin-record-high-donald-trump-election/"
        default_content = """Solana Hits 3-Year Peak as Bitcoin's Record High Fuels Post-Trump Crypto Rally

The week's gains pushed SOL into the elite group of cryptocurrencies with a market cap over $100b.

Solana's cryptocurrency SOL reached a three-year high on Sunday as Bitcoin's record-breaking surge fueled a broad crypto rally after Donald Trump's decisive election win.

SOL surged to $214 early Sunday before settling at $209.88 by 10:38 pm ET. Meanwhile, Bitcoin reached a new record, surpassing $81,000. This spike came in response to Trump's election victory, sparking investor expectations of a more lenient regulatory environment. BTC has now more than doubled since its yearly low of $38,505 in January.

The week's gains made SOL join the elite group of cryptocurrencies, boasting a market cap over $100b. Despite having a much shorter history in the market, it now stands shoulder to shoulder with Bitcoin, Ethereum's ether and Tether (USDT).

Solana's Validator Earnings Surge Past $30M, Fueling Value Growth Amid Network Upgrades
Further, Solana's value has surged due to a significant rise in validator earnings, now surpassing $30m daily. This growth is driven by recent improvements in the network's transaction processing and reward systems.

Solana last reached the $214 level in Dec. 2021, after peaking at around $260 the previous month and starting to decline. The crypto experienced a steep drop in early 2022, followed by another decline that spring as the crypto market cooled.

The situation deteriorated further with the FTX collapse in Nov. 2022, which impacted Solana significantly due to its connections with the exchange and its founder, Sam Bankman-Fried.

Mixed Future Predictions
Solana is known for its frequent downtime, stemming from its emphasis on fast transaction processing and scalability over network robustness. Its unique design, which includes using Proof of History for efficient time-stamping and transaction sequencing, enables high-speed performance but has sometimes caused operational issues.

While some are optimistic about Solana's future price trends, not everyone agrees. Analyst Benjamin Cowen, for example, has shown doubt about Solana's momentum compared to Bitcoin as 2024 ends.

Cowen predicts the Solana-to-Bitcoin exchange rate could decline in November and December, with recovery likely only early next year. This cautious view contrasts with the generally positive forecasts for Solana's USD performance"""

        original_url = st.text_input(
            label="Original English Article URL",
            value=default_url,
            help="Paste the URL of the original English article here."
        )

        original_article = st.text_area(
            label="Original English Article Content",
            value=default_content,
            height=300,
            key="original_article",
            help="Paste the full English article content here."
        )

        # Process Button
        if st.button("Process and Translate"):
            if not original_url.strip():
                st.warning("Please enter the Original English Article URL.")
            elif not original_article.strip():
                st.warning("Please paste the Original English Article content.")
            else:
                with st.spinner("Processing and translating..."):
                    asyncio.run(process_and_translate(original_url, original_article))

        # Display Comparison if available
        if 'original' in st.session_state and 'translated' in st.session_state:
            original = st.session_state['original']
            translated = st.session_state['translated']
            
            # Initialize current_tags in session state if not present
            if 'current_tags' not in st.session_state:
                st.session_state['current_tags'] = translated.get('tags', [])

            # Display content comparison
            display_comparison(original, translated)

            # Download Button
            if st.session_state['translated']:
                with st.spinner("Generating document..."):
                    # Get the latest edited content
                    edited_translated = {
                        'title': st.session_state.get('th_title', translated.get('title', '')),
                        'meta_description': st.session_state.get('th_meta', translated.get('meta_description', '')),
                        'published_time': translated.get('published_time', ''),
                        'sections': []
                    }
                    
                    # Get edited sections
                    for i, section in enumerate(translated.get('sections', [])):
                        translated_chunks = []
                        for j, chunk in enumerate(section.get('chunks', [])):
                            edited_chunk = {
                                'heading': section.get('heading', ''),
                                'content': st.session_state.get(f"th_section_{i}_chunk_{j}", chunk.get('content', ''))
                            }
                            translated_chunks.append(edited_chunk)
                        edited_translated['sections'].append({
                            'heading': section.get('heading', ''),
                            'chunks': translated_chunks
                        })
                    
                    # Use current tags from session state
                    edited_translated['tags'] = st.session_state.get('current_tags', [])
                    
                    # Create document with latest content and tags
                    doc = create_docx(original, edited_translated)
                    
                st.download_button(
                    label="📥 Download Translated Document",
                    data=doc,
                    file_name="Crypto_News_Translation.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
