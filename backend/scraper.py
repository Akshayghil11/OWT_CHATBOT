import os
import re
import time
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Configuration
URL = "https://www.oneworldtechnologies.com/"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "owt-new-project")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_PAGES = 800  # High enough to cover the full site and portfolio
CRAWL_DELAY = 0.3  # seconds between requests to be polite

# Standard browser User-Agent to avoid being blocked
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# File extensions to skip (non-HTML resources)
SKIP_EXTENSIONS = {
    '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp',
    '.mp4', '.mp3', '.zip', '.css', '.js', '.ico', '.woff',
    '.woff2', '.ttf', '.eot',
}


def normalize_url(url):
    """Normalize a URL by removing fragments, query params, and trailing slashes."""
    parsed = urlparse(url)
    # Ensure path ends with / for consistency
    path = parsed.path.rstrip('/') + '/' if parsed.path != '/' else '/'
    normalized = urlunparse((
        parsed.scheme,
        parsed.netloc.lower(),
        path,
        '',  # params
        '',  # query
        '',  # fragment
    ))
    return normalized


def should_skip_url(url):
    """Check if a URL should be skipped based on file extension or pattern."""
    parsed = urlparse(url)
    path_lower = parsed.path.lower()

    # Skip non-HTML file extensions
    for ext in SKIP_EXTENSIONS:
        if path_lower.endswith(ext):
            return True

    # Skip WordPress admin, feed, login, and unnecessary URLs
    skip_patterns = [
        '/wp-admin', '/wp-login', '/feed/', '/wp-json/',
        '/xmlrpc', '/wp-content/', '/wp-includes/',
        '/tag/', '/author/', '/attachment/',
        '/comment-page-', '?replytocom=', '/trackback/',
    ]
    for pattern in skip_patterns:
        if pattern in path_lower:
            return True

    return False


def normalize_text(text):
    """Clean up extracted text by collapsing whitespace."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_pinecone_index(pc, index_name):
    available_indexes = pc.list_indexes().names()
    if index_name not in available_indexes:
        print(f"Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,  # Dimension for all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")
    return index_name


def fetch_page(url, retries=2):
    """Fetch a page with retry logic and proper headers."""
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/{retries} for {url} (waiting {wait}s): {e}")
                time.sleep(wait)
            else:
                print(f"  Failed to fetch {url} after {retries + 1} attempts: {e}")
                return None


def extract_text(html_content):
    """Extract clean text from HTML content, removing boilerplate elements."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove non-content tags
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()

    # Remove known boilerplate sections (cookie banners, popups, etc.)
    for selector in ['.cky-consent-container', '.pum-container', '#mega-menu-wrap-primary']:
        for el in soup.select(selector):
            el.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    
    # Try to extract main content area first (Elementor / WordPress main content)
    main_content = ""
    main_el = soup.find('main') or soup.find(id='content') or soup.find(class_='site-content')
    if main_el:
        main_content = normalize_text(main_el.get_text(separator=" ", strip=True))
    
    # Fall back to full body text if main content is too short
    if len(main_content) < 100:
        body_text = normalize_text(soup.get_text(separator=" ", strip=True))
        combined = " ".join(part for part in [title, body_text] if part)
    else:
        combined = " ".join(part for part in [title, main_content] if part)
    
    return combined


def extract_links(html_content, base_url):
    """Extract all same-domain links from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    base_domain = urlparse(base_url).netloc.lower()
    links = set()

    for anchor in soup.find_all('a', href=True):
        href = anchor['href'].strip()
        if href.startswith(('mailto:', 'tel:', '#', 'javascript:')):
            continue

        absolute_url = urljoin(base_url, href)
        parsed = urlparse(absolute_url)

        # Only follow same-domain links (ignoring www. prefix)
        if parsed.netloc.lower().replace("www.", "") != base_domain.replace("www.", ""):
            continue

        # Skip non-HTML resources
        if should_skip_url(absolute_url):
            continue

        normalized = normalize_url(absolute_url)
        links.add(normalized)

    return links


def discover_sitemap_urls(base_url):
    """
    Discover all page URLs from the website's XML sitemap.
    This is the most reliable way to find all pages on a WordPress site.
    """
    urls = set()
    base_domain = urlparse(base_url).netloc.lower()

    # Common sitemap locations for WordPress / Yoast SEO
    sitemap_urls = [
        f"{base_url}sitemap_index.xml",
        f"{base_url}sitemap.xml",
        f"{base_url}wp-sitemap.xml",
        f"{base_url}portfolio-sitemap.xml",
        f"{base_url}owt-portfolio-sitemap.xml",
    ]

    def parse_sitemap(sitemap_url, depth=0):
        """Recursively parse sitemaps (handles sitemap index files)."""
        if depth > 3:  # Prevent infinite recursion
            return
        try:
            resp = fetch_page(sitemap_url)
            if resp is None:
                return

            # Parse XML
            try:
                root = ET.fromstring(resp.content)
            except ET.ParseError:
                print(f"  Could not parse XML from {sitemap_url}")
                return

            # Handle namespace
            ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

            # Check if this is a sitemap index (contains <sitemap> elements)
            for sitemap_el in root.findall('.//sm:sitemap/sm:loc', ns):
                child_sitemap_url = sitemap_el.text.strip()
                print(f"  Found sub-sitemap: {child_sitemap_url}")
                parse_sitemap(child_sitemap_url, depth + 1)
                time.sleep(CRAWL_DELAY)

            # Extract <url><loc> entries
            for url_el in root.findall('.//sm:url/sm:loc', ns):
                page_url = url_el.text.strip()
                parsed = urlparse(page_url)
                if parsed.netloc.lower().replace("www.", "") == base_domain.replace("www.", "") and not should_skip_url(page_url):
                    urls.add(normalize_url(page_url))

        except Exception as e:
            print(f"  Error processing sitemap {sitemap_url}: {e}")

    print("Discovering pages from XML sitemaps...")
    for sitemap_url in sitemap_urls:
        parse_sitemap(sitemap_url)

    print(f"  Found {len(urls)} URL(s) from sitemaps.")
    return urls


def crawl_website(start_url, max_pages=MAX_PAGES):
    """
    Hybrid discovery: use XML sitemaps first, then BFS crawl to discover
    any pages not listed in sitemaps (e.g., service sub-pages, portfolio items).
    """
    # Phase 1: Discover URLs from sitemaps
    sitemap_urls = discover_sitemap_urls(start_url)

    # Phase 2: Seed the BFS queue with sitemap URLs + start URL
    start_normalized = normalize_url(start_url)
    to_visit = [start_normalized]
    # Add sitemap URLs to the queue
    for url in sitemap_urls:
        if url not in to_visit:
            to_visit.append(url)

    visited = set()
    pages = []

    print(f"\nStarting hybrid crawl (sitemap + BFS) with {len(to_visit)} seed URLs (max {max_pages} pages)...")

    while to_visit and len(pages) < max_pages:
        current_url = to_visit.pop(0)

        if current_url in visited:
            continue
        visited.add(current_url)

        response = fetch_page(current_url)
        if response is None:
            continue

        html = response.text
        text = extract_text(html)

        if text and len(text) > 50:  # Skip pages with negligible content
            pages.append({"url": current_url, "text": text})
            print(f"  [{len(pages)}/{max_pages}] Scraped {current_url} ({len(text)} chars)")

        # Discover new links from this page (BFS expansion)
        new_links = extract_links(html, current_url)
        for link in new_links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)

        # Rate limiting — be polite to the server
        time.sleep(CRAWL_DELAY)

    print(f"\nCrawl complete. Scraped {len(pages)} page(s), visited {len(visited)} URL(s), "
          f"{len(to_visit)} URL(s) remaining in queue.")
    return pages


def ingest_data():
    if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here":
        raise ValueError("PINECONE_API_KEY is not set correctly in .env file.")

    # Use hybrid sitemap + BFS crawl
    scraped_pages = crawl_website(URL, max_pages=MAX_PAGES)

    if not scraped_pages:
        raise RuntimeError("No content extracted from the website. Aborting ingestion.")

    print("\nSplitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    docs = []
    for page in scraped_pages:
        chunks = text_splitter.split_text(page["text"])
        for chunk_index, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": page["url"], "chunk": chunk_index},
                )
            )

    if not docs:
        raise RuntimeError("Text splitting produced no chunks. Aborting ingestion.")

    print(f"Created {len(docs)} chunks from {len(scraped_pages)} pages.")

    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    resolved_index_name = ensure_pinecone_index(pc, PINECONE_INDEX_NAME)

    # Clear old vectors before re-ingesting to avoid stale data
    print("Clearing old vectors from Pinecone index...")
    index = pc.Index(resolved_index_name)
    try:
        index.delete(delete_all=True)
        print("  Old vectors deleted.")
    except Exception as e:
        print(f"  Could not clear index (may be empty): {e}")

    print("Uploading chunks to Pinecone...")
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=resolved_index_name,
        embedding=embeddings,
    )
    
    # Upload in batches to avoid timeouts
    batch_size = 50
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  Uploaded batch {i // batch_size + 1}/{(len(docs) + batch_size - 1) // batch_size} "
              f"({len(batch)} chunks)")

    # Wait a moment for index to update stats
    time.sleep(2)
    index_stats = index.describe_index_stats()
    total_vectors = index_stats.get("total_vector_count", "unknown")
    print(f"\nPinecone index '{resolved_index_name}' now has {total_vectors} vector(s).")

    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_data()
