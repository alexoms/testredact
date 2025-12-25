#!/usr/bin/env python3
"""
PDF Unredaction REST API Service

A FastAPI-based REST service for testing and unredacting PDF files.
"""

import tempfile
import os
import zipfile
import shutil
import re
import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import aiohttp
import requests
import cloudscraper
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from scrapingbee import ScrapingBeeClient

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pdf_unredact import unredact_pdf, check_metadata, analyze_pdf_structure, analyze_folder, save_to_mongodb

try:
    from pymongo import MongoClient
    from gridfs import GridFS
except ImportError:
    MongoClient = None
    GridFS = None

mongo_client = None
grid_fs = None

# Job tracking for async operations
jobs = {}


def get_brightdata_proxy():
    """Get Bright Data proxy configuration."""
    customer_id = os.environ.get("BRIGHTDATA_CUSTOMER_ID")
    zone = os.environ.get("BRIGHTDATA_ZONE", "unblocker")
    api_key = os.environ.get("BRIGHTDATA_API_KEY")

    if not customer_id or not api_key:
        return None

    # Bright Data proxy format
    # Format: brd-customer-<customer_id>-zone-<zone>
    port = os.environ.get("BRIGHTDATA_PORT", "33335")
    proxy_url = f"http://brd-customer-{customer_id}-zone-{zone}:{api_key}@brd.superproxy.io:{port}"
    return proxy_url


async def scrape_with_brightdata(url: str) -> tuple:
    """
    Scrape a URL using Bright Data Web Unlocker.
    Returns (html_content, error_message)
    """
    proxy_url = get_brightdata_proxy()
    if not proxy_url:
        return None, "Bright Data not configured"

    try:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(
                url,
                proxy=proxy_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                timeout=aiohttp.ClientTimeout(total=60),
                ssl=False
            ) as resp:
                if resp.status == 200:
                    content = await resp.text()
                    if len(content) > 1000:
                        return content, None
                    return None, f"Response too short: {len(content)} bytes"
                return None, f"HTTP {resp.status}"
    except Exception as e:
        return None, f"Bright Data error: {str(e)}"


def scrape_with_brightdata_sync(url: str) -> tuple:
    """
    Synchronous version of Bright Data scraping using requests.
    Returns (html_content, error_message)
    """
    proxy_url = get_brightdata_proxy()
    if not proxy_url:
        return None, "Bright Data not configured"

    try:
        proxies = {
            "http": proxy_url,
            "https": proxy_url
        }
        resp = requests.get(
            url,
            proxies=proxies,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=60,
            verify=False
        )
        if resp.status_code == 200 and len(resp.text) > 1000:
            return resp.text, None
        return None, f"HTTP {resp.status_code}, length: {len(resp.text)}"
    except Exception as e:
        return None, f"Bright Data error: {str(e)}"


def scrape_with_wget(url: str) -> tuple:
    """Scrape a URL using wget. Returns (html_content, error_message)."""
    import subprocess

    try:
        result = subprocess.run(
            ["wget", "-q", "-O", "-", "--timeout=60", url],
            capture_output=True,
            timeout=70,
            text=True
        )

        if result.returncode == 0 and len(result.stdout) > 1000:
            return result.stdout, None
        return None, f"wget failed: exit code {result.returncode}"

    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


def download_pdf_sync(url: str, proxy_url: str = None) -> tuple:
    """Download a single PDF using wget subprocess."""
    import subprocess

    try:
        # Create temp file for download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name

        # Use wget for download
        result = subprocess.run(
            ["wget", "-q", "-O", tmp_path, "--timeout=120", url],
            capture_output=True,
            timeout=130
        )

        if result.returncode == 0:
            with open(tmp_path, "rb") as f:
                content = f.read()
            os.unlink(tmp_path)

            if len(content) >= 100 and content[:4] == b"%PDF":
                return url, content, None
            return url, None, "Not a valid PDF"

        os.unlink(tmp_path) if os.path.exists(tmp_path) else None
        return url, None, f"wget failed: exit code {result.returncode}"

    except subprocess.TimeoutExpired:
        os.unlink(tmp_path) if os.path.exists(tmp_path) else None
        return url, None, "Timeout"
    except Exception as e:
        return url, None, str(e)


async def download_pdf_async(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore, proxy_url: str = None) -> tuple:
    """Download a single PDF asynchronously - uses sync requests in thread pool for reliability."""
    async with semaphore:
        # Run synchronous download in thread pool for better compatibility
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, download_pdf_sync, url, proxy_url)


async def download_pdfs_concurrent(urls: list, max_concurrent: int = 10, use_proxy: bool = True) -> list:
    """Download multiple PDFs concurrently with rate limiting."""
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=5)

    # Get proxy URL if available
    proxy_url = get_brightdata_proxy() if use_proxy else None

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_pdf_async(session, url, semaphore, proxy_url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return results


def run_background_analysis(job_id: str, file_ids: list, persist: bool = True):
    """Background task to analyze PDFs - runs synchronously in thread pool."""
    job = jobs[job_id]
    job["status"] = "analyzing"
    job["analyzed"] = 0
    job["vulnerable_count"] = 0
    job["errors"] = []

    from bson import ObjectId

    for file_id in file_ids:
        try:
            grid_file = grid_fs.get(ObjectId(file_id))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(grid_file.read())
                tmp_path = tmp.name

            try:
                result = unredact_pdf(tmp_path)
                is_vulnerable = len(result.get("potential_fake_redactions", [])) > 0

                if is_vulnerable:
                    job["vulnerable_count"] += 1

                if persist and mongo_client:
                    db = mongo_client["pdf_analysis"]
                    # Get source_url from GridFS file metadata
                    source_url = getattr(grid_file, 'source_url', None)
                    source_page = getattr(grid_file, 'source_page', None)
                    db["results"].insert_one({
                        "filename": grid_file.filename,
                        "file_id": file_id,
                        "folder": "async_download",
                        "job_id": job_id,
                        "source_url": source_url,
                        "source_page": source_page,
                        "analyzed_at": datetime.now().isoformat(),
                        "is_vulnerable": is_vulnerable,
                        "findings": result.get("findings", []),
                        "potential_fake_redactions": result.get("potential_fake_redactions", []),
                        "metadata": result.get("metadata", {}),
                        "structure": result.get("structure", {}),
                        "errors": result.get("errors", [])
                    })

                job["analyzed"] += 1
            finally:
                os.unlink(tmp_path)

        except Exception as e:
            job["errors"].append(f"Error analyzing {file_id}: {str(e)}")

    job["status"] = "completed"
    job["completed_at"] = datetime.now().isoformat()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mongo_client, grid_fs
    mongo_uri = os.environ.get("MONGO_URI")
    if mongo_uri and MongoClient:
        mongo_client = MongoClient(mongo_uri)
        db = mongo_client["pdf_analysis"]
        grid_fs = GridFS(db)
        print(f"Connected to MongoDB at {mongo_uri}")
    yield
    if mongo_client:
        mongo_client.close()

app = FastAPI(
    title="PDF Unredaction API",
    description="API for testing and recovering improperly redacted content from PDFs",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic Models
class FakeRedaction(BaseModel):
    """Represents a detected fake redaction with recoverable text."""
    page: int = Field(..., description="Page number where the fake redaction was found")
    rect: str = Field(..., description="Rectangle coordinates of the redaction")
    hidden_text: str = Field(..., description="Text recovered from under the redaction")
    type: str = Field(..., description="Type of fake redaction detected")


class RedactionAnnotation(BaseModel):
    """Represents a redaction annotation in the PDF."""
    page: int = Field(..., description="Page number")
    rect: str = Field(..., description="Rectangle coordinates")
    type: str = Field(..., description="Annotation type")
    applied: bool = Field(..., description="Whether the redaction has been applied")


class PDFMetadata(BaseModel):
    """PDF document metadata."""
    title: str = ""
    author: str = ""
    subject: str = ""
    keywords: str = ""
    creator: str = ""
    producer: str = ""
    creationDate: str = ""
    modDate: str = ""


class PDFStructure(BaseModel):
    """PDF document structure analysis."""
    page_count: int = Field(..., description="Number of pages")
    has_forms: bool = Field(..., description="Whether PDF contains forms")
    is_encrypted: bool = Field(..., description="Whether PDF is encrypted")
    is_repaired: bool = Field(..., description="Whether PDF was repaired during parsing")
    embedded_files: list = Field(default_factory=list, description="List of embedded files")
    javascript: bool = Field(..., description="Whether PDF contains JavaScript")


class AnalysisResult(BaseModel):
    """Complete PDF analysis result."""
    file: str = Field(..., description="Analyzed file name")
    success: bool = Field(..., description="Whether analysis completed successfully")
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")
    findings: list[str] = Field(default_factory=list, description="Security findings summary")
    extracted_text: dict[str, str] = Field(default_factory=dict, description="Extracted text by page")
    redactions_found: list[RedactionAnnotation] = Field(default_factory=list)
    potential_fake_redactions: list[FakeRedaction] = Field(default_factory=list)
    metadata: PDFMetadata = Field(default_factory=PDFMetadata)
    structure: PDFStructure = Field(default=None)
    is_vulnerable: bool = Field(..., description="Whether vulnerable redactions were detected")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    service: str = "pdf-unredaction-api"
    version: str = "1.0.0"


class FolderAnalysisRequest(BaseModel):
    """Request for folder analysis."""
    folder_path: str = Field(..., description="Path to folder containing PDF files")
    persist: bool = Field(True, description="Whether to persist results to MongoDB")


class FolderAnalysisResult(BaseModel):
    """Result of folder analysis."""
    folder: str = Field(..., description="Analyzed folder path")
    analyzed_at: str = Field(..., description="Timestamp of analysis")
    total_files: int = Field(..., description="Total number of PDF files analyzed")
    vulnerable_count: int = Field(..., description="Number of vulnerable PDFs found")
    results: list[AnalysisResult] = Field(default_factory=list, description="Individual file results")


class StoredResult(BaseModel):
    """A single stored analysis result from MongoDB."""
    id: str = Field(..., description="MongoDB document ID")
    filename: str
    folder: str
    analyzed_at: str
    is_vulnerable: bool
    findings: list[str] = []
    potential_fake_redactions: list[dict] = []
    metadata: dict = {}
    structure: dict = {}
    errors: list[str] = []
    source_url: Optional[str] = Field(None, description="Original PDF download URL")
    file_id: Optional[str] = Field(None, description="GridFS file ID for PDF retrieval")


class StoredResultsResponse(BaseModel):
    """Response containing stored results from MongoDB."""
    total: int = Field(..., description="Total matching results in database")
    page: int = Field(1, description="Current page number")
    per_page: int = Field(20, description="Results per page")
    pages: int = Field(1, description="Total number of pages")
    results: list[StoredResult] = Field(default_factory=list)


class UploadedFile(BaseModel):
    """Info about an uploaded PDF file."""
    filename: str
    file_id: str
    size: int


class UploadResponse(BaseModel):
    """Response from bulk upload."""
    uploaded: int = Field(..., description="Number of files uploaded")
    files: list[UploadedFile] = Field(default_factory=list)


class AnalyzeUploadedRequest(BaseModel):
    """Request to analyze uploaded PDFs."""
    file_ids: list[str] = Field(default=None, description="Specific file IDs to analyze (None = all)")
    persist: bool = Field(True, description="Persist results to MongoDB")


class AnalyzeUploadedResponse(BaseModel):
    """Response from analyzing uploaded PDFs."""
    analyzed: int
    vulnerable_count: int
    results: list[AnalysisResult]


class ZipUploadResponse(BaseModel):
    """Response from ZIP file upload and analysis."""
    zip_filename: str
    total_pdfs_found: int
    analyzed: int
    vulnerable_count: int
    results: list[AnalysisResult] = Field(default_factory=list)


class ScrapeUrlRequest(BaseModel):
    """Request to scrape PDFs from a URL."""
    url: str = Field(..., description="URL of webpage to scrape for PDF links")
    analyze: bool = Field(True, description="Whether to analyze PDFs immediately")
    persist: bool = Field(True, description="Whether to persist results to MongoDB")
    max_pdfs: int = Field(100, description="Maximum number of PDFs to download")
    pdf_urls: list[str] = Field(default=None, description="Optional: Direct PDF URLs to download instead of scraping")


class ScrapeUrlListRequest(BaseModel):
    """Request to scrape and store PDF URL list only."""
    url: str = Field(..., description="URL of webpage to scrape for PDF links")
    name: str = Field(..., description="Name for this URL list (e.g., 'epstein-court-records')")
    use_brightdata: bool = Field(False, description="Use Bright Data proxy (default: False, uses wget)")


class UrlListResponse(BaseModel):
    """Response with stored URL list."""
    name: str
    source_url: str
    total_urls: int
    urls: list[str] = Field(default_factory=list)


class DownloadFromListRequest(BaseModel):
    """Request to download and analyze PDFs from a stored URL list."""
    name: str = Field(..., description="Name of the stored URL list")
    analyze: bool = Field(True, description="Whether to analyze PDFs")
    persist: bool = Field(True, description="Whether to persist results to MongoDB")
    max_pdfs: int = Field(100, description="Maximum number of PDFs to download")
    offset: int = Field(0, description="Start from this index in the URL list")


class ScrapedPdf(BaseModel):
    """Info about a scraped PDF."""
    filename: str
    url: str
    file_id: str
    size: int


class ScrapeUrlResponse(BaseModel):
    """Response from URL scraping."""
    source_url: str
    pdfs_found: int
    pdfs_downloaded: int
    analyzed: int
    vulnerable_count: int
    pdfs: list[ScrapedPdf] = Field(default_factory=list)
    results: list[AnalysisResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AsyncDownloadRequest(BaseModel):
    """Request for async PDF download."""
    name: str = Field(..., description="Name of the stored URL list to download from")
    max_concurrent: int = Field(10, description="Max concurrent downloads (1-50)")
    max_pdfs: int = Field(1000, description="Maximum PDFs to download")
    offset: int = Field(0, description="Start from this index in the URL list")
    analyze_after: bool = Field(True, description="Trigger analysis after download completes")
    persist: bool = Field(True, description="Persist analysis results to MongoDB")


class AsyncDownloadResponse(BaseModel):
    """Response from async download initiation."""
    job_id: str
    status: str
    message: str
    total_urls: int


class JobStatusResponse(BaseModel):
    """Status of an async job."""
    job_id: str
    status: str
    phase: str = ""
    downloaded: int = 0
    failed: int = 0
    analyzed: int = 0
    vulnerable_count: int = 0
    total: int = 0
    errors: list[str] = Field(default_factory=list)
    file_ids: list[str] = Field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - returns service info."""
    return HealthResponse()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@app.get("/results", response_model=StoredResultsResponse)
async def get_results(
    vulnerable_only: bool = False,
    filename: Optional[str] = None,
    page: int = 1,
    per_page: int = 20
):
    """
    Retrieve stored analysis results from MongoDB with pagination.

    - **vulnerable_only**: Filter to only show vulnerable files
    - **filename**: Filter by filename (partial match)
    - **page**: Page number (default: 1)
    - **per_page**: Results per page (default: 20, max: 100)
    """
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not configured")

    db = mongo_client["pdf_analysis"]
    collection = db["results"]

    query = {}
    if vulnerable_only:
        query["is_vulnerable"] = True
    if filename:
        query["filename"] = {"$regex": filename, "$options": "i"}

    # Get total count
    total = collection.count_documents(query)

    # Pagination
    per_page = min(per_page, 100)  # Max 100 per page
    page = max(page, 1)
    skip = (page - 1) * per_page
    pages = (total + per_page - 1) // per_page if total > 0 else 1

    cursor = collection.find(query).sort("analyzed_at", -1).skip(skip).limit(per_page)
    results = []
    for doc in cursor:
        results.append(StoredResult(
            id=str(doc["_id"]),
            filename=doc.get("filename", ""),
            folder=doc.get("folder", ""),
            analyzed_at=doc.get("analyzed_at", ""),
            is_vulnerable=doc.get("is_vulnerable", False),
            findings=doc.get("findings", []),
            potential_fake_redactions=doc.get("potential_fake_redactions", []),
            metadata=doc.get("metadata", {}),
            structure=doc.get("structure", {}),
            errors=doc.get("errors", []),
            source_url=doc.get("source_url"),
            file_id=doc.get("file_id")
        ))

    return StoredResultsResponse(total=total, page=page, per_page=per_page, pages=pages, results=results)


@app.post("/scrape-urls", response_model=UrlListResponse)
async def scrape_url_list(request: ScrapeUrlListRequest):
    """
    Scrape a webpage for PDF links and store the URL list in MongoDB.
    Uses wget by default. Set use_brightdata=true to use Bright Data proxy.
    """
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not configured")

    html_content = None
    errors = []

    if request.use_brightdata:
        # Use Bright Data proxy
        brightdata_key = os.environ.get("BRIGHTDATA_API_KEY")
        if brightdata_key:
            html_content, err = await scrape_with_brightdata(request.url)
            if err:
                errors.append(f"Bright Data: {err}")
        else:
            errors.append("Bright Data not configured")
    else:
        # Default: use wget
        html_content, err = scrape_with_wget(request.url)
        if err:
            errors.append(f"wget: {err}")

    if not html_content:
        raise HTTPException(status_code=400, detail=f"Failed to fetch page. Errors: {'; '.join(errors)}")

    # Parse and find PDF links
    soup = BeautifulSoup(html_content, "html.parser")
    pdf_urls = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.lower().endswith(".pdf"):
            full_url = urljoin(request.url, href)
            pdf_urls.append(full_url)

    # Store in MongoDB
    db = mongo_client["pdf_analysis"]
    db["url_lists"].update_one(
        {"name": request.name},
        {"$set": {
            "name": request.name,
            "source_url": request.url,
            "urls": pdf_urls,
            "total_urls": len(pdf_urls),
            "created_at": datetime.now().isoformat()
        }},
        upsert=True
    )

    return UrlListResponse(
        name=request.name,
        source_url=request.url,
        total_urls=len(pdf_urls),
        urls=pdf_urls[:20]  # Return first 20 as preview
    )


@app.get("/url-lists")
async def list_url_lists():
    """List all stored URL lists."""
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not configured")

    db = mongo_client["pdf_analysis"]
    lists = []
    for doc in db["url_lists"].find():
        lists.append({
            "name": doc["name"],
            "source_url": doc["source_url"],
            "total_urls": doc["total_urls"],
            "created_at": doc.get("created_at")
        })
    return {"lists": lists}


@app.get("/url-lists/{name}")
async def get_url_list(name: str):
    """Get a stored URL list by name."""
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not configured")

    db = mongo_client["pdf_analysis"]
    doc = db["url_lists"].find_one({"name": name})
    if not doc:
        raise HTTPException(status_code=404, detail=f"URL list '{name}' not found")

    return UrlListResponse(
        name=doc["name"],
        source_url=doc["source_url"],
        total_urls=doc["total_urls"],
        urls=doc["urls"]
    )


@app.post("/download-async", response_model=AsyncDownloadResponse)
async def download_async(request: AsyncDownloadRequest, background_tasks: BackgroundTasks):
    """
    Start async concurrent download of PDFs from a stored URL list.

    Downloads happen concurrently with configurable parallelism.
    Analysis runs in the background after downloads complete.

    Returns immediately with a job_id to track progress via /jobs/{job_id}
    """
    if not mongo_client or not grid_fs:
        raise HTTPException(status_code=503, detail="MongoDB not configured")

    # Get stored URL list
    db = mongo_client["pdf_analysis"]
    doc = db["url_lists"].find_one({"name": request.name})
    if not doc:
        raise HTTPException(status_code=404, detail=f"URL list '{request.name}' not found")

    pdf_urls = doc["urls"][request.offset:request.offset + request.max_pdfs]
    if not pdf_urls:
        raise HTTPException(status_code=400, detail="No URLs to download")

    # Create job
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "downloading",
        "phase": "download",
        "total": len(pdf_urls),
        "downloaded": 0,
        "failed": 0,
        "analyzed": 0,
        "vulnerable_count": 0,
        "errors": [],
        "file_ids": [],
        "source_url": doc["source_url"],
        "started_at": datetime.now().isoformat(),
        "analyze_after": request.analyze_after,
        "persist": request.persist,
    }

    # Start download in background
    async def do_downloads():
        job = jobs[job_id]
        source_url = doc["source_url"]
        source_domain = urlparse(source_url).netloc

        # Limit concurrent downloads
        max_concurrent = min(max(1, request.max_concurrent), 50)
        semaphore = asyncio.Semaphore(max_concurrent)
        proxy_url = get_brightdata_proxy()

        async def download_and_store(url):
            """Download a single PDF and store in GridFS, updating job progress."""
            async with semaphore:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, download_pdf_sync, url, proxy_url)

                url, content, error = result
                if error:
                    job["errors"].append(f"{url}: {error}")
                    job["failed"] += 1
                    return

                # Store in GridFS
                try:
                    parsed = urlparse(url)
                    filename = os.path.basename(parsed.path)
                    if not filename.endswith(".pdf"):
                        filename = filename + ".pdf"
                    filename = unquote(filename)

                    file_id = grid_fs.put(
                        content,
                        filename=filename,
                        source_url=url,
                        source_page=source_url,
                        source_domain=source_domain,
                        job_id=job_id
                    )
                    job["file_ids"].append(str(file_id))
                    job["downloaded"] += 1
                except Exception as e:
                    job["errors"].append(f"GridFS error for {url}: {str(e)}")
                    job["failed"] += 1

        # Download all PDFs concurrently with real-time progress updates
        tasks = [download_and_store(url) for url in pdf_urls]
        await asyncio.gather(*tasks, return_exceptions=True)

        job["phase"] = "download_complete"

        # Trigger background analysis if requested
        if request.analyze_after and job["file_ids"]:
            import threading
            thread = threading.Thread(
                target=run_background_analysis,
                args=(job_id, job["file_ids"], request.persist)
            )
            thread.start()
        else:
            job["status"] = "completed"
            job["completed_at"] = datetime.now().isoformat()

    # Run downloads as async task
    asyncio.create_task(do_downloads())

    return AsyncDownloadResponse(
        job_id=job_id,
        status="started",
        message=f"Download started for {len(pdf_urls)} PDFs with max {request.max_concurrent} concurrent connections",
        total_urls=len(pdf_urls)
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of an async download/analysis job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        phase=job.get("phase", ""),
        downloaded=job.get("downloaded", 0),
        failed=job.get("failed", 0),
        analyzed=job.get("analyzed", 0),
        vulnerable_count=job.get("vulnerable_count", 0),
        total=job.get("total", 0),
        errors=job.get("errors", [])[:20],  # Limit errors in response
        file_ids=job.get("file_ids", [])[:50],  # Limit file_ids in response
        started_at=job.get("started_at", ""),
        completed_at=job.get("completed_at", "")
    )


@app.get("/jobs")
async def list_jobs():
    """List all async jobs."""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job.get("status"),
                "phase": job.get("phase"),
                "downloaded": job.get("downloaded", 0),
                "analyzed": job.get("analyzed", 0),
                "total": job.get("total", 0),
                "started_at": job.get("started_at")
            }
            for job_id, job in jobs.items()
        ]
    }


@app.post("/download-from-list", response_model=ScrapeUrlResponse)
async def download_from_list(request: DownloadFromListRequest):
    """
    Download and analyze PDFs from a stored URL list.
    Downloads directly without ScrapingBee.
    """
    if not mongo_client or not grid_fs:
        raise HTTPException(status_code=503, detail="MongoDB not configured")

    # Get stored URL list
    db = mongo_client["pdf_analysis"]
    doc = db["url_lists"].find_one({"name": request.name})
    if not doc:
        raise HTTPException(status_code=404, detail=f"URL list '{request.name}' not found")

    pdf_urls = doc["urls"][request.offset:request.offset + request.max_pdfs]
    source_url = doc["source_url"]
    source_domain = urlparse(source_url).netloc

    errors = []
    scraped_pdfs = []
    analysis_results = []
    vulnerable_count = 0

    from urllib.parse import unquote

    for pdf_url in pdf_urls:
        try:
            # Extract filename
            parsed = urlparse(pdf_url)
            filename = os.path.basename(parsed.path)
            if not filename.endswith(".pdf"):
                filename = filename + ".pdf"
            filename = unquote(filename)

            # Direct download (no ScrapingBee needed)
            resp = requests.get(pdf_url, timeout=60, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })

            if resp.status_code != 200:
                errors.append(f"HTTP {resp.status_code}: {pdf_url}")
                continue

            pdf_content = resp.content

            if len(pdf_content) < 100 or pdf_content[:4] != b"%PDF":
                errors.append(f"Not a valid PDF: {pdf_url}")
                continue

            # Store in GridFS
            file_id = grid_fs.put(
                pdf_content,
                filename=filename,
                source_url=pdf_url,
                source_page=source_url,
                source_domain=source_domain
            )

            scraped_pdfs.append(ScrapedPdf(
                filename=filename,
                url=pdf_url,
                file_id=str(file_id),
                size=len(pdf_content)
            ))

            # Analyze if requested
            if request.analyze:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_content)
                    tmp_path = tmp.name

                try:
                    result = unredact_pdf(tmp_path)
                    result["file"] = filename

                    is_vulnerable = len(result.get("potential_fake_redactions", [])) > 0
                    if is_vulnerable:
                        vulnerable_count += 1

                    if request.persist:
                        db["results"].insert_one({
                            "filename": filename,
                            "file_id": str(file_id),
                            "folder": source_domain,
                            "source_url": pdf_url,
                            "source_page": source_url,
                            "analyzed_at": datetime.now().isoformat(),
                            "is_vulnerable": is_vulnerable,
                            "findings": result.get("findings", []),
                            "potential_fake_redactions": result.get("potential_fake_redactions", []),
                            "metadata": result.get("metadata", {}),
                            "structure": result.get("structure", {}),
                            "errors": result.get("errors", [])
                        })

                    # Build response model
                    redactions = [
                        RedactionAnnotation(page=r["page"], rect=str(r["rect"]), type=r["type"], applied=r.get("applied", False))
                        for r in result.get("redactions_found", [])
                    ]
                    fake_redactions = [
                        FakeRedaction(page=f["page"], rect=f["rect"], hidden_text=f["hidden_text"], type=f["type"])
                        for f in result.get("potential_fake_redactions", [])
                    ]
                    metadata = PDFMetadata(**result.get("metadata", {}))
                    structure_data = result.get("structure", {})
                    structure = PDFStructure(
                        page_count=structure_data.get("page_count", 0),
                        has_forms=structure_data.get("has_forms", False),
                        is_encrypted=structure_data.get("is_encrypted", False),
                        is_repaired=structure_data.get("is_repaired", False),
                        embedded_files=structure_data.get("embedded_files", []),
                        javascript=structure_data.get("javascript", False)
                    )

                    analysis_results.append(AnalysisResult(
                        file=filename,
                        success=result.get("success", False),
                        errors=result.get("errors", []),
                        findings=result.get("findings", []),
                        extracted_text={str(k): v for k, v in result.get("extracted_text", {}).items()},
                        redactions_found=redactions,
                        potential_fake_redactions=fake_redactions,
                        metadata=metadata,
                        structure=structure,
                        is_vulnerable=is_vulnerable
                    ))
                finally:
                    os.unlink(tmp_path)

        except Exception as e:
            errors.append(f"Error downloading {pdf_url}: {str(e)}")

    return ScrapeUrlResponse(
        source_url=source_url,
        pdfs_found=len(pdf_urls),
        pdfs_downloaded=len(scraped_pdfs),
        analyzed=len(analysis_results),
        vulnerable_count=vulnerable_count,
        pdfs=scraped_pdfs,
        results=analysis_results,
        errors=errors
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: list[UploadFile] = File(..., description="PDF files to upload")):
    """
    Upload multiple PDF files to MongoDB for later analysis.

    Files are stored in GridFS and can be analyzed using /analyze-uploaded.
    """
    if not grid_fs:
        raise HTTPException(status_code=503, detail="MongoDB/GridFS not configured")

    uploaded_files = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue

        content = await file.read()
        file_id = grid_fs.put(content, filename=file.filename)

        uploaded_files.append(UploadedFile(
            filename=file.filename,
            file_id=str(file_id),
            size=len(content)
        ))

    return UploadResponse(uploaded=len(uploaded_files), files=uploaded_files)


@app.post("/upload-zip", response_model=ZipUploadResponse)
async def upload_zip(
    file: UploadFile = File(..., description="ZIP file containing PDFs"),
    analyze: bool = True,
    persist: bool = True
):
    """
    Upload a ZIP file containing PDF files in folders.

    - Extracts the ZIP and recursively finds all PDF files
    - Stores each PDF in GridFS with its relative path
    - Optionally analyzes all PDFs immediately
    - Persists results to MongoDB if persist=true

    Parameters:
    - **file**: ZIP file to upload
    - **analyze**: Whether to analyze PDFs immediately (default: true)
    - **persist**: Whether to persist results to MongoDB (default: true)
    """
    if not grid_fs:
        raise HTTPException(status_code=503, detail="MongoDB/GridFS not configured")

    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    # Save ZIP to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        content = await file.read()
        tmp_zip.write(content)
        zip_path = tmp_zip.name

    # Create temp directory for extraction
    extract_dir = tempfile.mkdtemp()

    try:
        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Recursively find all PDFs
        pdf_files = list(Path(extract_dir).rglob("*.pdf"))

        analysis_results = []
        vulnerable_count = 0

        for pdf_path in pdf_files:
            # Get relative path from extract dir for folder info
            rel_path = pdf_path.relative_to(extract_dir)
            folder_path = str(rel_path.parent) if rel_path.parent != Path(".") else "/"

            # Store in GridFS
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
                file_id = grid_fs.put(
                    pdf_content,
                    filename=pdf_path.name,
                    folder=folder_path,
                    zip_source=file.filename
                )

            # Analyze if requested
            if analyze:
                result = unredact_pdf(str(pdf_path))
                result["file"] = pdf_path.name

                is_vulnerable = len(result.get("potential_fake_redactions", [])) > 0
                if is_vulnerable:
                    vulnerable_count += 1

                # Persist to results collection
                if persist:
                    db = mongo_client["pdf_analysis"]
                    db["results"].insert_one({
                        "filename": pdf_path.name,
                        "file_id": str(file_id),
                        "folder": folder_path,
                        "zip_source": file.filename,
                        "analyzed_at": datetime.now().isoformat(),
                        "is_vulnerable": is_vulnerable,
                        "findings": result.get("findings", []),
                        "potential_fake_redactions": result.get("potential_fake_redactions", []),
                        "metadata": result.get("metadata", {}),
                        "structure": result.get("structure", {}),
                        "errors": result.get("errors", [])
                    })

                # Convert to Pydantic model
                redactions = [
                    RedactionAnnotation(
                        page=r["page"],
                        rect=str(r["rect"]),
                        type=r["type"],
                        applied=r.get("applied", False)
                    )
                    for r in result.get("redactions_found", [])
                ]

                fake_redactions = [
                    FakeRedaction(
                        page=f["page"],
                        rect=f["rect"],
                        hidden_text=f["hidden_text"],
                        type=f["type"]
                    )
                    for f in result.get("potential_fake_redactions", [])
                ]

                metadata = PDFMetadata(**result.get("metadata", {}))
                structure_data = result.get("structure", {})
                structure = PDFStructure(
                    page_count=structure_data.get("page_count", 0),
                    has_forms=structure_data.get("has_forms", False),
                    is_encrypted=structure_data.get("is_encrypted", False),
                    is_repaired=structure_data.get("is_repaired", False),
                    embedded_files=structure_data.get("embedded_files", []),
                    javascript=structure_data.get("javascript", False)
                )

                analysis_results.append(AnalysisResult(
                    file=f"{folder_path}/{pdf_path.name}" if folder_path != "/" else pdf_path.name,
                    success=result.get("success", False),
                    errors=result.get("errors", []),
                    findings=result.get("findings", []),
                    extracted_text={str(k): v for k, v in result.get("extracted_text", {}).items()},
                    redactions_found=redactions,
                    potential_fake_redactions=fake_redactions,
                    metadata=metadata,
                    structure=structure,
                    is_vulnerable=is_vulnerable
                ))

        return ZipUploadResponse(
            zip_filename=file.filename,
            total_pdfs_found=len(pdf_files),
            analyzed=len(analysis_results) if analyze else 0,
            vulnerable_count=vulnerable_count,
            results=analysis_results
        )

    finally:
        # Cleanup
        os.unlink(zip_path)
        shutil.rmtree(extract_dir)


@app.post("/scrape-url", response_model=ScrapeUrlResponse)
async def scrape_url(request: ScrapeUrlRequest):
    """
    Scrape a webpage for PDF links, download them to MongoDB, and analyze.

    - Fetches the webpage and finds all PDF links
    - Downloads each PDF and stores in GridFS
    - Optionally analyzes all PDFs
    - Persists results to MongoDB if persist=true

    Example: Scrape court records from justice.gov
    """
    if not grid_fs:
        raise HTTPException(status_code=503, detail="MongoDB/GridFS not configured")

    errors = []
    scraped_pdfs = []
    analysis_results = []
    vulnerable_count = 0
    source_domain = urlparse(request.url).netloc

    # Initialize ScrapingBee client if API key is available
    scrapingbee_key = os.environ.get("SCRAPINGBEE_API_KEY")
    sb_client = ScrapingBeeClient(api_key=scrapingbee_key) if scrapingbee_key else None

    # If direct PDF URLs are provided, use those instead of scraping
    if request.pdf_urls:
        pdf_links = request.pdf_urls[:request.max_pdfs]
    else:
        html_content = None

        # Use ScrapingBee as primary method (best for Cloudflare sites)
        if sb_client:
            try:
                response = sb_client.get(
                    request.url,
                    params={
                        'render_js': 'true',
                        'stealth_proxy': 'true',
                        'block_resources': 'false'
                    }
                )
                if response.status_code == 200:
                    html_content = response.text
            except Exception as e:
                errors.append(f"ScrapingBee error: {str(e)}")

        # Fallback to cloudscraper
        if not html_content:
            scraper = cloudscraper.create_scraper(
                browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
            )
            try:
                response = scraper.get(request.url, timeout=30, allow_redirects=True)
                if response.status_code == 200 and len(response.text) > 5000:
                    html_content = response.text
            except Exception:
                pass

        if not html_content:
            raise HTTPException(status_code=400, detail="Failed to retrieve page content")

        # Parse HTML and find PDF links
        soup = BeautifulSoup(html_content, "html.parser")
        pdf_links = set()

        # Find all links ending in .pdf
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.lower().endswith(".pdf"):
                full_url = urljoin(request.url, href)
                pdf_links.add(full_url)

        # Also check for links with pdf in query params or path
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if ".pdf" in href.lower() or "pdf" in href.lower():
                full_url = urljoin(request.url, href)
                if full_url not in pdf_links:
                    if re.search(r'\.pdf($|\?|#)', href.lower()):
                        pdf_links.add(full_url)

        pdf_links = list(pdf_links)[:request.max_pdfs]

    # Download each PDF - try direct download first to save ScrapingBee credits
    from urllib.parse import unquote

    for pdf_url in pdf_links:
        try:
            # Extract filename from URL
            parsed = urlparse(pdf_url)
            filename = os.path.basename(parsed.path)
            if not filename.endswith(".pdf"):
                filename = filename + ".pdf"
            filename = unquote(filename)

            pdf_content = None

            # Try direct download first (saves ScrapingBee API credits)
            try:
                resp = requests.get(pdf_url, timeout=60, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "application/pdf,*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                })
                if resp.status_code == 200 and len(resp.content) > 100 and resp.content[:4] == b"%PDF":
                    pdf_content = resp.content
            except Exception:
                pass

            # Fallback to ScrapingBee only if direct download failed
            if pdf_content is None and sb_client:
                try:
                    response = sb_client.get(
                        pdf_url,
                        params={
                            'stealth_proxy': 'true'
                        }
                    )
                    if response.status_code == 200 and len(response.content) > 100:
                        pdf_content = response.content
                except Exception as e:
                    errors.append(f"ScrapingBee PDF download error for {filename}: {str(e)}")

            if pdf_content is None or len(pdf_content) < 100:
                errors.append(f"Failed to download: {pdf_url}")
                continue

            # Verify it's actually a PDF
            if not pdf_content[:4] == b"%PDF":
                errors.append(f"Not a PDF: {pdf_url}")
                continue

            # Store in GridFS
            file_id = grid_fs.put(
                pdf_content,
                filename=filename,
                source_url=pdf_url,
                source_page=request.url,
                source_domain=source_domain
            )

            scraped_pdfs.append(ScrapedPdf(
                filename=filename,
                url=pdf_url,
                file_id=str(file_id),
                size=len(pdf_content)
            ))

            # Analyze if requested
            if request.analyze:
                # Write to temp file for analysis
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_content)
                    tmp_path = tmp.name

                try:
                    result = unredact_pdf(tmp_path)
                    result["file"] = filename

                    is_vulnerable = len(result.get("potential_fake_redactions", [])) > 0
                    if is_vulnerable:
                        vulnerable_count += 1

                    # Persist to results collection
                    if request.persist:
                        db = mongo_client["pdf_analysis"]
                        db["results"].insert_one({
                            "filename": filename,
                            "file_id": str(file_id),
                            "folder": source_domain,
                            "source_url": pdf_url,
                            "source_page": request.url,
                            "analyzed_at": datetime.now().isoformat(),
                            "is_vulnerable": is_vulnerable,
                            "findings": result.get("findings", []),
                            "potential_fake_redactions": result.get("potential_fake_redactions", []),
                            "metadata": result.get("metadata", {}),
                            "structure": result.get("structure", {}),
                            "errors": result.get("errors", [])
                        })

                    # Convert to Pydantic model
                    redactions = [
                        RedactionAnnotation(
                            page=r["page"],
                            rect=str(r["rect"]),
                            type=r["type"],
                            applied=r.get("applied", False)
                        )
                        for r in result.get("redactions_found", [])
                    ]

                    fake_redactions = [
                        FakeRedaction(
                            page=f["page"],
                            rect=f["rect"],
                            hidden_text=f["hidden_text"],
                            type=f["type"]
                        )
                        for f in result.get("potential_fake_redactions", [])
                    ]

                    metadata = PDFMetadata(**result.get("metadata", {}))
                    structure_data = result.get("structure", {})
                    structure = PDFStructure(
                        page_count=structure_data.get("page_count", 0),
                        has_forms=structure_data.get("has_forms", False),
                        is_encrypted=structure_data.get("is_encrypted", False),
                        is_repaired=structure_data.get("is_repaired", False),
                        embedded_files=structure_data.get("embedded_files", []),
                        javascript=structure_data.get("javascript", False)
                    )

                    analysis_results.append(AnalysisResult(
                        file=filename,
                        success=result.get("success", False),
                        errors=result.get("errors", []),
                        findings=result.get("findings", []),
                        extracted_text={str(k): v for k, v in result.get("extracted_text", {}).items()},
                        redactions_found=redactions,
                        potential_fake_redactions=fake_redactions,
                        metadata=metadata,
                        structure=structure,
                        is_vulnerable=is_vulnerable
                    ))
                finally:
                    os.unlink(tmp_path)

        except Exception as e:
            errors.append(f"Failed to download {pdf_url}: {str(e)}")

    return ScrapeUrlResponse(
        source_url=request.url,
        pdfs_found=len(pdf_links),
        pdfs_downloaded=len(scraped_pdfs),
        analyzed=len(analysis_results),
        vulnerable_count=vulnerable_count,
        pdfs=scraped_pdfs,
        results=analysis_results,
        errors=errors
    )


@app.get("/uploads")
async def list_uploads():
    """List all uploaded PDF files in GridFS."""
    if not grid_fs:
        raise HTTPException(status_code=503, detail="MongoDB/GridFS not configured")

    files = []
    for f in grid_fs.find():
        files.append({
            "file_id": str(f._id),
            "filename": f.filename,
            "size": f.length,
            "upload_date": f.upload_date.isoformat()
        })

    return {"total": len(files), "files": files}


@app.post("/analyze-uploaded", response_model=AnalyzeUploadedResponse)
async def analyze_uploaded(request: AnalyzeUploadedRequest = None):
    """
    Analyze PDF files that were previously uploaded to MongoDB.

    - If file_ids is provided, only those files are analyzed
    - If file_ids is None/empty, all uploaded files are analyzed
    - Results are persisted to MongoDB if persist=true
    """
    if not grid_fs:
        raise HTTPException(status_code=503, detail="MongoDB/GridFS not configured")

    # Get files to analyze
    if request and request.file_ids:
        from bson import ObjectId
        files_to_analyze = [grid_fs.get(ObjectId(fid)) for fid in request.file_ids]
    else:
        files_to_analyze = list(grid_fs.find())

    if not files_to_analyze:
        raise HTTPException(status_code=404, detail="No files found to analyze")

    analysis_results = []
    vulnerable_count = 0
    persist = request.persist if request else True

    for grid_file in files_to_analyze:
        # Write to temp file for analysis
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(grid_file.read())
            tmp_path = tmp.name

        try:
            result = unredact_pdf(tmp_path)
            result["file"] = grid_file.filename

            is_vulnerable = len(result.get("potential_fake_redactions", [])) > 0
            if is_vulnerable:
                vulnerable_count += 1

            # Persist to results collection
            if persist:
                db = mongo_client["pdf_analysis"]
                db["results"].insert_one({
                    "filename": grid_file.filename,
                    "file_id": str(grid_file._id),
                    "folder": "gridfs",
                    "analyzed_at": datetime.now().isoformat(),
                    "is_vulnerable": is_vulnerable,
                    "findings": result.get("findings", []),
                    "potential_fake_redactions": result.get("potential_fake_redactions", []),
                    "metadata": result.get("metadata", {}),
                    "structure": result.get("structure", {}),
                    "errors": result.get("errors", [])
                })

            # Convert to Pydantic model
            redactions = [
                RedactionAnnotation(
                    page=r["page"],
                    rect=str(r["rect"]),
                    type=r["type"],
                    applied=r.get("applied", False)
                )
                for r in result.get("redactions_found", [])
            ]

            fake_redactions = [
                FakeRedaction(
                    page=f["page"],
                    rect=f["rect"],
                    hidden_text=f["hidden_text"],
                    type=f["type"]
                )
                for f in result.get("potential_fake_redactions", [])
            ]

            metadata = PDFMetadata(**result.get("metadata", {}))
            structure_data = result.get("structure", {})
            structure = PDFStructure(
                page_count=structure_data.get("page_count", 0),
                has_forms=structure_data.get("has_forms", False),
                is_encrypted=structure_data.get("is_encrypted", False),
                is_repaired=structure_data.get("is_repaired", False),
                embedded_files=structure_data.get("embedded_files", []),
                javascript=structure_data.get("javascript", False)
            )

            analysis_results.append(AnalysisResult(
                file=grid_file.filename,
                success=result.get("success", False),
                errors=result.get("errors", []),
                findings=result.get("findings", []),
                extracted_text={str(k): v for k, v in result.get("extracted_text", {}).items()},
                redactions_found=redactions,
                potential_fake_redactions=fake_redactions,
                metadata=metadata,
                structure=structure,
                is_vulnerable=is_vulnerable
            ))
        finally:
            os.unlink(tmp_path)

    return AnalyzeUploadedResponse(
        analyzed=len(analysis_results),
        vulnerable_count=vulnerable_count,
        results=analysis_results
    )


@app.get("/uploads/{file_id}")
async def get_uploaded_pdf(file_id: str, highlight: bool = False):
    """Retrieve a PDF file from GridFS by file ID. Set highlight=true to add yellow highlights at redaction locations."""
    if not grid_fs:
        raise HTTPException(status_code=503, detail="MongoDB/GridFS not configured")

    from bson import ObjectId
    from fastapi.responses import StreamingResponse
    import io
    import fitz  # PyMuPDF

    try:
        grid_file = grid_fs.get(ObjectId(file_id))
        pdf_content = grid_file.read()

        if highlight:
            # Get the analysis results for this file to find redaction coordinates
            db = mongo_client["pdf_analysis"]
            result = db["results"].find_one({"file_id": file_id})

            if result and result.get("potential_fake_redactions"):
                # Open PDF and add highlights
                doc = fitz.open(stream=pdf_content, filetype="pdf")

                for idx, redaction in enumerate(result["potential_fake_redactions"]):
                    page_num = redaction.get("page", 1) - 1  # 0-indexed
                    rect_str = redaction.get("rect", "")
                    hidden_text = redaction.get("hidden_text", "")

                    # Parse rect string like "Rect(x0, y0, x1, y1)"
                    import re
                    match = re.search(r'Rect\(([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)', rect_str)
                    if match and 0 <= page_num < len(doc):
                        x0, y0, x1, y1 = map(float, match.groups())
                        rect = fitz.Rect(x0, y0, x1, y1)
                        page = doc[page_num]

                        # Expand rect for visibility
                        expanded_rect = rect + (-5, -5, 5, 5)

                        # Draw a bright yellow/red border around the fake redaction area
                        shape = page.new_shape()
                        shape.draw_rect(expanded_rect)
                        shape.finish(color=(1, 0, 0), width=3)  # Thick red border
                        shape.commit()

                        # Add a numbered circle marker
                        center_x = expanded_rect.x0 - 15
                        center_y = expanded_rect.y0 - 15
                        shape2 = page.new_shape()
                        shape2.draw_circle((center_x, center_y), 10)
                        shape2.finish(color=(1, 0, 0), fill=(1, 1, 0), width=2)
                        shape2.commit()

                        # Add number in circle
                        page.insert_text(
                            (center_x - 4, center_y + 4),
                            str(idx + 1),
                            fontsize=10,
                            color=(1, 0, 0)
                        )

                        # Add text annotation showing the hidden text
                        annot = page.add_text_annot(
                            (expanded_rect.x1, expanded_rect.y0),
                            f"HIDDEN TEXT FOUND:\n{hidden_text}"
                        )
                        annot.set_colors(stroke=(1, 0, 0))
                        annot.update()

                # Save to bytes
                pdf_content = doc.write()
                doc.close()

        return StreamingResponse(
            io.BytesIO(pdf_content),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{grid_file.filename}"',
                "Content-Length": str(len(pdf_content)),
                "X-Frame-Options": "SAMEORIGIN",
                "Content-Security-Policy": "frame-ancestors 'self'",
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}")


@app.delete("/uploads/{file_id}")
async def delete_upload(file_id: str):
    """Delete an uploaded file from GridFS."""
    if not grid_fs:
        raise HTTPException(status_code=503, detail="MongoDB/GridFS not configured")

    from bson import ObjectId
    try:
        grid_fs.delete(ObjectId(file_id))
        return {"deleted": file_id}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}")


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_pdf(file: UploadFile = File(..., description="PDF file to analyze")):
    """
    Analyze a PDF file for vulnerable redactions.

    Uploads a PDF and returns analysis including:
    - Detected fake redactions (black boxes over text)
    - Unapplied redaction annotations
    - Recovered hidden text
    - Document metadata and structure
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run analysis
        results = unredact_pdf(tmp_path)

        # Convert to Pydantic models
        redactions = [
            RedactionAnnotation(
                page=r["page"],
                rect=str(r["rect"]),
                type=r["type"],
                applied=r.get("applied", False)
            )
            for r in results.get("redactions_found", [])
        ]

        fake_redactions = [
            FakeRedaction(
                page=f["page"],
                rect=f["rect"],
                hidden_text=f["hidden_text"],
                type=f["type"]
            )
            for f in results.get("potential_fake_redactions", [])
        ]

        metadata = PDFMetadata(**results.get("metadata", {}))

        structure_data = results.get("structure", {})
        structure = PDFStructure(
            page_count=structure_data.get("page_count", 0),
            has_forms=structure_data.get("has_forms", False),
            is_encrypted=structure_data.get("is_encrypted", False),
            is_repaired=structure_data.get("is_repaired", False),
            embedded_files=structure_data.get("embedded_files", []),
            javascript=structure_data.get("javascript", False)
        )

        return AnalysisResult(
            file=file.filename,
            success=results.get("success", False),
            errors=results.get("errors", []),
            findings=results.get("findings", []),
            extracted_text={str(k): v for k, v in results.get("extracted_text", {}).items()},
            redactions_found=redactions,
            potential_fake_redactions=fake_redactions,
            metadata=metadata,
            structure=structure,
            is_vulnerable=len(fake_redactions) > 0
        )

    finally:
        # Cleanup temp file
        os.unlink(tmp_path)


@app.post("/unredact")
async def unredact_pdf_file(file: UploadFile = File(..., description="PDF file to unredact")):
    """
    Attempt to create an unredacted version of the PDF.

    Returns the unredacted PDF file if fake redactions were found,
    otherwise returns the original file with analysis results.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
        content = await file.read()
        tmp_in.write(content)
        input_path = tmp_in.name

    output_path = tempfile.mktemp(suffix="_unredacted.pdf")

    try:
        # Run unredaction
        results = unredact_pdf(input_path, output_path)

        if results.get("potential_fake_redactions") and os.path.exists(output_path):
            # Return unredacted PDF
            return FileResponse(
                output_path,
                media_type="application/pdf",
                filename=f"unredacted_{file.filename}",
                headers={
                    "X-Findings-Count": str(len(results.get("findings", []))),
                    "X-Is-Vulnerable": "true"
                }
            )
        else:
            # No fake redactions found, return analysis as JSON
            raise HTTPException(
                status_code=200,
                detail={
                    "message": "No fake redactions found to remove",
                    "findings": results.get("findings", []),
                    "is_vulnerable": False
                }
            )

    finally:
        # Cleanup input temp file (output cleaned up after response)
        os.unlink(input_path)


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(..., description="PDF file to extract text from")):
    """
    Extract all text from a PDF file.

    Returns all text content organized by page number.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        results = unredact_pdf(tmp_path)
        return {
            "file": file.filename,
            "page_count": results.get("structure", {}).get("page_count", 0),
            "text_by_page": {str(k): v for k, v in results.get("extracted_text", {}).items()}
        }
    finally:
        os.unlink(tmp_path)


@app.post("/analyze-folder", response_model=FolderAnalysisResult)
async def analyze_folder_endpoint(request: FolderAnalysisRequest):
    """
    Analyze all PDF files in a folder.

    Returns analysis results for all PDFs including:
    - Total files analyzed
    - Number of vulnerable files found
    - Individual results for each file
    - Optionally persists results to MongoDB
    """
    folder = Path(request.folder_path)
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Invalid directory: {request.folder_path}")

    mongo_uri = os.environ.get("MONGO_URI") if request.persist else None
    batch_results = analyze_folder(str(folder), mongo_uri=mongo_uri)

    if batch_results.get("error"):
        raise HTTPException(status_code=400, detail=batch_results["error"])

    # Convert results to Pydantic models
    analysis_results = []
    for result in batch_results.get("results", []):
        redactions = [
            RedactionAnnotation(
                page=r["page"],
                rect=str(r["rect"]),
                type=r["type"],
                applied=r.get("applied", False)
            )
            for r in result.get("redactions_found", [])
        ]

        fake_redactions = [
            FakeRedaction(
                page=f["page"],
                rect=f["rect"],
                hidden_text=f["hidden_text"],
                type=f["type"]
            )
            for f in result.get("potential_fake_redactions", [])
        ]

        metadata = PDFMetadata(**result.get("metadata", {}))

        structure_data = result.get("structure", {})
        structure = PDFStructure(
            page_count=structure_data.get("page_count", 0),
            has_forms=structure_data.get("has_forms", False),
            is_encrypted=structure_data.get("is_encrypted", False),
            is_repaired=structure_data.get("is_repaired", False),
            embedded_files=structure_data.get("embedded_files", []),
            javascript=structure_data.get("javascript", False)
        )

        analysis_results.append(AnalysisResult(
            file=Path(result["file"]).name,
            success=result.get("success", False),
            errors=result.get("errors", []),
            findings=result.get("findings", []),
            extracted_text={str(k): v for k, v in result.get("extracted_text", {}).items()},
            redactions_found=redactions,
            potential_fake_redactions=fake_redactions,
            metadata=metadata,
            structure=structure,
            is_vulnerable=result.get("is_vulnerable", False)
        ))

    return FolderAnalysisResult(
        folder=batch_results["folder"],
        analyzed_at=batch_results["analyzed_at"],
        total_files=batch_results["total_files"],
        vulnerable_count=batch_results["vulnerable_count"],
        results=analysis_results
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=16250)
