#!/usr/bin/env python3
"""
PDF Unredaction Tool - Test and recover improperly redacted content from PDFs.

This tool is intended for:
- Security testing of PDF redaction implementations
- Digital forensics and document analysis
- Educational purposes to understand PDF security
- CTF challenges

Usage:
    python pdf_unredact.py <pdf_file> [options]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF is required. Install with: pip install PyMuPDF")
    sys.exit(1)

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None


def extract_all_text(doc: fitz.Document) -> dict:
    """Extract all text from PDF, page by page."""
    text_by_page = {}
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_by_page[page_num + 1] = page.get_text("text")
    return text_by_page


def find_redaction_annotations(doc: fitz.Document) -> list:
    """Find all redaction annotations in the PDF."""
    redactions = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        for annot in page.annots() or []:
            if annot.type[0] == 12:  # Redact annotation type
                redactions.append({
                    "page": page_num + 1,
                    "rect": annot.rect,
                    "type": "redact_annotation",
                    "applied": annot.info.get("subject", "") == "Redact"
                })
    return redactions


def find_black_rectangles(doc: fitz.Document) -> list:
    """Find black rectangles that might be covering text (fake redactions)."""
    black_rects = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        drawings = page.get_drawings()

        for drawing in drawings:
            # Check for filled black rectangles
            fill = drawing.get("fill")
            is_black = fill in [(0, 0, 0), (0.0, 0.0, 0.0)]

            if is_black:
                # Check if any item in the drawing is a rectangle
                items = drawing.get("items", [])
                for item in items:
                    if item[0] == "re":  # Rectangle item
                        rect = drawing.get("rect")
                        if rect:
                            black_rects.append({
                                "page": page_num + 1,
                                "rect": fitz.Rect(rect),
                                "type": "black_rectangle"
                            })
                            break
    return black_rects


def extract_text_under_rect(page: fitz.Page, rect: fitz.Rect) -> str:
    """Extract text that falls within a given rectangle."""
    # Get all text with position information
    text_dict = page.get_text("dict")
    found_text = []

    for block in text_dict.get("blocks", []):
        if block.get("type") == 0:  # Text block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_rect = fitz.Rect(span["bbox"])
                    # Check if span intersects with the redaction rectangle
                    if rect.intersects(span_rect):
                        found_text.append(span["text"])

    return " ".join(found_text)


def extract_hidden_text_layers(doc: fitz.Document) -> dict:
    """Extract text from all layers, including hidden ones."""
    hidden_text = {}
    for page_num in range(len(doc)):
        page = doc[page_num]

        # Try different text extraction methods
        methods = {
            "text": page.get_text("text"),
            "rawdict": str(page.get_text("rawdict")),
            "xhtml": page.get_text("xhtml"),
            "xml": page.get_text("xml"),
        }

        hidden_text[page_num + 1] = methods

    return hidden_text


def check_metadata(doc: fitz.Document) -> dict:
    """Extract PDF metadata that might contain sensitive info."""
    metadata = doc.metadata
    return {
        "title": metadata.get("title", ""),
        "author": metadata.get("author", ""),
        "subject": metadata.get("subject", ""),
        "keywords": metadata.get("keywords", ""),
        "creator": metadata.get("creator", ""),
        "producer": metadata.get("producer", ""),
        "creationDate": metadata.get("creationDate", ""),
        "modDate": metadata.get("modDate", ""),
    }


def analyze_pdf_structure(doc: fitz.Document) -> dict:
    """Analyze PDF structure for hidden content."""
    analysis = {
        "page_count": len(doc),
        "has_forms": doc.is_form_pdf,
        "is_encrypted": doc.is_encrypted,
        "is_repaired": doc.is_repaired,
        "embedded_files": [],
        "javascript": False,
    }

    # Check for embedded files
    try:
        if doc.embfile_count() > 0:
            for i in range(doc.embfile_count()):
                info = doc.embfile_info(i)
                analysis["embedded_files"].append(info)
    except Exception:
        pass

    # Check for JavaScript
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            for annot in page.annots() or []:
                if annot.info.get("name") == "JavaScript":
                    analysis["javascript"] = True
                    break
    except Exception:
        pass

    return analysis


def unredact_pdf(input_path: str, output_path: str = None) -> dict:
    """
    Main function to analyze and attempt to unredact a PDF.

    Returns a dict with all findings.
    """
    results = {
        "file": input_path,
        "success": False,
        "errors": [],
        "findings": [],
        "extracted_text": {},
        "redactions_found": [],
        "potential_fake_redactions": [],
        "metadata": {},
        "structure": {},
    }

    try:
        doc = fitz.open(input_path)
    except Exception as e:
        results["errors"].append(f"Failed to open PDF: {e}")
        return results

    try:
        # Extract metadata
        results["metadata"] = check_metadata(doc)

        # Analyze structure
        results["structure"] = analyze_pdf_structure(doc)

        # Extract all visible text
        results["extracted_text"] = extract_all_text(doc)

        # Find redaction annotations
        redactions = find_redaction_annotations(doc)
        results["redactions_found"] = redactions

        # Find potential fake redactions (black rectangles)
        black_rects = find_black_rectangles(doc)

        # For each black rectangle, try to extract text underneath
        for rect_info in black_rects:
            page = doc[rect_info["page"] - 1]
            hidden_text = extract_text_under_rect(page, rect_info["rect"])
            if hidden_text.strip():
                results["potential_fake_redactions"].append({
                    "page": rect_info["page"],
                    "rect": str(rect_info["rect"]),
                    "hidden_text": hidden_text,
                    "type": "text_under_black_rectangle"
                })
                results["findings"].append(
                    f"VULNERABLE: Found text under black rectangle on page {rect_info['page']}: '{hidden_text}'"
                )

        # Check for text in redaction annotation areas
        for redact in redactions:
            page = doc[redact["page"] - 1]
            hidden_text = extract_text_under_rect(page, redact["rect"])
            if hidden_text.strip():
                results["potential_fake_redactions"].append({
                    "page": redact["page"],
                    "rect": str(redact["rect"]),
                    "hidden_text": hidden_text,
                    "type": "unapplied_redaction"
                })
                results["findings"].append(
                    f"VULNERABLE: Unapplied redaction on page {redact['page']}: '{hidden_text}'"
                )

        # If output path specified, create a version with redactions removed
        if output_path and results["potential_fake_redactions"]:
            try:
                # Remove black rectangles by re-rendering without them
                new_doc = fitz.open()
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
                    new_page.show_pdf_page(new_page.rect, doc, page_num)
                new_doc.save(output_path)
                new_doc.close()
                results["findings"].append(f"Created unredacted version: {output_path}")
            except Exception as e:
                results["errors"].append(f"Failed to create unredacted version: {e}")

        results["success"] = True

    except Exception as e:
        results["errors"].append(f"Analysis error: {e}")
    finally:
        doc.close()

    return results


def save_to_mongodb(batch_results: dict, mongo_uri: str):
    """Persist analysis results to MongoDB."""
    if MongoClient is None:
        print("Error: pymongo is required for MongoDB support. Install with: pip install pymongo")
        return

    client = MongoClient(mongo_uri)
    db = client["pdf_analysis"]
    collection = db["results"]

    for result in batch_results["results"]:
        doc = {
            "filename": Path(result["file"]).name,
            "folder": batch_results["folder"],
            "analyzed_at": batch_results["analyzed_at"],
            "is_vulnerable": result.get("is_vulnerable", False),
            "findings": result.get("findings", []),
            "potential_fake_redactions": result.get("potential_fake_redactions", []),
            "metadata": result.get("metadata", {}),
            "structure": result.get("structure", {}),
            "errors": result.get("errors", [])
        }
        collection.insert_one(doc)

    client.close()
    print(f"Saved {len(batch_results['results'])} results to MongoDB")


def analyze_folder(folder_path: str, output_log: str = None, mongo_uri: str = None) -> dict:
    """Analyze all PDF files in a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return {"error": f"Not a valid directory: {folder_path}"}

    pdf_files = list(folder.glob("*.pdf"))

    batch_results = {
        "folder": str(folder.resolve()),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "total_files": len(pdf_files),
        "vulnerable_count": 0,
        "results": []
    }

    print(f"\nAnalyzing {len(pdf_files)} PDF files in {folder}...")

    for pdf_file in pdf_files:
        print(f"  Processing: {pdf_file.name}")
        result = unredact_pdf(str(pdf_file))
        result["analyzed_at"] = datetime.now(timezone.utc).isoformat()
        result["is_vulnerable"] = len(result.get("potential_fake_redactions", [])) > 0

        if result["is_vulnerable"]:
            batch_results["vulnerable_count"] += 1
        batch_results["results"].append(result)

    # Write JSON log
    if output_log:
        with open(output_log, "w") as f:
            json.dump(batch_results, f, indent=2, default=str)
        print(f"\nLog written to: {output_log}")

    # Persist to MongoDB
    if mongo_uri:
        save_to_mongodb(batch_results, mongo_uri)

    return batch_results


def print_folder_results(batch_results: dict):
    """Print summary of folder analysis results."""
    print("\n" + "=" * 60)
    print(f"Folder Analysis Summary: {batch_results['folder']}")
    print("=" * 60)
    print(f"  Total files analyzed: {batch_results['total_files']}")
    print(f"  Vulnerable files: {batch_results['vulnerable_count']}")
    print(f"  Analysis time: {batch_results['analyzed_at']}")

    if batch_results['vulnerable_count'] > 0:
        print("\n[VULNERABLE FILES]")
        for result in batch_results["results"]:
            if result.get("is_vulnerable"):
                print(f"  - {Path(result['file']).name}")
                for finding in result.get("findings", []):
                    print(f"      {finding}")

    print("\n" + "=" * 60)


def print_results(results: dict, verbose: bool = False):
    """Print analysis results in a readable format."""
    print("\n" + "=" * 60)
    print(f"PDF Unredaction Analysis: {results['file']}")
    print("=" * 60)

    if results["errors"]:
        print("\n[ERRORS]")
        for error in results["errors"]:
            print(f"  - {error}")

    # Structure info
    print("\n[DOCUMENT INFO]")
    struct = results["structure"]
    print(f"  Pages: {struct.get('page_count', 'N/A')}")
    print(f"  Encrypted: {struct.get('is_encrypted', 'N/A')}")
    print(f"  Has Forms: {struct.get('has_forms', 'N/A')}")
    if struct.get("embedded_files"):
        print(f"  Embedded Files: {len(struct['embedded_files'])}")

    # Metadata
    if any(results["metadata"].values()):
        print("\n[METADATA]")
        for key, value in results["metadata"].items():
            if value:
                print(f"  {key}: {value}")

    # Findings
    if results["findings"]:
        print("\n[SECURITY FINDINGS]")
        for finding in results["findings"]:
            print(f"  {finding}")
    else:
        print("\n[SECURITY FINDINGS]")
        print("  No vulnerable redactions detected.")

    # Redactions found
    if results["redactions_found"]:
        print(f"\n[REDACTION ANNOTATIONS]: {len(results['redactions_found'])} found")
        for redact in results["redactions_found"]:
            print(f"  - Page {redact['page']}: {redact['rect']}")

    # Potential fake redactions
    if results["potential_fake_redactions"]:
        print(f"\n[RECOVERED TEXT FROM FAKE REDACTIONS]")
        for fake in results["potential_fake_redactions"]:
            print(f"  Page {fake['page']} ({fake['type']}):")
            print(f"    Text: {fake['hidden_text']}")

    # Full text extraction (verbose mode)
    if verbose:
        print("\n[FULL TEXT EXTRACTION]")
        for page_num, text in results["extracted_text"].items():
            print(f"\n  --- Page {page_num} ---")
            print(f"  {text[:500]}..." if len(text) > 500 else f"  {text}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="PDF Unredaction Tool - Test and recover improperly redacted content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf                    # Analyze a PDF for vulnerable redactions
  %(prog)s document.pdf -o unredacted.pdf  # Save unredacted version
  %(prog)s document.pdf -v                 # Verbose output with full text
  %(prog)s document.pdf --json             # Output results as JSON
  %(prog)s --folder /path/to/pdfs          # Analyze all PDFs in folder
  %(prog)s --folder /path/to/pdfs --log results.json  # Save log file
  %(prog)s --folder /path/to/pdfs --mongo mongodb://localhost:28017  # Persist to MongoDB
        """
    )

    parser.add_argument("pdf_file", nargs="?", help="Path to the PDF file to analyze")
    parser.add_argument("-o", "--output", help="Path to save unredacted PDF")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--folder", help="Analyze all PDF files in folder")
    parser.add_argument("--log", help="Path to save JSON log file (folder mode)")
    parser.add_argument("--mongo", help="MongoDB URI for persisting results")

    args = parser.parse_args()

    # Folder analysis mode
    if args.folder:
        if not Path(args.folder).is_dir():
            print(f"Error: Not a valid directory: {args.folder}")
            sys.exit(1)

        batch_results = analyze_folder(args.folder, args.log, args.mongo)

        if args.json:
            print(json.dumps(batch_results, indent=2, default=str))
        else:
            print_folder_results(batch_results)

        # Exit code based on findings
        if batch_results.get("vulnerable_count", 0) > 0:
            sys.exit(2)
        elif batch_results.get("error"):
            sys.exit(1)
        else:
            sys.exit(0)

    # Single file mode
    if not args.pdf_file:
        parser.print_help()
        sys.exit(1)

    # Validate input file
    if not Path(args.pdf_file).exists():
        print(f"Error: File not found: {args.pdf_file}")
        sys.exit(1)

    # Run analysis
    results = unredact_pdf(args.pdf_file, args.output)

    # Output results
    if args.json:
        # Convert Rect objects to strings for JSON serialization
        print(json.dumps(results, indent=2, default=str))
    else:
        print_results(results, args.verbose)

    # Exit code based on findings
    if results["potential_fake_redactions"]:
        sys.exit(2)  # Vulnerable redactions found
    elif results["errors"]:
        sys.exit(1)  # Errors occurred
    else:
        sys.exit(0)  # Clean


if __name__ == "__main__":
    main()
