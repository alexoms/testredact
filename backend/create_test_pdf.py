#!/usr/bin/env python3
"""
Create test PDFs with various types of redactions for testing the unredaction tool.
"""

import sys

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF is required. Install with: pip install PyMuPDF")
    sys.exit(1)


def create_fake_redaction_pdf(output_path: str = "test_fake_redaction.pdf"):
    """
    Create a PDF with fake redactions (black rectangles over text).
    The underlying text is still present and extractable.
    """
    doc = fitz.open()
    page = doc.new_page()

    # Add some regular text
    page.insert_text(
        (72, 72),
        "PUBLIC DOCUMENT - Unclassified Information",
        fontsize=14
    )

    page.insert_text(
        (72, 120),
        "Employee Name: John Smith",
        fontsize=12
    )

    # Add sensitive text that will be "redacted"
    page.insert_text(
        (72, 150),
        "SSN: 123-45-6789",
        fontsize=12
    )

    page.insert_text(
        (72, 180),
        "Credit Card: 4111-1111-1111-1111",
        fontsize=12
    )

    page.insert_text(
        (72, 210),
        "Password: SuperSecret123!",
        fontsize=12
    )

    page.insert_text(
        (72, 260),
        "This text is visible and not redacted.",
        fontsize=12
    )

    # Add fake redactions (black rectangles) over sensitive data
    # These cover the text but don't remove it!
    shape = page.new_shape()

    # Cover SSN
    shape.draw_rect(fitz.Rect(100, 140, 200, 160))
    shape.finish(color=(0, 0, 0), fill=(0, 0, 0))

    # Cover Credit Card
    shape.draw_rect(fitz.Rect(130, 170, 290, 190))
    shape.finish(color=(0, 0, 0), fill=(0, 0, 0))

    # Cover Password
    shape.draw_rect(fitz.Rect(130, 200, 260, 220))
    shape.finish(color=(0, 0, 0), fill=(0, 0, 0))

    shape.commit()

    doc.save(output_path)
    doc.close()
    print(f"Created fake redaction PDF: {output_path}")
    print("  - Contains hidden SSN, Credit Card, and Password under black rectangles")


def create_proper_redaction_pdf(output_path: str = "test_proper_redaction.pdf"):
    """
    Create a PDF with proper redactions where text is actually removed.
    """
    doc = fitz.open()
    page = doc.new_page()

    # Add some text
    page.insert_text(
        (72, 72),
        "PUBLIC DOCUMENT - Properly Redacted",
        fontsize=14
    )

    page.insert_text(
        (72, 120),
        "Employee Name: John Smith",
        fontsize=12
    )

    page.insert_text(
        (72, 150),
        "SSN: 123-45-6789",
        fontsize=12
    )

    page.insert_text(
        (72, 180),
        "This SSN above has been properly redacted.",
        fontsize=12
    )

    # Add proper redaction annotation
    redact_rect = fitz.Rect(100, 140, 200, 160)
    page.add_redact_annot(redact_rect)

    # Apply the redaction (this actually removes the text)
    page.apply_redactions()

    doc.save(output_path)
    doc.close()
    print(f"Created properly redacted PDF: {output_path}")
    print("  - SSN text has been properly removed, not just covered")


def create_unapplied_redaction_pdf(output_path: str = "test_unapplied_redaction.pdf"):
    """
    Create a PDF with redaction annotations that haven't been applied yet.
    The text is still visible and extractable.
    """
    doc = fitz.open()
    page = doc.new_page()

    page.insert_text(
        (72, 72),
        "CONFIDENTIAL - Pending Redaction Review",
        fontsize=14
    )

    page.insert_text(
        (72, 120),
        "Secret Project Codename: PHOENIX",
        fontsize=12
    )

    page.insert_text(
        (72, 150),
        "API Key: sk-abc123xyz789secret",
        fontsize=12
    )

    page.insert_text(
        (72, 180),
        "Database Password: admin123",
        fontsize=12
    )

    # Add redaction annotations but DON'T apply them
    page.add_redact_annot(fitz.Rect(180, 110, 350, 130))  # Codename
    page.add_redact_annot(fitz.Rect(100, 140, 320, 160))  # API Key
    page.add_redact_annot(fitz.Rect(160, 170, 280, 190))  # Password

    # Note: NOT calling page.apply_redactions()

    doc.save(output_path)
    doc.close()
    print(f"Created unapplied redaction PDF: {output_path}")
    print("  - Contains redaction annotations that haven't been applied")
    print("  - Text is still visible and extractable")


def main():
    print("Creating test PDFs for unredaction testing...\n")

    create_fake_redaction_pdf()
    create_proper_redaction_pdf()
    create_unapplied_redaction_pdf()

    print("\n" + "=" * 50)
    print("Test files created! Run the unredaction tool:")
    print("  python pdf_unredact.py test_fake_redaction.pdf")
    print("  python pdf_unredact.py test_proper_redaction.pdf")
    print("  python pdf_unredact.py test_unapplied_redaction.pdf")
    print("=" * 50)


if __name__ == "__main__":
    main()
