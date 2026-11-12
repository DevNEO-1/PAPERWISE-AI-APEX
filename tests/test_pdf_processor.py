"""
Unit tests for pdf_processor.py
Tests validation, error handling, scanned/empty detection, and chunking.
"""

import io
import pytest
import pypdf

from pdf_processor import (
    PageContent,
    ExtractedDocument,
    MalformedPDFError,
    OversizedPDFError,
    ScannedOrEmptyPDFError,
    validate_pdf_stream,
    extract_pdf_content,
    chunk_document_pages,
)


def create_blank_pdf_bytes(num_pages: int = 1) -> bytes:
    """Helper to synthesize a syntactically valid PDF with blank pages."""
    writer = pypdf.PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=300, height=400)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class TestPDFValidation:
    """Tests for PDF format and size validation."""

    def test_empty_bytes_raises_malformed(self):
        """Empty input must raise MalformedPDFError."""
        with pytest.raises(MalformedPDFError, match="missing standard %PDF magic header"):
            validate_pdf_stream(b"")

    def test_invalid_header_raises_malformed(self):
        """Files without %PDF header must raise MalformedPDFError."""
        fake_file = b"This is a plain text file pretending to be PDF."
        with pytest.raises(MalformedPDFError, match="missing standard %PDF magic header"):
            validate_pdf_stream(fake_file)

    def test_oversized_file_raises_error(self):
        """Files exceeding max_size_mb must raise OversizedPDFError."""
        # 2 MB byte sequence with %PDF prefix
        large_bytes = b"%PDF-1.4 " + b"X" * (2 * 1024 * 1024)
        with pytest.raises(OversizedPDFError, match="exceeds the allowable limit"):
            validate_pdf_stream(large_bytes, max_size_mb=1)

    def test_valid_header_passes(self):
        """Standard %PDF header within limits passes validation."""
        valid_header = b"%PDF-1.7\nSample content"
        validate_pdf_stream(valid_header, max_size_mb=10)


class TestPDFExtraction:
    """Tests for PDF text extraction and error states."""

    def test_blank_scanned_pdf_raises_scanned_error(self):
        """A valid PDF with no extractable text must trigger ScannedOrEmptyPDFError."""
        blank_pdf = create_blank_pdf_bytes(num_pages=2)
        with pytest.raises(ScannedOrEmptyPDFError, match="contains only 0 extractable digital words"):
            extract_pdf_content(blank_pdf, filename="scanned_test.pdf", min_word_threshold=50)

    def test_chunk_document_pages(self):
        """Page chunking should split into accurate batches without dropping pages."""
        pages = [
            PageContent(page_number=i, text=f"Text for page {i}", word_count=4, char_count=15)
            for i in range(1, 16)
        ]
        doc = ExtractedDocument(
            filename="test_long_paper.pdf",
            total_pages=15,
            total_words=60,
            total_chars=225,
            pages=pages,
        )

        chunks = chunk_document_pages(doc, chunk_size=5)
        assert len(chunks) == 3
        assert len(chunks[0]) == 5
        assert chunks[0][0].page_number == 1
        assert chunks[0][-1].page_number == 5
        assert chunks[2][0].page_number == 11
        assert chunks[2][-1].page_number == 15

    def test_document_full_text_page_markers(self):
        """ExtractedDocument.get_full_text must inject readable page markers for LLM citation."""
        pages = [
            PageContent(page_number=1, text="Introduction and background.", word_count=3, char_count=30),
            PageContent(page_number=2, text="Methodology and results.", word_count=3, char_count=24),
        ]
        doc = ExtractedDocument(
            filename="sample.pdf",
            total_pages=2,
            total_words=6,
            total_chars=54,
            pages=pages,
        )

        full_text = doc.get_full_text(with_page_markers=True)
        assert "--- [START PAGE 1] ---" in full_text
        assert "Introduction and background." in full_text
        assert "--- [END PAGE 1] ---" in full_text
        assert "--- [START PAGE 2] ---" in full_text
        assert "Methodology and results." in full_text

    def test_sample_academic_paper_extraction(self):
        """Test end-to-end extraction on a real multi-page scientific paper PDF."""
        import os
        sample_path = os.path.join(os.path.dirname(__file__), "..", "sample_papers", "sample_research_paper.pdf")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                doc = extract_pdf_content(f.read(), filename="sample_research_paper.pdf")
            assert doc.total_pages == 3
            assert doc.total_words > 100
            assert "DocScale" in doc.get_full_text()
            assert doc.is_scanned_or_empty is False

    def test_genuine_arxiv_paper_extraction(self):
        """Test extraction on a genuine published arXiv peer-reviewed paper PDF."""
        import os
        arxiv_path = os.path.join(os.path.dirname(__file__), "..", "sample_papers", "genuine_research_paper.pdf")
        if os.path.exists(arxiv_path):
            with open(arxiv_path, "rb") as f:
                doc = extract_pdf_content(f.read(), filename="genuine_research_paper.pdf")
            assert doc.total_pages == 15
            assert doc.total_words > 5000
            assert "Attention Is All You Need" in doc.get_full_text()
            assert "Vaswani" in doc.get_full_text()
            assert doc.is_scanned_or_empty is False


