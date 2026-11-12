"""
PAPERWISE AI - PDF Processor Module
Robust, page-by-page PDF extraction and validation engine using pypdf.
Handles empty, scanned, encrypted, malformed, and oversized documents gracefully.
"""

from dataclasses import dataclass, field
import io
from typing import BinaryIO, Dict, List, Optional, Tuple, Union
import pypdf
from pypdf.errors import EmptyFileError, PdfReadError

from config import config


# ---------------------------------------------------------------------------
# Custom Exceptions for Granular Error Feedback
# ---------------------------------------------------------------------------

class PDFProcessingError(Exception):
    """Base exception for all PDF processing errors."""
    pass


class EncryptedPDFError(PDFProcessingError):
    """Raised when a PDF is password protected or encrypted."""
    pass


class MalformedPDFError(PDFProcessingError):
    """Raised when a PDF file is corrupt, invalid, or cannot be parsed."""
    pass


class OversizedPDFError(PDFProcessingError):
    """Raised when a PDF exceeds size or page count limits."""
    pass


class ScannedOrEmptyPDFError(PDFProcessingError):
    """Raised when a PDF has insufficient or zero extractable digital text."""
    pass


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class PageContent:
    """Individual page extraction data."""
    page_number: int  # 1-indexed
    text: str
    word_count: int
    char_count: int


@dataclass
class ExtractedDocument:
    """Consolidated document model after extraction and validation."""
    filename: str
    total_pages: int
    total_words: int
    total_chars: int
    pages: List[PageContent]
    metadata: Dict[str, str] = field(default_factory=dict)
    is_scanned_or_empty: bool = False
    warnings: List[str] = field(default_factory=list)

    def get_full_text(self, with_page_markers: bool = True) -> str:
        """
        Combine all pages into a unified text string.
        Optionally injects clear page markers to aid the LLM in page-accurate citations.
        """
        if not with_page_markers:
            return "\n\n".join(p.text for p in self.pages if p.text.strip())

        parts = []
        for p in self.pages:
            if p.text.strip():
                parts.append(f"--- [START PAGE {p.page_number}] ---\n{p.text}\n--- [END PAGE {p.page_number}] ---")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Core Extraction and Validation Functions
# ---------------------------------------------------------------------------

def validate_pdf_stream(
    file_bytes: bytes,
    filename: str = "document.pdf",
    max_size_mb: Optional[int] = None,
) -> None:
    """
    Validate raw byte size and PDF magic header (%PDF).
    """
    max_mb = max_size_mb or config.max_file_size_mb
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > max_mb:
        raise OversizedPDFError(
            f"File '{filename}' is {size_mb:.1f} MB, which exceeds the allowable limit of {max_mb} MB."
        )

    if len(file_bytes) < 4 or not file_bytes.startswith(b"%PDF"):
        raise MalformedPDFError(
            f"File '{filename}' does not appear to be a valid PDF document (missing standard %PDF magic header)."
        )


def extract_pdf_content(
    file_source: Union[bytes, BinaryIO, io.BytesIO],
    filename: str = "document.pdf",
    max_pages: Optional[int] = None,
    min_word_threshold: Optional[int] = None,
) -> ExtractedDocument:
    """
    Extract text page-by-page from a PDF stream or bytes.

    Features:
    - Verifies encryption and flags encrypted files immediately.
    - Extracts metadata safely (Title, Author, Producer).
    - Cleans and normalizes page text while counting words and characters.
    - Flags scanned documents (where text is rendered as bitmap images without OCR).
    - Enforces maximum page limits to prevent out-of-memory or timeout errors.

    Returns:
        ExtractedDocument object containing all pages and metadata.

    Raises:
        EncryptedPDFError, MalformedPDFError, OversizedPDFError, ScannedOrEmptyPDFError
    """
    # Normalize input into BytesIO stream and bytes
    if isinstance(file_source, bytes):
        raw_bytes = file_source
        stream = io.BytesIO(raw_bytes)
    elif hasattr(file_source, "read"):
        current_pos = file_source.tell() if hasattr(file_source, "tell") else 0
        raw_bytes = file_source.read()
        # Restore position if possible
        if hasattr(file_source, "seek"):
            file_source.seek(current_pos)
        stream = io.BytesIO(raw_bytes)
    else:
        raise MalformedPDFError(f"Unsupported file source type: {type(file_source)}")

    # 1. Byte and magic header validation
    validate_pdf_stream(raw_bytes, filename)

    # 2. Parse PDF with pypdf
    try:
        reader = pypdf.PdfReader(stream)
    except EmptyFileError:
        raise MalformedPDFError(f"The PDF file '{filename}' is completely empty (0 bytes or empty structure).")
    except PdfReadError as e:
        raise MalformedPDFError(f"Failed to read PDF '{filename}': {str(e)}")
    except Exception as e:
        raise MalformedPDFError(f"Unexpected parsing error for '{filename}': {str(e)}")

    # 3. Check encryption
    if reader.is_encrypted:
        try:
            # Try decrypting with empty password
            decrypted = reader.decrypt("")
            if decrypted == 0:  # 0 indicates decryption failed
                raise EncryptedPDFError(
                    f"PDF '{filename}' is password-protected or encrypted. "
                    f"Please provide an unprotected PDF."
                )
        except Exception:
            raise EncryptedPDFError(
                f"PDF '{filename}' is password-protected or encrypted. "
                f"Please provide an unprotected PDF."
            )

    # 4. Check page count limits
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise MalformedPDFError(f"PDF '{filename}' contains 0 pages.")

    max_p = max_pages or config.max_page_count
    warnings: List[str] = []

    pages_to_process = total_pages
    if total_pages > max_p:
        warnings.append(
            f"Document has {total_pages} pages; processing was capped to the first {max_p} pages to ensure optimal analysis."
        )
        pages_to_process = max_p

    # 5. Extract metadata safely
    metadata_dict: Dict[str, str] = {}
    try:
        if reader.metadata:
            for k, v in reader.metadata.items():
                clean_key = str(k).lstrip("/").strip()
                if v:
                    metadata_dict[clean_key] = str(v).strip()
    except Exception:
        # Non-critical, ignore metadata failure
        pass

    # 6. Extract page-by-page text
    pages: List[PageContent] = []
    total_words = 0
    total_chars = 0

    for idx in range(pages_to_process):
        page_num = idx + 1
        page_obj = reader.pages[idx]

        try:
            extracted_text = page_obj.extract_text() or ""
        except Exception as e:
            warnings.append(f"Warning: Page {page_num} text extraction partially failed: {str(e)}")
            extracted_text = ""

        # Normalize whitespace
        cleaned_text = " ".join(extracted_text.split())
        words = cleaned_text.split()
        word_count = len(words)
        char_count = len(cleaned_text)

        total_words += word_count
        total_chars += char_count

        pages.append(
            PageContent(
                page_number=page_num,
                text=extracted_text.strip(),
                word_count=word_count,
                char_count=char_count,
            )
        )

    # 7. Detect Scanned / Empty PDFs
    threshold = min_word_threshold or config.min_word_count_threshold
    if total_words < threshold:
        raise ScannedOrEmptyPDFError(
            f"The uploaded PDF '{filename}' contains only {total_words} extractable digital words across {total_pages} pages.\n\n"
            f"This usually occurs when a research paper is scanned as a photographic image without embedded digital text.\n"
            f"PAPERWISE AI requires digital text to generate an accurate analysis. Scanned documents require an OCR "
            f"(Optical Character Recognition) tool like Adobe Acrobat OCR or Tesseract to convert scanned images into selectable text first."
        )

    return ExtractedDocument(
        filename=filename,
        total_pages=total_pages,
        total_words=total_words,
        total_chars=total_chars,
        pages=pages,
        metadata=metadata_dict,
        is_scanned_or_empty=False,
        warnings=warnings,
    )


def chunk_document_pages(
    doc: ExtractedDocument,
    chunk_size: Optional[int] = None,
) -> List[List[PageContent]]:
    """
    Split document pages into sequential batches for synthesis of large papers.
    Prevents context truncation and maintains page tracking across chunks.
    """
    size = chunk_size or config.chunk_page_size
    chunks: List[List[PageContent]] = []
    for i in range(0, len(doc.pages), size):
        chunks.append(doc.pages[i:i + size])
    return chunks
