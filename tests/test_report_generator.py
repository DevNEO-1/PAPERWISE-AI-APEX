"""
Unit tests for report_generator module in PAPERWISE AI APEX.
"""

from gemini_service import PaperAnalysis, SectionAnalysis, Takeaway
from pdf_processor import ExtractedDocument, PageContent
from report_generator import generate_markdown_report


def test_markdown_report_generation_structure():
    """Verify that generated markdown includes all required sections, badges, and citations."""
    analysis = PaperAnalysis(
        paper_title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        problem=SectionAnalysis(
            content="Recurrent models compute hidden states sequentially, inhibiting parallelization.",
            evidence_pages="Page 2",
            is_explicit=True,
        ),
        method=SectionAnalysis(
            content="The Transformer relies entirely on an attention mechanism without recurrence.",
            evidence_pages="Page 3, 4",
            is_explicit=True,
        ),
        dataset=SectionAnalysis(
            content="WMT 2014 English-to-German and English-to-French.",
            evidence_pages="Page 7",
            is_explicit=True,
        ),
        results=SectionAnalysis(
            content="Achieved 28.4 BLEU on English-to-German, outperforming existing best results.",
            evidence_pages="Page 8",
            is_explicit=True,
        ),
        limitations=SectionAnalysis(
            content="Requires fixed context window; restricted attention needed for very long sequences.",
            evidence_pages="Page 2, 7",
            is_explicit=False,
        ),
        takeaways=[
            Takeaway(number=1, title="No Recurrence", summary="Self-attention replaces RNNs.", page_reference="Page 1"),
            Takeaway(number=2, title="Parallel Training", summary="Training speed increased drastically.", page_reference="Page 2"),
            Takeaway(number=3, title="Scaled Dot-Product", summary="Prevents vanishing softmax gradients.", page_reference="Page 4"),
            Takeaway(number=4, title="Multi-Head Attention", summary="Attends to multiple subspaces.", page_reference="Page 5"),
            Takeaway(number=5, title="State-of-the-art BLEU", summary="28.4 BLEU score on WMT 2014.", page_reference="Page 8"),
        ],
        disclosure_notes="Test synthetic verification.",
    )

    doc = ExtractedDocument(
        filename="genuine_research_paper.pdf",
        total_pages=15,
        total_words=6123,
        total_chars=35000,
        pages=[PageContent(page_number=1, text="Sample text", word_count=2, char_count=11)],
        metadata={"title": "Attention Is All You Need"},
    )

    report = generate_markdown_report(analysis, doc, model_name="Gemini 3.5 Flash-Lite")

    assert "# 📑 PAPERWISE AI APEX" in report
    assert "Attention Is All You Need" in report
    assert "Ashish Vaswani" in report
    assert "1. Problem Statement" in report
    assert "2. Methodology & Approach" in report
    assert "3. Dataset & Experimental Setup" in report
    assert "4. Key Results & Findings" in report
    assert "5. Limitations & Boundaries" in report
    assert "6. Five Key Takeaways" in report
    assert "[Explicit Author Evidence]" in report
    assert "[Contextual Interpretation]" in report
    assert "[01] No Recurrence" in report
    assert "[05] State-of-the-art BLEU" in report
    assert "28.4 BLEU" in report
    assert "7. Academic Integrity & Disclosures" in report


def test_markdown_report_model_attribution_and_fallback_badges():
    """Verify that model fallback status and multi-stage attribution table are rendered."""
    analysis = PaperAnalysis(
        paper_title="Multi-Stage Paper",
        authors=["Author One"],
        problem=SectionAnalysis(content="Problem", evidence_pages="Page 1", is_explicit=True),
        method=SectionAnalysis(content="Method", evidence_pages="Page 2", is_explicit=True),
        dataset=SectionAnalysis(content="Dataset", evidence_pages="Page 3", is_explicit=True),
        results=SectionAnalysis(content="Results", evidence_pages="Page 4", is_explicit=True),
        limitations=SectionAnalysis(content="Limitations", evidence_pages="Page 5", is_explicit=True),
        takeaways=[Takeaway(number=i, title=f"T{i}", summary=f"S{i}", page_reference="P1") for i in range(1, 6)],
        disclosure_notes="Multi-stage test note.",
        model_used="gemini-3.8-flash",
        stage_models={
            "Pages 1-5": "gemini-3.5-flash-lite",
            "Pages 6-10": "gemini-3.8-flash",
            "Final Synthesis": "gemini-3.8-flash",
            "Pages 11-15": "Omitted (API capacity)",
        },
        is_fallback_used=True,
    )

    doc = ExtractedDocument(
        filename="multistage.pdf",
        total_pages=10,
        total_words=2000,
        total_chars=12000,
        pages=[PageContent(page_number=1, text="Text", word_count=1, char_count=4)],
    )

    report = generate_markdown_report(analysis, doc)

    assert "gemini-3.8-flash" in report
    assert "Verified Fallback Active" in report
    assert "Processing Pipeline & Model Attribution" in report
    assert "Pages 1-5" in report
    assert "Final Synthesis" in report
    assert "Not processed (API capacity)" in report
    assert "Pages 11-15" in report
