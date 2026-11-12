"""
Unit tests for gemini_service module in PAPERWISE AI APEX.
"""

from gemini_service import GeminiService, PaperAnalysis, SectionAnalysis, Takeaway
from pdf_processor import ExtractedDocument, PageContent


def test_paper_analysis_pydantic_validation():
    """Verify PaperAnalysis Pydantic schema validation."""
    data = {
        "paper_title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "problem": {
            "content": "Sequential computation prevents parallelization.",
            "evidence_pages": "Page 1, 2",
            "is_explicit": True,
        },
        "method": {
            "content": "Multi-head self-attention without recurrence.",
            "evidence_pages": "Page 3, 4",
            "is_explicit": True,
        },
        "dataset": {
            "content": "WMT 2014 English-German.",
            "evidence_pages": "Page 7",
            "is_explicit": True,
        },
        "results": {
            "content": "28.4 BLEU score.",
            "evidence_pages": "Page 8",
            "is_explicit": True,
        },
        "limitations": {
            "content": "Limited attention span for infinite contexts.",
            "evidence_pages": "Page 9",
            "is_explicit": False,
        },
        "takeaways": [
            {"number": 1, "title": "Takeaway 1", "summary": "Summary 1", "page_reference": "Page 1"},
            {"number": 2, "title": "Takeaway 2", "summary": "Summary 2", "page_reference": "Page 2"},
            {"number": 3, "title": "Takeaway 3", "summary": "Summary 3", "page_reference": "Page 3"},
            {"number": 4, "title": "Takeaway 4", "summary": "Summary 4", "page_reference": "Page 4"},
            {"number": 5, "title": "Takeaway 5", "summary": "Summary 5", "page_reference": "Page 5"},
        ],
        "disclosure_notes": "Grounding verified.",
    }

    model = PaperAnalysis.model_validate(data)
    assert model.paper_title == "Attention Is All You Need"
    assert len(model.takeaways) == 5
    assert model.takeaways[4].number == 5


def test_json_healing_and_takeaways_normalization():
    """Test JSON markdown stripping and guaranteeing exactly 5 takeaways."""
    svc = GeminiService(api_key="test_dummy_key_not_leaked")
    doc = ExtractedDocument(
        filename="dummy.pdf",
        total_pages=1,
        total_words=100,
        total_chars=500,
        pages=[PageContent(page_number=1, text="Text", word_count=1, char_count=4)],
    )

    # Raw response with markdown code fences and only 3 takeaways
    raw_markdown_json = """```json
    {
        "paper_title": "Test Title",
        "authors": ["Author A"],
        "problem": {"content": "Problem", "evidence_pages": "Page 1", "is_explicit": true},
        "method": {"content": "Method", "evidence_pages": "Page 1", "is_explicit": true},
        "dataset": {"content": "Dataset", "evidence_pages": "Page 1", "is_explicit": true},
        "results": {"content": "Results", "evidence_pages": "Page 1", "is_explicit": true},
        "limitations": {"content": "Limitations", "evidence_pages": "Page 1", "is_explicit": true},
        "takeaways": [
            {"number": 1, "title": "T1", "summary": "S1", "page_reference": "Page 1"},
            {"number": 2, "title": "T2", "summary": "S2", "page_reference": "Page 1"},
            {"number": 3, "title": "T3", "summary": "S3", "page_reference": "Page 1"}
        ],
        "disclosure_notes": "Test"
    }
    ```"""

    parsed = svc._parse_json_response(raw_markdown_json, doc)
    assert parsed.paper_title == "Test Title"
    assert len(parsed.takeaways) == 5  # Automatically healed to exactly 5
    assert parsed.takeaways[3].number == 4
    assert parsed.takeaways[4].number == 5


def test_stage_model_tracking_and_fallback_flag():
    """Verify PaperAnalysis tracks model_used, stage_models, and is_fallback_used."""
    analysis = PaperAnalysis(
        paper_title="Test Tracking",
        authors=["Author"],
        problem=SectionAnalysis(content="P", evidence_pages="1", is_explicit=True),
        method=SectionAnalysis(content="M", evidence_pages="1", is_explicit=True),
        dataset=SectionAnalysis(content="D", evidence_pages="1", is_explicit=True),
        results=SectionAnalysis(content="R", evidence_pages="1", is_explicit=True),
        limitations=SectionAnalysis(content="L", evidence_pages="1", is_explicit=True),
        takeaways=[Takeaway(number=i, title=f"T{i}", summary=f"S{i}", page_reference="1") for i in range(1, 6)],
        disclosure_notes="Test note",
        model_used="gemini-3.8-flash",
        stage_models={"Pages 1-5": "gemini-3.8-flash", "Synthesis": "gemini-3.8-flash"},
        is_fallback_used=True,
    )
    assert analysis.model_used == "gemini-3.8-flash"
    assert analysis.is_fallback_used is True
    assert len(analysis.stage_models) == 2



def test_omitted_stage_is_not_counted_as_verified_fallback():
    """An API-capacity omission is not a model and must not be labeled a fallback."""
    from gemini_service import _is_verified_fallback_model

    assert _is_verified_fallback_model("gemini-3.8-flash", "gemini-3.5-flash-lite")
    assert not _is_verified_fallback_model("gemini-3.5-flash-lite", "gemini-3.5-flash-lite")
    assert not _is_verified_fallback_model("Omitted (API capacity)", "gemini-3.5-flash-lite")


def test_fallback_models_follow_project_allowlist():
    """
    Regression test enforcing this project's explicit fallback allowlist policy.
    Gemini 1.5, 2.0, and 2.5 families are intentionally excluded by project policy;
    this does not assert that every excluded model is officially deprecated or shut down.
    """
    from gemini_service import FALLBACK_MODELS

    policy_excluded_exact = {
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    }
    policy_excluded_prefixes = ("gemini-1.5", "gemini-2.0", "gemini-2.5")

    for model in FALLBACK_MODELS:
        assert model not in policy_excluded_exact, f"Policy-excluded model '{model}' found in FALLBACK_MODELS"
        for prefix in policy_excluded_prefixes:
            assert not model.startswith(prefix), f"Model '{model}' matches project-excluded prefix '{prefix}'"

    supported_models = {
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
    }
    for model in FALLBACK_MODELS:
        assert model in supported_models, f"Model '{model}' is not in known supported Gemini models: {supported_models}"

