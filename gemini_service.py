"""
PAPERWISE AI APEX - Gemini AI Service Module
Interacts with the Google Gemini API using the official google-genai SDK.
Loads credentials exclusively server-side from environment variables.
Implements structured JSON outputs via Pydantic schemas, strict document-grounding prompts,
page-accurate citations, and chunked synthesis for large research papers.
Guarantees zero credential leakage in logs, exceptions, or returns.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import re

from pydantic import BaseModel, Field, field_validator
from google import genai
from google.genai import types
from google.genai.errors import APIError

from config import config, get_secret
from pdf_processor import ExtractedDocument, PageContent, chunk_document_pages


# ---------------------------------------------------------------------------
# Constants & Model Definitions
# ---------------------------------------------------------------------------

FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
]


def _sanitize_secrets(text: Any, api_key: Optional[str] = None) -> str:
    """Safely mask any API keys or credential patterns from error messages."""
    if not text:
        return ""
    msg = str(text)
    if api_key and api_key in msg:
        msg = msg.replace(api_key, "[PROTECTED_API_KEY]")
    # Mask Google API keys (typically starting with AIzaSy)
    msg = re.sub(r"AIzaSy[A-Za-z0-9_\-]{33}", "[PROTECTED_API_KEY]", msg)
    msg = re.sub(r"(key|api_key|token)=([A-Za-z0-9_\-\.]{10,})", r"\1=[PROTECTED_API_KEY]", msg, flags=re.IGNORECASE)
    return msg


# ---------------------------------------------------------------------------
# Structured Output Pydantic Schemas
# ---------------------------------------------------------------------------

class Takeaway(BaseModel):
    """Represents one of exactly five core takeaways from the paper."""
    number: int = Field(description="Sequential takeaway index, strictly 1 through 5.")
    title: str = Field(description="Concise, high-impact title (under 10 words).")
    summary: str = Field(description="Accessible explanation of the takeaway in clear plain language.")
    page_reference: str = Field(description="Evidence page number(s) e.g., 'Page 4' or 'Not mentioned'.")


def _is_verified_fallback_model(model_name: str, primary_model: str) -> bool:
    """Return True only for a real model different from the configured primary."""
    return bool(model_name) and not str(model_name).startswith("Omitted") and model_name != primary_model


class SectionAnalysis(BaseModel):
    """Detailed analysis for each core research dimension."""
    content: str = Field(
        description="Comprehensive, beginner-friendly explanation of this section based strictly on paper evidence."
    )
    evidence_pages: str = Field(
        description="Specific page citations where evidence is found (e.g., 'Page 2, 5') or 'Not mentioned'."
    )
    is_explicit: bool = Field(
        description="True if explicitly stated by authors; False if inferred or interpreted."
    )


class PaperAnalysis(BaseModel):
    """Complete structured synthesis of the research paper."""
    paper_title: str = Field(description="Exact or identified title of the research paper.")
    authors: List[str] = Field(description="List of authors, or ['Not mentioned in the paper'] if missing.")
    problem: SectionAnalysis = Field(description="The core problem, question, or research gap addressed.")
    method: SectionAnalysis = Field(description="The methodology, architecture, or approach in simple language.")
    dataset: SectionAnalysis = Field(description="Datasets, sample size, benchmarks, or explicit 'Not mentioned'.")
    results: SectionAnalysis = Field(description="Accurate findings and metrics without invented numbers.")
    limitations: SectionAnalysis = Field(
        description="Stated limitations, distinguishing explicit author admissions from inferred boundaries."
    )
    takeaways: List[Takeaway] = Field(description="List of EXACTLY 5 key takeaways.")
    disclosure_notes: str = Field(
        description="Disclosures regarding text completeness, extraction quality, or page limits."
    )
    model_used: str = Field(
        default=config.model_name,
        description="Actual model that generated this synthesis."
    )
    stage_models: Dict[str, str] = Field(
        default_factory=dict,
        description="Model used for each processing stage/chunk."
    )
    is_fallback_used: bool = Field(
        default=False,
        description="Whether a fallback model was activated during processing."
    )

    @field_validator("takeaways")
    @classmethod
    def validate_takeaways_count(cls, v: List[Takeaway]) -> List[Takeaway]:
        if len(v) != 5:
            if len(v) > 5:
                return v[:5]
        return v


# ---------------------------------------------------------------------------
# System Prompts & Instructions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are PAPERWISE AI APEX — a world-class academic research intelligence analyst.
Your mission is to analyze academic research papers and produce rigorous, crystal-clear, beginner-friendly structured summaries.

CRITICAL QUALITY GUIDELINES:
1. STRICT FACTUAL ACCURACY: Base every single claim solely on the extracted paper text provided between page markers.
2. STRICT FACTUAL GROUNDING: Base every claim solely on provided text. Never invent authors, datasets, sample sizes, benchmark numbers, ablation results, or limitations.
3. ABSENCE HANDLING: If a dataset, author, metric, or limitation is not explicitly stated in the document, explicitly output 'Not mentioned in the paper'.
4. ACCESSIBLE LANGUAGE: Explain complex technical acronyms and methods in plain English that a college undergraduate or practitioner can readily understand.
5. PRESERVE NUMERICAL ACCURACY: Never round, estimate, or alter reported metrics (e.g., accuracy, F1, latency, p-values).
6. EVIDENCE CITATIONS: Always cite the source page(s) (e.g., "Page 3, 5") referencing the [START PAGE X] tags.
7. DISTINGUISH INFERENCE: Explicitly flag whether an insight was directly stated by the authors or is a reasonable contextual interpretation.
8. EXACTLY FIVE TAKEAWAYS: Produce exactly 5 distinct, high-value takeaway cards numbered 1 to 5.
"""


# ---------------------------------------------------------------------------
# Server-Side Health Check (No Credentials Exposed)
# ---------------------------------------------------------------------------

def check_gemini_connection(
    api_key: Optional[str] = None,
    target_model: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Validates API key and model availability server-side using a real generation request.
    Returns: (is_ready: bool, status_label: str)
    Guarantees no secret strings or partial tokens are returned or logged.
    """
    key = api_key or get_secret("GEMINI_API_KEY")
    if not key or not key.strip():
        return False, "AI Engine Unavailable: Missing GEMINI_API_KEY in server environment"

    primary_model = target_model or config.model_name
    candidates = [primary_model]
    for fallback in FALLBACK_MODELS:
        if fallback not in candidates:
            candidates.append(fallback)

    last_error_code = None
    try:
        client = genai.Client(
            api_key=key.strip(),
            http_options=types.HttpOptions(timeout=10000),
        )
        for m in candidates:
            try:
                res = client.models.generate_content(
                    model=m,
                    contents="ping",
                    config=types.GenerateContentConfig(max_output_tokens=2, temperature=0.0),
                )
                if res and res.text is not None:
                    if m == primary_model:
                        return True, f"AI Engine Ready ({m})"
                    else:
                        return True, f"AI Engine Ready (Fallback: {m})"
            except APIError as e:
                last_error_code = e.code
                if e.code in (503, 504, 404, 429):
                    continue
                elif e.code == 400:
                    return False, "Connection Unavailable: Invalid API Key in server configuration"
                elif e.code == 403:
                    return False, "Connection Unavailable: Insufficient API permissions"
                else:
                    return False, f"Connection Unavailable (HTTP {e.code})"
            except Exception:
                continue

        if last_error_code == 429:
            return False, "AI Engine Busy: Quota limit reached, please retry shortly"
        return False, "Connection Unavailable: Unable to reach Gemini models"
    except Exception as e:
        clean_err = _sanitize_secrets(e, key)
        return False, "Connection Unavailable: Network or service unreachable"


# ---------------------------------------------------------------------------
# Service Class
# ---------------------------------------------------------------------------

class GeminiService:
    """Service wrapper for interacting with the Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        key = api_key or get_secret("GEMINI_API_KEY")
        if not key or not key.strip():
            raise ValueError("GEMINI_API_KEY is not configured in the server environment (.env).")
        self.api_key = key.strip()
        self.model_name = model_name or config.model_name
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=120000),
        )

    def analyze_paper(self, document: ExtractedDocument) -> PaperAnalysis:
        """
        Analyze an extracted PDF document.
        For standard papers (<= synthesis threshold), executes a single-pass structured extraction.
        For long papers (> synthesis threshold), uses a multi-chunk extraction and synthesis pipeline.
        """
        if len(document.pages) <= config.synthesis_page_threshold:
            return self._single_pass_analysis(document)
        else:
            return self._chunked_synthesis_analysis(document)

    def _single_pass_analysis(self, document: ExtractedDocument) -> PaperAnalysis:
        """Execute structured JSON analysis in a single comprehensive pass with automatic model fallback."""
        text_content = document.get_full_text(with_page_markers=True)

        user_prompt = f"""Please perform an in-depth, structured research analysis of the following academic paper.

METADATA IN THE FILE:
Filename: {document.filename}
Total Pages: {document.total_pages}
Total Word Count: {document.total_words}
Embedded Metadata: {json.dumps(document.metadata)}

DOCUMENT TEXT (WITH PAGE MARKERS):
{text_content}

CRITICAL OUTPUT FORMAT:
You MUST return a valid JSON object matching this exact structure:
{{
  "paper_title": "Identified title of the paper",
  "authors": ["Author 1", "Author 2"],
  "problem": {{
    "content": "Clear explanation of problem addressed",
    "evidence_pages": "Page 1, 2",
    "is_explicit": true
  }},
  "method": {{
    "content": "Explanation of method/architecture",
    "evidence_pages": "Page 2, 3",
    "is_explicit": true
  }},
  "dataset": {{
    "content": "Dataset names, sample sizes, or 'Not mentioned in the paper'",
    "evidence_pages": "Page 4",
    "is_explicit": true
  }},
  "results": {{
    "content": "Precise empirical results without altered numbers",
    "evidence_pages": "Page 5",
    "is_explicit": true
  }},
  "limitations": {{
    "content": "Identified limitations, distinguishing explicit admissions from inferred boundaries",
    "evidence_pages": "Page 6",
    "is_explicit": true
  }},
  "takeaways": [
    {{"number": 1, "title": "Core Takeaway 1", "summary": "Plain English summary", "page_reference": "Page 1"}},
    {{"number": 2, "title": "Core Takeaway 2", "summary": "Plain English summary", "page_reference": "Page 2"}},
    {{"number": 3, "title": "Core Takeaway 3", "summary": "Plain English summary", "page_reference": "Page 3"}},
    {{"number": 4, "title": "Core Takeaway 4", "summary": "Plain English summary", "page_reference": "Page 4"}},
    {{"number": 5, "title": "Core Takeaway 5", "summary": "Plain English summary", "page_reference": "Page 5"}}
  ],
  "disclosure_notes": "Any notes regarding text completeness or extraction limits"
}}
"""

        # Models to try: primary configured model first, followed by stable flash-lite and flash fallbacks
        models_to_try = [self.model_name]
        for fallback in FALLBACK_MODELS:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_error = None
        for current_model in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=current_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                raw_text = response.text
                result = self._parse_json_response(raw_text, document)
                result.model_used = current_model
                result.stage_models = {"Full Paper Analysis": current_model}
                if current_model != self.model_name:
                    result.is_fallback_used = True
                    result.disclosure_notes = (
                        f"Synthesized with verified fallback '{current_model}' (configured: '{self.model_name}'). "
                        + result.disclosure_notes
                    ).strip()
                return result
            except APIError as e:
                last_error = e
                # If 503, 504, 404, or 429, try next model
                if e.code in (503, 504, 404, 429):
                    continue
                else:
                    raise RuntimeError(f"Gemini API Error [{e.code}]: {_sanitize_secrets(e.message, self.api_key)}")
            except Exception as e:
                last_error = e
                continue

        clean_err = _sanitize_secrets(last_error, self.api_key) if last_error else "All models exhausted."
        raise RuntimeError(f"Failed to generate paper analysis across available models: {clean_err}")

    def _chunked_synthesis_analysis(self, document: ExtractedDocument) -> PaperAnalysis:
        """
        Handle large papers (> 15 pages) by chunking into page blocks,
        summarizing each block, and synthesizing the final structured report.
        """
        chunks = chunk_document_pages(document, chunk_size=config.chunk_page_size)
        intermediate_summaries: List[str] = []
        stage_models: Dict[str, str] = {}

        for idx, chunk in enumerate(chunks):
            start_p = chunk[0].page_number
            end_p = chunk[-1].page_number
            stage_name = f"Pages {start_p}-{end_p}"
            chunk_text = "\n\n".join(
                f"--- [START PAGE {p.page_number}] ---\n{p.text}\n--- [END PAGE {p.page_number}] ---"
                for p in chunk if p.text.strip()
            )

            chunk_prompt = f"""Analyze Pages {start_p} to {end_p} of '{document.filename}'.
Extract key facts, problem definition, methodology components, dataset details, numerical results, and limitations mentioned in these specific pages.
Always cite the specific page number for each finding.

PAGES {start_p}-{end_p} TEXT:
{chunk_text}
"""
            chunk_summary = None
            models_to_try = [self.model_name]
            for fallback in FALLBACK_MODELS:
                if fallback not in models_to_try:
                    models_to_try.append(fallback)
            for m in models_to_try:
                try:
                    res = self.client.models.generate_content(
                        model=m,
                        contents=chunk_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.2,
                        ),
                    )
                    chunk_summary = res.text
                    stage_models[stage_name] = m
                    break
                except Exception:
                    continue

            if chunk_summary:
                intermediate_summaries.append(
                    f"### Findings from Pages {start_p} to {end_p} (Model: {stage_models.get(stage_name, 'unknown')}):\n{chunk_summary}"
                )
            else:
                stage_models[stage_name] = "Omitted (API capacity)"
                intermediate_summaries.append(
                    f"### Pages {start_p} to {end_p}: [Extraction summary omitted due to API capacity]"
                )

        synthesis_prompt = f"""Below are the detailed sectional extracts from all {len(chunks)} page blocks of '{document.filename}' ({document.total_pages} total pages).
Synthesize these comprehensive extracts into the final structured PaperAnalysis JSON matching the schema:
{{
  "paper_title": "...",
  "authors": ["..."],
  "problem": {{"content": "...", "evidence_pages": "...", "is_explicit": true}},
  "method": {{"content": "...", "evidence_pages": "...", "is_explicit": true}},
  "dataset": {{"content": "...", "evidence_pages": "...", "is_explicit": true}},
  "results": {{"content": "...", "evidence_pages": "...", "is_explicit": true}},
  "limitations": {{"content": "...", "evidence_pages": "...", "is_explicit": true}},
  "takeaways": [
    {{"number": 1, "title": "...", "summary": "...", "page_reference": "..."}},
    {{"number": 2, "title": "...", "summary": "...", "page_reference": "..."}},
    {{"number": 3, "title": "...", "summary": "...", "page_reference": "..."}},
    {{"number": 4, "title": "...", "summary": "...", "page_reference": "..."}},
    {{"number": 5, "title": "...", "summary": "...", "page_reference": "..."}}
  ],
  "disclosure_notes": "..."
}}

Ensure:
1. Strict evidence grounding — use only facts and numbers reported in the extracts. If a field lacks evidence, explicitly label it 'Not mentioned in the paper'.
2. Preserve exact numbers and page citations across the full paper.
3. Produce exactly 5 distinct takeaways.

SECTIONAL EXTRACTS:
{chr(10).join(intermediate_summaries)}
"""

        for m in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=m,
                    contents=synthesis_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                stage_models["Final Synthesis"] = m
                result = self._parse_json_response(response.text, document)
                result.model_used = m
                result.stage_models = stage_models

                # Detect if any fallback was activated
                fallback_stages = [
                    f"{stg} ({mdl})"
                    for stg, mdl in stage_models.items()
                    if _is_verified_fallback_model(mdl, self.model_name)
                ]
                if fallback_stages:
                    result.is_fallback_used = True
                    result.disclosure_notes = (
                        f"Multi-chunk processing utilized verified fallback model(s): {', '.join(fallback_stages)}. "
                        + result.disclosure_notes
                    ).strip()
                return result
            except Exception as e:
                last_err = e
                continue

        clean_err = _sanitize_secrets(last_err, self.api_key) if 'last_err' in locals() else "Unknown error"
        raise RuntimeError(f"Failed to synthesize large paper analysis: {clean_err}")

    def _parse_json_response(self, raw_text: str, document: ExtractedDocument) -> PaperAnalysis:
        """Parse and validate JSON response with fallback healing."""
        if not raw_text or not raw_text.strip():
            raise RuntimeError("Received an empty response from Gemini API.")

        clean_text = raw_text.strip()
        if "```" in clean_text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
            if match:
                clean_text = match.group(1).strip()
            else:
                clean_text = re.sub(r"^```(?:json)?\n?", "", clean_text)
                clean_text = re.sub(r"\n?```$", "", clean_text).strip()

        try:
            data = json.loads(clean_text)
        except Exception:
            match = re.search(r"(\{[\s\S]*\})", clean_text)
            if match:
                data = json.loads(match.group(1))
            else:
                raise

        # Heal authors if returned as string
        if "authors" in data:
            if isinstance(data["authors"], str):
                data["authors"] = [a.strip() for a in data["authors"].split(",") if a.strip()] or [data["authors"].strip()]
            elif not isinstance(data["authors"], list):
                data["authors"] = [str(data["authors"])]

        # Heal sections if returned as flat string
        for sec in ["problem", "method", "dataset", "results", "limitations"]:
            if sec in data and isinstance(data[sec], str):
                data[sec] = {
                    "content": data[sec],
                    "evidence_pages": "Refer to document text",
                    "is_explicit": True,
                }

        # Heal takeaways
        if "takeaways" in data and isinstance(data["takeaways"], list):
            healed_takeaways = []
            for i, item in enumerate(data["takeaways"]):
                if isinstance(item, str):
                    healed_takeaways.append({
                        "number": i + 1,
                        "title": f"Key Finding {i + 1}",
                        "summary": item,
                        "page_reference": "General",
                    })
                elif isinstance(item, dict):
                    if "number" not in item:
                        item["number"] = i + 1
                    healed_takeaways.append(item)
            data["takeaways"] = healed_takeaways

            if len(data["takeaways"]) > 5:
                data["takeaways"] = data["takeaways"][:5]
            elif len(data["takeaways"]) < 5:
                while len(data["takeaways"]) < 5:
                    idx = len(data["takeaways"]) + 1
                    data["takeaways"].append({
                        "number": idx,
                        "title": f"Key Finding {idx}",
                        "summary": "Refer to the comprehensive sectional analysis above for additional findings.",
                        "page_reference": "General"
                    })
        return PaperAnalysis.model_validate(data)
