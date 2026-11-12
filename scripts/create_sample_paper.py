"""
Utility script to generate a sample academic research paper PDF
with valid PDF syntax, multiple pages, and realistic text content
for offline and end-to-end testing of PAPERWISE AI.
"""

import io
import os


def generate_sample_academic_pdf(output_path: str = "sample_papers/sample_research_paper.pdf") -> str:
    """
    Creates a valid multi-page PDF with selectable text formatted
    as a realistic computer science academic research paper.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pages_text = [
        # Page 1: Abstract & Problem
        [
            "Attention-Driven Multi-Scale Representation Learning for Document Intelligence",
            "Dr. Elena Vance, Dr. Marcus Thorne, Sophia Lin",
            "Department of Computer Science, Institute of Advanced Technology",
            "",
            "ABSTRACT",
            "Academic document understanding requires joint reasoning over dense typographic layouts and semantic text.",
            "Existing vision-language models frequently suffer from spatial misalignment and high computational complexity.",
            "In this work, we present DocScale, a novel multi-scale self-attention architecture tailored for document parsing.",
            "",
            "1. INTRODUCTION AND PROBLEM STATEMENT",
            "Academic and scientific PDFs present unique challenges due to multi-column formats, mathematical equations, and inline figures.",
            "Current optical character pipelines and transformer models struggle with boundary detection across contiguous pages.",
            "Specifically, existing baselines fail to maintain cross-page continuity, resulting in a 24.3% error rate in multi-page reasoning.",
            "Our primary objective is to resolve spatial degradation while reducing model parameter footprint by 35% without sacrificing accuracy.",
        ],
        # Page 2: Methodology & Dataset
        [
            "2. METHODOLOGY AND ARCHITECTURE",
            "DocScale incorporates a hierarchical cross-attention mechanism that decouples layout spatial coordinates from lexical embeddings.",
            "The architecture consists of 12 encoder layers equipped with relative positional bias and gated visual pooling.",
            "By computing cross-attention over multi-scale feature pyramids, the model preserves fine-grained token boundaries.",
            "",
            "3. DATASET AND EXPERIMENTAL SETUP",
            "We evaluate DocScale on the newly curated SciDoc-50k benchmark, consisting of 50,000 open-access peer-reviewed papers.",
            "The dataset spans four major scientific domains: Computer Science, Physics, Bioengineering, and Computational Linguistics.",
            "Data preprocessing involved standard normalization and filtering of documents with fewer than 100 tokens.",
            "All models were trained on 8 NVIDIA A100 GPUs using AdamW optimizer with a learning rate of 2e-4 and cosine decay.",
        ],
        # Page 3: Results & Limitations
        [
            "4. RESULTS AND EMPIRICAL FINDINGS",
            "DocScale achieves a state-of-the-art Macro F1-score of 94.6% on document layout analysis, outperforming previous baselines by 4.8%.",
            "Inference throughput reached 142 pages per second on an A100 GPU, reflecting a 3.2x speedup over dense vision transformers.",
            "Ablation studies confirm that multi-scale positional gates contribute 2.9% of the total F1 improvement.",
            "",
            "5. LIMITATIONS",
            "Despite strong empirical performance, DocScale exhibits several stated limitations.",
            "First, the model performance degrades on historical manuscripts printed prior to 1950 due to font irregularities.",
            "Second, our current implementation assumes single-language English documents and does not natively parse right-to-left scripts.",
            "Third, inference memory scales quadratically for documents exceeding 100 pages, necessitating future work in linear attention.",
            "",
            "6. CONCLUSION",
            "DocScale provides an efficient, verifiable framework for scientific document summarization and semantic extraction.",
        ],
    ]

    # Build valid PDF objects
    objects = []
    # 1 0: Catalog
    # 2 0: Pages
    # 3..: Font, Page objects, Content stream objects

    # Object 1: Catalog
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    # Object 2: Pages placeholder (will update with kids)
    # We will build objects sequentially

    pdf_objs = {}
    obj_id = 1

    catalog_id = 1
    pages_id = 2
    font_id = 3

    pdf_objs[catalog_id] = b"<< /Type /Catalog /Pages 2 0 R >>"
    pdf_objs[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    page_ids = []
    content_ids = []

    current_id = 4
    for p_idx, lines in enumerate(pages_text):
        p_id = current_id
        c_id = current_id + 1
        current_id += 2

        page_ids.append(p_id)
        content_ids.append(c_id)

        # Build content stream
        stream_lines = [b"BT", b"/F1 11 Tf", b"50 750 Td", b"16 TL"]
        for line in lines:
            safe_text = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_lines.append(f"({safe_text}) Tj".encode("latin1", errors="replace"))
            stream_lines.append(b"T*")
        stream_lines.append(b"ET")

        stream_data = b"\n".join(stream_lines)
        c_obj = f"<< /Length {len(stream_data)} >>\nstream\n".encode("latin1") + stream_data + b"\nendstream"
        pdf_objs[c_id] = c_obj

        p_obj = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {c_id} 0 R >>".encode("latin1")
        pdf_objs[p_id] = p_obj

    # Build Pages object
    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
    pdf_objs[pages_id] = f"<< /Type /Pages /Kids [ {kids_str} ] /Count {len(page_ids)} >>".encode("latin1")

    # Assemble PDF with cross-reference table
    output = bytearray()
    output.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    xref_offsets = {}
    for oid in sorted(pdf_objs.keys()):
        xref_offsets[oid] = len(output)
        output.extend(f"{oid} 0 obj\n".encode("latin1"))
        output.extend(pdf_objs[oid])
        output.extend(b"\nendobj\n")

    xref_pos = len(output)
    output.extend(b"xref\n")
    max_id = max(pdf_objs.keys())
    output.extend(f"0 {max_id + 1}\n".encode("latin1"))
    output.extend(b"0000000000 65535 f \n")
    for oid in range(1, max_id + 1):
        pos = xref_offsets.get(oid, 0)
        output.extend(f"{pos:010d} 00000 n \n".encode("latin1"))

    output.extend(b"trailer\n")
    output.extend(f"<< /Size {max_id + 1} /Root 1 0 R >>\n".encode("latin1"))
    output.extend(b"startxref\n")
    output.extend(f"{xref_pos}\n%%EOF\n".encode("latin1"))

    with open(output_path, "wb") as f:
        f.write(output)

    print(f"Generated sample research paper at: {output_path} ({len(output)} bytes)")
    return output_path


if __name__ == "__main__":
    generate_sample_academic_pdf()
