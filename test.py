import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
PDF_DIR        = "./pdfs"          # folder with your PDFs
OUTPUT_JSONL   = "./output.jsonl"  # final output
API_KEY        = os.getenv("LLAMA_CLOUD_API_KEY")


# ─────────────────────────────────────────
# STEP 1 — Initialize Parser
# ─────────────────────────────────────────
def init_parser(mode: str = "markdown") -> LlamaParse:
    """
    result_type options:
        - "markdown"  → clean markdown with tables, headings
        - "text"      → plain text only
        - "json"      → structured JSON (premium feature)
    """
    parser = LlamaParse(
        api_key=API_KEY,
        result_type=mode,
        num_workers=4,              # parallel processing
        verbose=True,
        language="en",
        skip_diagonal_text=True,    # skip watermarks
        do_not_unroll_columns=True, # preserve column layout
        page_separator="\n---\n",   # separator between pages
    )
    print(f"[✓] Parser initialized in '{mode}' mode")
    return parser


# ─────────────────────────────────────────
# STEP 2 — Parse Single PDF
# ─────────────────────────────────────────
def parse_single_pdf(pdf_path: str, parser: LlamaParse) -> list[dict]:
    """
    Parse one PDF and return list of page records.
    Each record = one page of the PDF.
    """
    print(f"\n[→] Parsing: {pdf_path}")
    
    try:
        # LlamaParse returns list of Document objects (one per page)
        documents = parser.load_data(pdf_path)
        
        records = []
        for doc in documents:
            record = {
                "doc_id":    str(Path(pdf_path).stem),   # filename without extension
                "file_path": str(pdf_path),
                "page":      doc.metadata.get("page_label", "unknown"),
                "text":      doc.text,                    # extracted content
                "metadata":  doc.metadata,
                "char_count": len(doc.text),
            }
            records.append(record)
        
        print(f"[✓] Extracted {len(records)} pages from {Path(pdf_path).name}")
        return records

    except Exception as e:
        print(f"[✗] Failed to parse {pdf_path}: {e}")
        return []


# ─────────────────────────────────────────
# STEP 3 — Parse Full Directory of PDFs
# ─────────────────────────────────────────
def parse_pdf_directory(pdf_dir: str, parser: LlamaParse) -> list[dict]:
    """
    Loop through all PDFs in a directory and parse each one.
    Returns combined list of all page records.
    """
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    if not pdf_files:
        print(f"[!] No PDFs found in {pdf_dir}")
        return []
    
    print(f"\n[→] Found {len(pdf_files)} PDFs in {pdf_dir}")
    
    all_records = []
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        records = parse_single_pdf(str(pdf_path), parser)
        all_records.extend(records)
        
        # Small delay to avoid API rate limits
        if idx < len(pdf_files):
            time.sleep(1)
    
    print(f"\n[✓] Total records extracted: {len(all_records)}")
    return all_records


# ─────────────────────────────────────────
# STEP 4 — Write to JSONL
# ─────────────────────────────────────────
def write_jsonl(records: list[dict], output_path: str) -> None:
    """
    Write records to JSONL file.
    Each line = one JSON record (one page).
    """
    os.makedirs(Path(output_path).parent, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"\n[✓] Written {len(records)} records to {output_path}")


# ─────────────────────────────────────────
# STEP 5 — Verify Output
# ─────────────────────────────────────────
def verify_output(output_path: str, show_n: int = 2) -> None:
    """
    Read back JSONL and print first N records to verify.
    """
    print(f"\n{'='*50}")
    print(f"VERIFYING OUTPUT — first {show_n} records")
    print(f"{'='*50}")
    
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"Total lines in JSONL: {len(lines)}")
    
    for i, line in enumerate(lines[:show_n]):
        record = json.loads(line)
        print(f"\n--- Record {i+1} ---")
        print(f"  doc_id   : {record['doc_id']}")
        print(f"  page     : {record['page']}")
        print(f"  chars    : {record['char_count']}")
        print(f"  preview  : {record['text'][:200]}...")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    # 1. Init parser
    parser = init_parser(mode="markdown")
    
    # 2. Parse all PDFs
    records = parse_pdf_directory(PDF_DIR, parser)
    
    if not records:
        print("[!] No records extracted. Check your PDFs and API key.")
        return
    
    # 3. Write to JSONL
    write_jsonl(records, OUTPUT_JSONL)
    
    # 4. Verify
    verify_output(OUTPUT_JSONL)
    
    print("\n[✓] Done!")


if __name__ == "__main__":
    main()