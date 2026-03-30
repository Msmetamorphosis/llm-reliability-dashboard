"""
config.py
Central configuration. Paths resolve relative to this file
so the project works both locally and on Render.
"""

from pathlib import Path

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
LABELED   = DATA_DIR / "labeled"
CORPUS    = DATA_DIR / "corpus"

TASK1_DATASET = LABELED / "task1_extraction_dataset.json"
TASK2_DATASET = LABELED / "task2_qa_dataset.json"
CORPUS_FILE   = CORPUS  / "va_benefits_corpus.json"

MODEL_DEFAULT = "claude-opus-4-5"
MAX_TOKENS    = 1500
TOP_K         = 3

PROMPT_STYLES = ["structured", "ambiguous", "verbose", "casual"]

EXTRACTION_SCHEMA = {
    "type": "object",
    "required": [
        "benefit_name", "eligibility_criteria", "application_form",
        "application_method", "max_benefit_amount", "housing_allowance",
        "books_stipend_annual", "transferable_to_dependents",
        "transfer_requirements", "required_documents"
    ],
    "properties": {
        "benefit_name":               {"type": "string"},
        "eligibility_criteria":       {"type": "array", "items": {"type": "string"}},
        "application_form":           {"type": ["string", "null"]},
        "application_method":         {"type": ["string", "null"]},
        "max_benefit_amount":         {"type": ["number", "null"]},
        "housing_allowance":          {"type": ["string", "null"]},
        "books_stipend_annual":       {"type": ["number", "null"]},
        "transferable_to_dependents": {"type": "boolean"},
        "transfer_requirements":      {"type": ["string", "null"]},
        "required_documents":         {"type": "array", "items": {"type": "string"}}
    },
    "additionalProperties": False
}

QA_SCHEMA = {
    "type": "object",
    "required": ["answer", "confidence", "supported_by_context"],
    "properties": {
        "answer":               {"type": "string"},
        "confidence":           {"type": "string", "enum": ["high", "medium", "low"]},
        "supported_by_context": {"type": "boolean"}
    },
    "additionalProperties": False
}
