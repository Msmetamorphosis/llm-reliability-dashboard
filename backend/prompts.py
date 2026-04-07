"""
prompts.py  -  Four behavioral prompt wrappers for Task 1 and Task 2.
"""

SCHEMA_T1 = """Return ONLY a JSON object with exactly these fields:
benefit_name (string), eligibility_criteria (array of strings),
application_form (string or null), application_method (string or null),
max_benefit_amount (number or null), housing_allowance (string or null),
books_stipend_annual (number or null), transferable_to_dependents (boolean),
transfer_requirements (string or null), required_documents (array of strings).
No text outside the JSON object."""

SCHEMA_QA = """Return ONLY a JSON object:
{"answer": string, "confidence": "high"|"medium"|"low", "supported_by_context": boolean}
No text outside the JSON object."""


def task1_prompt(document: str, style: str) -> str:
    if style == "structured":
        return f"Extract structured information from the following VA benefits document.\n\nDocument:\n{document}\n\n{SCHEMA_T1}"
    if style == "ambiguous":
        return f"Look at this document and pull out the relevant information:\n\n{document}\n\nGive me the details in JSON."
    if style == "verbose":
        return (
            f"Hi, I have a VA benefits document I need help processing for our veteran services "
            f"system. It is really important that we capture all information correctly so that "
            f"veterans do not miss out on benefits they are entitled to. Here is the document:\n\n"
            f"{document}\n\nCan you help me extract all key information as JSON with the benefit "
            f"name, who qualifies, how to apply, and any dollar amounts? Please return properly "
            f"formatted JSON."
        )
    if style == "casual":
        return f"hey can u just grab the key info from this benefits doc and put it in json?\n\n{document}\n\nthx"
    raise ValueError(f"Unknown style: {style}")


def task2_prompt(question: str, context_passages: list[str], style: str) -> str:
    context = "\n\n".join(
        f"Passage {i+1}:\n{p}" for i, p in enumerate(context_passages)
    )
    if style == "structured":
        return (
            f"Answer the following question using only the provided context passages.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\n{SCHEMA_QA}"
        )
    if style == "ambiguous":
        return f"Based on the info below, answer this:\n\n{context}\n\nQ: {question}\n\nJSON response."
    if style == "verbose":
        return (
            f"I am building a veteran benefits assistant and need accurate answers grounded "
            f"strictly in official source material. Veterans depend on this information to "
            f"make important decisions. Here is the context:\n\n{context}\n\n"
            f"The question is: {question}\n\nPlease answer strictly from context and return "
            f"JSON with answer, confidence, and supported_by_context."
        )
    if style == "casual":
        return f"quick question, just use these passages:\n\n{context}\n\nquestion: {question}\n\njson pls"
    raise ValueError(f"Unknown style: {style}")
