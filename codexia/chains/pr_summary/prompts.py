from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from codedog.models import PRSummary
from codedog.templates import grimoire_en

parser = PydanticOutputParser(pydantic_object=PRSummary)

# 🔥 IMPORTANT: force string conversion
PR_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["metadata", "change_files", "code_summaries"],
    template=str(grimoire_en.PR_SUMMARY),   # ✅ FIX
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

CODE_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["name", "language", "content"],
    template=str(grimoire_en.CODE_SUMMARY),   # ✅ FIX
)
