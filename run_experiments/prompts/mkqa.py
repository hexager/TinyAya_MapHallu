"""
prompts/mkqa.py — Prompt template and response_format for the MKQA QA task.

Asks the model to answer a question and return a JSON object ``{"answer": "..."}``.
Language instructions are appended so the model answers in the target language.
"""

MKQA_PROMPT_TEMPLATE = (
    '{query} You MUST output a valid JSON object with a single key "answer".{lang_instruction}'
)

MKQA_RESPONSE_FORMAT = {"type": "json_object"}

# Per-language instruction suffixes (MKQA queries are English-only).
LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "en": "",
    "es": " Por favor responde en español.",
    "hi": " कृपया हिंदी में उत्तर दें।",
    "ar": " يرجى الإجابة باللغة العربية.",
    "fr": " Veuillez répondre en français.",
    "de": " Bitte antworten Sie auf Deutsch.",
    "zh": " 请用中文回答。",
    "ja": " 日本語で答えてください。",
    "ko": " 한국어로 답변해 주세요.",
    "th": " กรุณาตอบเป็นภาษาไทย",
    "tr": " Lütfen Türkçe cevap verin.",
    "vi": " Vui lòng trả lời bằng tiếng Việt.",
    "ru": " Пожалуйста, ответьте на русском языке.",
    "pt": " Por favor, responda em português.",
    "it": " Per favore rispondi in italiano.",
    "nl": " Antwoord alstublieft in het Nederlands.",
    "pl": " Proszę odpowiedzieć po polsku.",
    "sw": " Tafadhali jibu kwa Kiswahili.",
    "he": " אנא ענה בעברית.",
    "da": " Svar venligst på dansk.",
    "fi": " Vastaa suomeksi.",
    "hu": " Kérjük, válaszoljon magyarul.",
    "no": " Vennligst svar på norsk.",
    "sv": " Vänligen svara på svenska.",
    "ms": " Sila jawab dalam Bahasa Melayu.",
    "km": " សូមឆ្លើយជាភាសាខ្មែរ។",
}


def build_mkqa_prompt(query: str, language: str = "en") -> str:
    lang_instr = LANGUAGE_INSTRUCTIONS.get(language, "")
    return MKQA_PROMPT_TEMPLATE.format(query=query, lang_instruction=lang_instr)
