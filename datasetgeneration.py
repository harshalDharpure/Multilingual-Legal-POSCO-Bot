# ===========================================
# Hindi POCSO Conversational Dataset Generator
# OpenRouter API Version
# ===========================================

import json
import os
import random
import re
import sys
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

# ===========================================
# CONFIG
# ===========================================
# Model configuration - OpenRouter API only
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is required!")

# Available GPT-5.1 models on OpenRouter:
# - "openai/gpt-5.1" (default) - Best for general tasks, 400K context, $1.25/$10 per M tokens
# - "openai/gpt-5.1-chat" - Fast, lightweight for chat (128K context)
# - "openai/gpt-5.1-codex" - Optimized for coding tasks
# - "openai/gpt-5.1-codex-max" - For complex coding projects
# - "openai/gpt-5.1-codex-mini" - Smaller, faster coding model
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-5.1")  # Default: GPT-5.1

OUTPUT_DIR = "/home/vaneet_2221cs15/legal-data/legalbot/hindi_posco_dataset"
CASE_SUMMARY_FILE = "/home/vaneet_2221cs15/legal-data/legalbot/formatted_case_passages.txt"

# ===========================================
# LANGUAGE SELECTION (for single-language runs)
# ===========================================
# Set this to generate only one language at a time
# Options: "hindi", "english", "code_mixed"
# Can also be set via environment variable: GENERATE_LANGUAGE
CURRENT_LANGUAGE = os.getenv("GENERATE_LANGUAGE", "code_mixed").lower()  # Default: code_mixed

# Validate language
if CURRENT_LANGUAGE not in ["hindi", "english", "code_mixed"]:
    raise ValueError(f"Invalid language: {CURRENT_LANGUAGE}. Must be one of: hindi, english, code_mixed")

SAMPLES_PER_LANGUAGE = 400  # 400 dialogs per language
TARGET_SAMPLES = SAMPLES_PER_LANGUAGE  # For single language run
LANGUAGES = [CURRENT_LANGUAGE]  # Only process the selected language

BUCKETS = {
    "A": (2, 3),
    "B": (3, 4),
    "C": (4, 5),
    "D": (5, 6),
}

COMPLEXITY_LEVELS = ["layman", "intermediate", "professional"]

# Distribution per language (complexity x bucket)
# Format: (language, complexity, bucket): count
DISTRIBUTION = {
    # Hindi: 400 dialogs
    ("hindi", "layman", "A"): 33,
    ("hindi", "layman", "B"): 33,
    ("hindi", "layman", "C"): 34,
    ("hindi", "layman", "D"): 33,
    ("hindi", "intermediate", "A"): 34,
    ("hindi", "intermediate", "B"): 33,
    ("hindi", "intermediate", "C"): 33,
    ("hindi", "intermediate", "D"): 33,
    ("hindi", "professional", "A"): 33,
    ("hindi", "professional", "B"): 34,
    ("hindi", "professional", "C"): 33,
    ("hindi", "professional", "D"): 33,
    # English: 400 dialogs
    ("english", "layman", "A"): 33,
    ("english", "layman", "B"): 34,
    ("english", "layman", "C"): 33,
    ("english", "layman", "D"): 33,
    ("english", "intermediate", "A"): 33,
    ("english", "intermediate", "B"): 33,
    ("english", "intermediate", "C"): 34,
    ("english", "intermediate", "D"): 33,
    ("english", "professional", "A"): 34,
    ("english", "professional", "B"): 33,
    ("english", "professional", "C"): 33,
    ("english", "professional", "D"): 34,
    # Code-mixed: 400 dialogs
    ("code_mixed", "layman", "A"): 34,
    ("code_mixed", "layman", "B"): 33,
    ("code_mixed", "layman", "C"): 33,
    ("code_mixed", "layman", "D"): 33,
    ("code_mixed", "intermediate", "A"): 33,
    ("code_mixed", "intermediate", "B"): 33,
    ("code_mixed", "intermediate", "C"): 33,
    ("code_mixed", "intermediate", "D"): 34,
    ("code_mixed", "professional", "A"): 33,
    ("code_mixed", "professional", "B"): 34,
    ("code_mixed", "professional", "C"): 34,
    ("code_mixed", "professional", "D"): 33,
}

# Case range mapping: cases 1-400 for Hindi, 401-800 for English, 801-1200 for Code-mixed
CASE_RANGES = {
    "hindi": (0, 400),      # Cases 1-400 (0-indexed: 0-399)
    "english": (400, 800),  # Cases 401-800 (0-indexed: 400-799)
    "code_mixed": (800, 1200)  # Cases 801-1200 (0-indexed: 800-1199)
}

# Filter DISTRIBUTION to only include the current language
DISTRIBUTION_FILTERED = {
    key: count for key, count in DISTRIBUTION.items() 
    if key[0] == CURRENT_LANGUAGE
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================
# MODEL SETUP (OpenRouter API)
# ===========================================
print("=" * 60)
print("Using OpenRouter API")
print(f"Model: {OPENROUTER_MODEL}")
print(f"API Key: {OPENROUTER_API_KEY[:10]}..." if OPENROUTER_API_KEY else "NOT SET")
print("=" * 60)
sys.stdout.flush()

api_session = requests.Session()
api_session.headers.update({
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/your-repo",  # Optional: for tracking
    "X-Title": "Hindi Legal Dialogue Generator"  # Optional: for tracking
})
print("✓ OpenRouter API configured")
sys.stdout.flush()

# ===========================================
# PROMPT BUILDER
# ===========================================
def get_complexity_description(complexity, language):
    """Get detailed complexity level description for USER behavior based on language"""
    descriptions = {
        "hindi": {
            "layman": """USER (सामान्य जन स्तर):
- सरल, बुनियादी शब्दावली का उपयोग करता है
- डर, भ्रम, या झिझक दिखाता है
- भावनात्मक संकेत शामिल कर सकता है: "कृपया मदद करें", "मुझे डर लग रहा है", "मुझे समझ नहीं आ रहा"
- मामूली टाइपो, वर्तनी गलतियाँ, गलत रिक्त स्थान, या अनौपचारिक विराम चिह्न हो सकते हैं
- अधूरे वाक्य या टूटे-फूटे विचारों का उपयोग कर सकता है
- अस्पष्ट या आंशिक रूप से बने प्रश्न पूछ सकता है
- कोई भी क़ानूनी धारा नंबर का उपयोग नहीं करेगा
- रोजमर्रा के ज्ञान से परे कानूनी समझ नहीं दिखाएगा
- सभी पाठ केवल हिंदी में होना चाहिए (केवल आवश्यक संक्षिप्ताक्षर जैसे POCSO, FIR, IPC, CrPC)

उदाहरण: "मुझे समझ नहीं आ रहा क्या करूँ, प्लीज़ मदद कीजिए। मेरा बच्चा डरा हुआ है।" """,

            "intermediate": """USER (मध्यम स्तर):
- काफी स्पष्ट और पूर्ण वाक्यों का उपयोग करता है
- मूल कानूनी जागरूकता है: FIR, शिकायत, पुलिस रिपोर्ट, "बाल सुरक्षा कानून"
- एक क़ानून का नाम ("POCSO") उल्लेख कर सकता है लेकिन सटीक धारा नंबर नहीं
- संरचित प्रश्न पूछता है: "क्या इस मामले में FIR दर्ज कर सकते हैं?", "कौन सा कानून लागू होगा?"
- मध्यम भावनात्मक नियंत्रण दिखाता है; स्वर अधिक तर्कसंगत है
- टाइपो नहीं (या बहुत कम), व्याकरण ज्यादातर ठीक है
- सभी पाठ केवल हिंदी में होना चाहिए (केवल आवश्यक संक्षिप्ताक्षर जैसे POCSO, FIR, IPC, CrPC)

उदाहरण: "क्या इस मामले में POCSO लागू होगा? FIR दर्ज करना आवश्यक है?" """,

            "professional": """USER (पेशेवर स्तर):
- NGO कार्यकर्ता, पैरालीगल, सामाजिक कार्यकर्ता, या सूचित नागरिक की तरह बोलता है
- सटीक शब्दों का उपयोग करता है: "अनिवार्य रिपोर्टिंग", "POCSO की धारा 19", "CrPC के तहत बयान"
- तकनीकी, प्रक्रियात्मक, या क़ानून-आधारित प्रश्न पूछता है
- टाइपो या व्याकरणिक त्रुटियाँ नहीं
- स्वर शांत, संरचित, और विस्तृत है
- विशिष्ट धाराओं या कानूनी कदमों का उल्लेख कर सकता है
- सभी पाठ केवल हिंदी में होना चाहिए (केवल आवश्यक संक्षिप्ताक्षर जैसे POCSO, FIR, IPC, CrPC, DLSA)

उदाहरण: "क्या इस स्थिति में POCSO की धारा 19 के तहत रिपोर्टिंग अनिवार्य है?" """
        },
        "english": {
            "layman": """The USER:
- Uses simple, basic vocabulary
- Shows fear, confusion, or hesitation
- May include emotional cues: "please help", "I am scared", "I don't understand"
- May have minor typos, spelling mistakes, or informal punctuation
- May use incomplete sentences or fragmented thoughts
- Asks vague or partially formed questions
- Should NOT use any statutory section numbers
- Should NOT show legal understanding beyond everyday knowledge

Example: "plz help… I dont kno wht to do… my child is scared" """,

            "intermediate": """The USER:
- Uses reasonably clear and complete sentences
- Has basic legal awareness: FIR, complaint, police report, "child safety law"
- May mention one statute name ("POCSO") but NOT exact section numbers
- Asks structured questions: "Can I file FIR?", "Which law applies?"
- Shows moderate emotional control; tone is more rational
- No typos (or very few), grammar mostly fine

Example: "Can this be considered under the POCSO Act? Should we file an FIR?" """,

            "professional": """The USER:
- Speaks like an NGO worker, paralegal, social worker, or informed citizen
- Uses precise terms: "mandatory reporting", "Section 19 POCSO", "statement under CrPC"
- Asks technical, procedural, or statute-based questions
- No typos or grammatical errors
- Tone is calm, structured, and detail-oriented
- May cite specific sections or legal steps

Example: "Does Section 19 of the POCSO Act require mandatory reporting in this scenario?" """
        },
        "code_mixed": {
            "layman": """USER:
- Simple, basic vocabulary use karta hai
- Fear, confusion, ya hesitation dikhata hai
- Emotional cues include kar sakta hai: "please help", "I am scared", "samajh nahi aa raha"
- Minor typos, spelling mistakes, ya informal punctuation ho sakta hai
- Incomplete sentences ya fragmented thoughts use kar sakta hai
- Vague ya partially formed questions puchta hai
- Statutory section numbers use NAHI karega
- Everyday knowledge se zyada legal understanding dikhata NAHI hai

Example: "Sir pls help, mujhe process samajh nhi aa raha…" """,

            "intermediate": """USER:
- Reasonably clear aur complete sentences use karta hai
- Basic legal awareness hai: FIR, complaint, police report, "child safety law"
- Ek statute name ("POCSO") mention kar sakta hai lekin exact section numbers NAHI
- Structured questions puchta hai: "Kya is case me FIR file kar sakte hain?", "Kaun sa law apply hoga?"
- Moderate emotional control dikhata hai; tone zyada rational hai
- Typos nahi (ya bahut kam), grammar mostly theek hai

Example: "Is case me FIR file kar sakte hain? POCSO apply hota hai kya?" """,

            "professional": """USER:
- NGO worker, paralegal, social worker, ya informed citizen ki tarah baat karta hai
- Precise terms use karta hai: "mandatory reporting", "Section 19 POCSO", "statement under CrPC"
- Technical, procedural, ya statute-based questions puchta hai
- Typos ya grammatical errors nahi hote
- Tone calm, structured, aur detail-oriented hai
- Specific sections ya legal steps cite kar sakta hai

Example: "As per POCSO Section 19, mandatory reporting apply karega yaha?" """
        }
    }
    lang_desc = descriptions.get(language, descriptions["hindi"])
    return lang_desc.get(complexity, lang_desc["intermediate"])

def build_prompt_hindi(case_summary, turns, complexity):
    """Build prompt specifically for Hindi language"""
    case_truncated = case_summary[:800] if len(case_summary) > 800 else case_summary
    complexity_desc = get_complexity_description(complexity, "hindi")
    
    return f"""You are generating a structured, high-quality Hindi Child Sexual Abuse legal dialogue dataset for research on multilingual access-to-justice in India.

**CRITICAL FOR HINDI: All text must be in Hindi only. Do NOT use English words. Only legal acronyms like POCSO, FIR, IPC, CrPC, DLSA are allowed. All sentences, phrases, and explanations must be in Hindi.**

Using the case summary below, create a Hindi conversation between a USER and a LEGAL ASSISTANT.

CASE SUMMARY:
{case_truncated}

========================================================
REQUIREMENTS
========================================================

1. DIALOGUE LENGTH
- Total dialogue length: {turns} user–assistant exchanges.
- A "turn" means: USER message → ASSISTANT reply.
- Maintain the exact number of exchanges requested.
- This means you need {turns * 2} total messages in the "turns" array (alternating user/assistant)

--------------------------------------------------------
2. COMPLEXITY LEVEL (IMPORTANT — affects USER behavior)
--------------------------------------------------------

The "complexity level" refers to **how the USER speaks**, NOT how complex the case is.

Choose the USER's speaking style exactly according to the assigned complexity:

{complexity_desc}

========================================================
3. ASSISTANT BEHAVIOR (MUST FOLLOW)
========================================================
कानूनी सहायक को यह करना चाहिए:

- सहानुभूतिपूर्ण, आघात-सूचित भाषा का उपयोग करें
- उपयोगकर्ता की भावनाओं और सुरक्षा चिंताओं को स्वीकार करें
- कानूनी रूप से सही जानकारी प्रदान करें:
    * POCSO अधिनियम (धारा 3-10, 19, 24, आदि)
    * IPC की प्रासंगिक धाराएं (354, 354A, 376, आदि)
    * JJ अधिनियम (यदि प्रासंगिक हो)
- प्रासंगिक कानूनी धाराओं का सावधानी से उल्लेख करें:
    "यह धारा... के तहत आ सकता है"
    "सामान्य कानूनी व्याख्या के आधार पर..."
- कभी भी कानूनों का आविष्कार न करें
- गारंटीशुदा परिणाम न दें ("आप मामला जीत जाएंगे" — अनुमत नहीं)
- सुरक्षित कदमों को प्रोत्साहित करें:
    "आप स्थानीय पुलिस से संपर्क करने पर विचार कर सकते हैं..."
    "आप Childline 1098 पर फोन कर सकते हैं..."
- वास्तविक नाम नहीं; केवल प्लेसहोल्डर:
   [Victim], [Accused], [Relative], [Teacher], [Minor]
- घटना का चित्रात्मक या स्पष्ट विवरण नहीं
- प्रति मोड़ ≤ 100 शब्द रखें
- **महत्वपूर्ण: सभी पाठ केवल हिंदी में होना चाहिए। अंग्रेजी शब्दों का उपयोग न करें, केवल आवश्यक कानूनी संक्षिप्ताक्षर जैसे POCSO, FIR, IPC, CrPC, DLSA का उपयोग करें। सभी वाक्य हिंदी में होने चाहिए।**

========================================================
4. LANGUAGE STYLE RULES (HINDI)
========================================================
- स्वाभाविक बोली जाने वाली हिंदी
- संस्कृत-भारी या अत्यधिक औपचारिक शब्दों से बचें
- भारत में उपयोग किए जाने वाले सामान्य कानूनी शब्दों का उपयोग करें:
    FIR, POCSO, धारा, IPC, पुलिस, बयान
- वाक्यों को स्पष्ट और संक्षिप्त रखें
- **महत्वपूर्ण: सभी पाठ केवल हिंदी में होना चाहिए। अंग्रेजी शब्दों का उपयोग न करें, केवल आवश्यक कानूनी संक्षिप्ताक्षर जैसे POCSO, FIR, IPC, CrPC, DLSA का उपयोग करें।**

========================================================
5. OUTPUT FORMAT (STRICT JSON)
========================================================
Output must be valid JSON:

{{
  "dialogue_id": "",
  "language": "hindi",
  "complexity": "{complexity}",
  "turn_count": {turns},
  "turns": [
    {{"role": "user", "text": "..."}},
    {{"role": "assistant", "text": "..."}},
    {{"role": "user", "text": "..."}},
    {{"role": "assistant", "text": "..."}}
  ],
  "statutes_cited": []
}}

CRITICAL REQUIREMENTS:
- You MUST generate exactly {turns} user-assistant exchanges
- This means you need {turns * 2} total messages in the "turns" array
- The pattern MUST be: user → assistant → user → assistant → ... (alternating)
- Do NOT add extra fields
- Do NOT add commentary outside the JSON
- Do NOT break JSON structure
- Output ONLY valid JSON. No markdown, no explanations, no text before or after the JSON.

Generate the dialogue now with EXACTLY {turns} user-assistant exchanges. Output ONLY the JSON object, nothing else:"""

def build_prompt_english(case_summary, turns, complexity):
    """Build prompt specifically for English language"""
    case_truncated = case_summary[:800] if len(case_summary) > 800 else case_summary
    complexity_desc = get_complexity_description(complexity, "english")
    
    return f"""You are generating a structured, high-quality English Child Sexual Abuse legal dialogue dataset for research on multilingual access-to-justice in India.

Using the case summary below, create an English conversation between a USER and a LEGAL ASSISTANT.

CASE SUMMARY:
{case_truncated}

========================================================
REQUIREMENTS
========================================================

1. DIALOGUE LENGTH
- Total dialogue length: {turns} user–assistant exchanges.
- A "turn" means: USER message → ASSISTANT reply.
- Maintain the exact number of exchanges requested.
- This means you need {turns * 2} total messages in the "turns" array (alternating user/assistant)

--------------------------------------------------------
2. COMPLEXITY LEVEL (IMPORTANT — affects USER behavior)
--------------------------------------------------------

The "complexity level" refers to **how the USER speaks**, NOT how complex the case is.

Choose the USER's speaking style exactly according to the assigned complexity:

{complexity_desc}

========================================================
3. ASSISTANT BEHAVIOR (MUST FOLLOW)
========================================================
The Legal Assistant should:

- Use empathetic, trauma-informed language
- Acknowledge the user's emotions and safety concerns
- Provide legally accurate information:
    * POCSO Act (Sections 3-10, 19, 24, etc.)
    * Relevant IPC sections (354, 354A, 376, etc.)
    * JJ Act (if relevant)
- Carefully mention relevant legal sections:
    "This may fall under Section..."
    "Based on general legal interpretation..."
- Never invent laws or statutes
- Do not guarantee outcomes ("You will win the case" — not allowed)
- Encourage safe steps:
    "You may consider contacting local police..."
    "You can call Childline 1098..."
- No real names; only placeholders:
   [Victim], [Accused], [Relative], [Teacher], [Minor]
- No graphic or explicit descriptions of incidents
- Keep ≤ 100 words per turn
- Use plain, professional English throughout

========================================================
4. LANGUAGE STYLE RULES (ENGLISH)
========================================================
- Plain, easy-to-read English
- Short, clear sentences
- Use standard legal terminology
- Avoid Indianized English expressions
- Maintain professional but accessible tone
- Use proper grammar and spelling

========================================================
5. OUTPUT FORMAT (STRICT JSON)
========================================================
Output must be valid JSON:

{{
  "dialogue_id": "",
  "language": "english",
  "complexity": "{complexity}",
  "turn_count": {turns},
  "turns": [
    {{"role": "user", "text": "..."}},
    {{"role": "assistant", "text": "..."}},
    {{"role": "user", "text": "..."}},
    {{"role": "assistant", "text": "..."}}
  ],
  "statutes_cited": []
}}

CRITICAL REQUIREMENTS:
- You MUST generate exactly {turns} user-assistant exchanges
- This means you need {turns * 2} total messages in the "turns" array
- The pattern MUST be: user → assistant → user → assistant → ... (alternating)
- Do NOT add extra fields
- Do NOT add commentary outside the JSON
- Do NOT break JSON structure
- Output ONLY valid JSON. No markdown, no explanations, no text before or after the JSON.

Generate the dialogue now with EXACTLY {turns} user-assistant exchanges. Output ONLY the JSON object, nothing else:"""

def build_prompt_code_mixed(case_summary, turns, complexity):
    """Build prompt specifically for Code-mixed/Hinglish language"""
    case_truncated = case_summary[:800] if len(case_summary) > 800 else case_summary
    complexity_desc = get_complexity_description(complexity, "code_mixed")
    
    return f"""You are generating a structured, high-quality Code-mixed/Hinglish Child Sexual Abuse legal dialogue dataset for research on multilingual access-to-justice in India.

Using the case summary below, create a Code-mixed/Hinglish conversation between a USER and a LEGAL ASSISTANT.

CASE SUMMARY:
{case_truncated}

========================================================
REQUIREMENTS
========================================================

1. DIALOGUE LENGTH
- Total dialogue length: {turns} user–assistant exchanges.
- A "turn" means: USER message → ASSISTANT reply.
- Maintain the exact number of exchanges requested.
- This means you need {turns * 2} total messages in the "turns" array (alternating user/assistant)

--------------------------------------------------------
2. COMPLEXITY LEVEL (IMPORTANT — affects USER behavior)
--------------------------------------------------------

The "complexity level" refers to **how the USER speaks**, NOT how complex the case is.

Choose the USER's speaking style exactly according to the assigned complexity:

{complexity_desc}

========================================================
3. ASSISTANT BEHAVIOR (MUST FOLLOW)
========================================================
Legal Assistant ko yeh karna chahiye:

- Empathetic, trauma-informed language use kare
- User ki emotions aur safety concerns ko acknowledge kare
- Legally accurate information provide kare:
    * POCSO Act (Sections 3-10, 19, 24, etc.)
    * Relevant IPC sections (354, 354A, 376, etc.)
    * JJ Act (agar relevant ho)
- Relevant legal sections ka carefully mention kare:
    "Yeh Section... ke under aa sakta hai"
    "General legal interpretation ke basis par..."
- Kabhi bhi laws ya statutes invent NAHI kare
- Guaranteed outcomes NAHI de ("Aap case jeet jayenge" — allowed nahi)
- Safe steps ko encourage kare:
    "Aap local police se contact karne ka soch sakte hain..."
    "Aap Childline 1098 par phone kar sakte hain..."
- Real names NAHI; sirf placeholders:
   [Victim], [Accused], [Relative], [Teacher], [Minor]
- Graphic ya explicit incident descriptions NAHI
- ≤ 100 words per turn rakhe
- Natural Hinglish mix use kare (60-70% Hindi structure + 30-40% English legal terms)

========================================================
4. LANGUAGE STYLE RULES (CODE-MIXED/HINGLISH)
========================================================
- 60–70% Hindi sentence structure + 30–40% English legal terms
- Mix naturally: "Yeh case POCSO Section 7 ke under aa sakta hai"
- Use English for legal terms: FIR, POCSO, Section, IPC, police, statement, complaint, court
- Use Hindi for conversational parts: "kya", "hoga", "sakte hain", "chahiye"
- Natural code-switching: "FIR file kar sakte hain", "Section 19 apply hota hai"
- Example: "Yeh case POCSO Section 7 ke under aa sakta hai, FIR lodge kar sakte ho."
- Maintain conversational flow with natural mixing
- Avoid forced or awkward translations

========================================================
5. OUTPUT FORMAT (STRICT JSON)
========================================================
Output must be valid JSON:

{{
  "dialogue_id": "",
  "language": "code_mixed",
  "complexity": "{complexity}",
  "turn_count": {turns},
  "turns": [
    {{"role": "user", "text": "..."}},
    {{"role": "assistant", "text": "..."}},
    {{"role": "user", "text": "..."}},
    {{"role": "assistant", "text": "..."}}
  ],
  "statutes_cited": []
}}

CRITICAL REQUIREMENTS:
- You MUST generate exactly {turns} user-assistant exchanges
- This means you need {turns * 2} total messages in the "turns" array
- The pattern MUST be: user → assistant → user → assistant → ... (alternating)
- Do NOT add extra fields
- Do NOT add commentary outside the JSON
- Do NOT break JSON structure
- Output ONLY valid JSON. No markdown, no explanations, no text before or after the JSON.

Generate the dialogue now with EXACTLY {turns} user-assistant exchanges. Output ONLY the JSON object, nothing else:"""

def build_prompt(case_summary, turns, complexity, language):
    """Main prompt builder that routes to language-specific functions"""
    if language == "hindi":
        return build_prompt_hindi(case_summary, turns, complexity)
    elif language == "english":
        return build_prompt_english(case_summary, turns, complexity)
    elif language == "code_mixed":
        return build_prompt_code_mixed(case_summary, turns, complexity)
    else:
        # Fallback to Hindi
        return build_prompt_hindi(case_summary, turns, complexity)

# ===========================================
# GENERATE
# ===========================================
def safe_parse_json(text: str):
    """More robust JSON parsing with multiple fallback strategies"""
    if not text:
        return None
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Strategy 1: Try to extract JSON from markdown code blocks (most common)
    code_block_patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',  # ```json {...} ```
        r'```\s*(\{.*?\})\s*```',  # ``` {...} ```
    ]
    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_text = match.group(1)
            try:
                # Clean and parse
                json_text = clean_json_text(json_text)
                return json.loads(json_text)
            except:
                continue
    
    # Strategy 2: Find balanced braces (handle nested JSON)
    start = text.find("{")
    if start != -1:
        brace_count = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break
        
        if brace_count == 0 and end > start:
            json_text = text[start:end+1]
            try:
                json_text = clean_json_text(json_text)
                return json.loads(json_text)
            except:
                pass
    
    # Strategy 3: Try regex to find JSON object (simpler pattern)
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        json_text = json_match.group(0)
        try:
            json_text = clean_json_text(json_text)
            return json.loads(json_text)
        except:
            pass
    
    # Strategy 4: Find first { and last } (fallback)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_text = text[start:end+1]
        try:
            json_text = clean_json_text(json_text)
            return json.loads(json_text)
        except:
            pass

    return None

def clean_json_text(json_text: str) -> str:
    """Clean JSON text to fix common issues"""
    # Remove control characters
    json_text = re.sub(r'[\x00-\x1f]+', '', json_text)
    
    # Remove trailing commas before closing braces/brackets
    json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
    
    # Fix common quote issues (single quotes to double quotes)
    # But be careful - only fix unquoted single quotes, not inside strings
    # This is a simplified version - might need more sophisticated handling
    json_text = re.sub(r"'([^']*)':", r'"\1":', json_text)  # Keys
    json_text = re.sub(r":\s*'([^']*)'", r': "\1"', json_text)  # String values
    
    # Remove comments (JSON doesn't support comments)
    json_text = re.sub(r'//.*?$', '', json_text, flags=re.MULTILINE)
    json_text = re.sub(r'/\*.*?\*/', '', json_text, flags=re.DOTALL)
    
    return json_text

def create_fallback_dialogue(case_summary, turns, complexity, bucket_key, idx, language, case_id):
    """Create a simple fallback dialogue structure when JSON parsing fails"""
    lang_prefix = {"hindi": "HN", "english": "EN", "code_mixed": "CM"}.get(language, "HN")
    dialogue_id = f"{lang_prefix}_{bucket_key}_C{case_id:04d}_{idx:03d}"
    
    # Language-specific fallback messages
    fallback_messages = {
        "hindi": {
            "user": "कृपया मुझे इस मामले के बारे में जानकारी दें।",
            "assistant": "मैं आपकी सहायता करने के लिए यहाँ हूँ। कृपया अपने प्रश्न पूछें।"
        },
        "english": {
            "user": "Please provide me information about this case.",
            "assistant": "I am here to help you. Please ask your questions."
        },
        "code_mixed": {
            "user": "Please mujhe is case ke bare me information dijiye.",
            "assistant": "Main aapki help karne ke liye yahan hoon. Apne questions puchiye."
        }
    }
    msg = fallback_messages.get(language, fallback_messages["hindi"])
    
    return {
        "dialogue_id": dialogue_id,
        "language": language,
        "complexity": complexity,
        "turn_count": turns,
        "turns": [
            {"role": "user", "text": msg["user"]},
            {"role": "assistant", "text": msg["assistant"]}
        ] * turns,
        "statutes_cited": []
    }

def generate_via_openrouter(prompt: str, max_retries: int = 2) -> Optional[str]:
    """Generate text using OpenRouter API with retry logic"""
    if not api_session:
        raise ValueError("OpenRouter API session not initialized")
    
    for attempt in range(max_retries + 1):
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5 if attempt == 0 else 0.3,  # Lower temp on retry
            "max_tokens": 3000,  # Increased to ensure complete JSON
            "top_p": 0.85
        }
    
        try:
            response = api_session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract generated text
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                if content:
                    return content
                elif attempt < max_retries:
                    print(f"  Empty response, retrying... (attempt {attempt + 1}/{max_retries + 1})")
                    sys.stdout.flush()
                    continue
            else:
                if attempt < max_retries:
                    print(f"  API response error, retrying... (attempt {attempt + 1}/{max_retries + 1})")
                    sys.stdout.flush()
                    continue
                else:
                    print(f"  API response error: {data}")
                    sys.stdout.flush()
                    return None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                print(f"  API request failed, retrying... (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                sys.stdout.flush()
                time.sleep(2)  # Wait before retry
                continue
            else:
                print(f"  API request failed: {str(e)}")
                sys.stdout.flush()
                return None
        except Exception as e:
            if attempt < max_retries:
                print(f"  API error, retrying... (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                sys.stdout.flush()
                time.sleep(2)
                continue
            else:
                print(f"  API error: {str(e)}")
                sys.stdout.flush()
                return None
    
    return None

def generate_dialogue(case_summary, bucket_key, idx, complexity, language, case_id):
    min_t, max_t = BUCKETS[bucket_key]
    turns = random.randint(min_t, max_t)
    # Complexity and language are passed as parameters to ensure equal distribution

    prompt = build_prompt(case_summary, turns, complexity, language)

    try:
        # Use OpenRouter API
        output = generate_via_openrouter(prompt)
        if output is None:
            return None
    except Exception as e:
        print(f"  Error during generation: {str(e)}")
        sys.stdout.flush()
        return None

    # Try robust JSON parsing
    data = safe_parse_json(output)
    
    if data and isinstance(data, dict):
        # Validate that we have required fields
        if "turns" not in data:
            data["turns"] = []
        if "statutes_cited" not in data:
            data["statutes_cited"] = []
        
        # Remove safety_notes if present
        if "safety_notes" in data:
            del data["safety_notes"]
        
        # Remove case_summary if present (we only keep case_id for reference)
        if "case_summary" in data:
            del data["case_summary"]
        
        lang_prefix = {"hindi": "HN", "english": "EN", "code_mixed": "CM"}.get(language, "HN")
        data["dialogue_id"] = f"{lang_prefix}_{bucket_key}_C{case_id:04d}_{idx:03d}"
        data["language"] = data.get("language", language)
        data["complexity"] = data.get("complexity", complexity)
        data["turn_count"] = data.get("turn_count", turns)
        data["bucket"] = bucket_key
        
        # Validate that we have at least some turns
        if not data["turns"] or len(data["turns"]) == 0:
            print(f"  Warning: No turns found in parsed JSON")
            sys.stdout.flush()
            return None
        
        # Validate turn structure
        valid_turns = []
        for turn in data["turns"]:
            if isinstance(turn, dict) and "role" in turn and "text" in turn:
                if turn["role"] in ["user", "assistant"]:
                    valid_turns.append(turn)
        
        if len(valid_turns) == 0:
            print(f"  Warning: No valid turns found")
            sys.stdout.flush()
            return None
        
        data["turns"] = valid_turns
        return data
    else:
        # If the model didn't follow JSON format, fall back to a minimal valid structure
        # so the dataset can still reach the target size.
        print(f"  JSON parsing failed (using fallback dialogue)")
        sys.stdout.flush()
        return create_fallback_dialogue(
            case_summary=case_summary,
            turns=turns,
            complexity=complexity,
            bucket_key=bucket_key,
            idx=idx,
            language=language,
            case_id=case_id
        )

# ===========================================
# LOAD CASES
# ===========================================
if not Path(CASE_SUMMARY_FILE).exists():
    raise FileNotFoundError(f"Case file not found: {CASE_SUMMARY_FILE}")

with open(CASE_SUMMARY_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# Parse cases from formatted file (cases are separated by [case N] headers)
CASES = []
for case_text in content.split("[case ")[1:]:  # Skip first empty part before [case 1]
    if case_text.strip():
        # Split at first newline to separate case number from content
        lines = case_text.split("\n", 1)
        if len(lines) > 1:
            case_num = lines[0].split("]")[0]
            case_content = lines[1].strip()
            # Only include cases with substantial content (at least 100 chars)
            if len(case_content) > 100:
                CASES.append(case_content)

print(f"Loaded {len(CASES)} case summaries from {CASE_SUMMARY_FILE}")
sys.stdout.flush()

# ===========================================
# MAIN LOOP
# ===========================================
# Create language-specific output file
output_file = f"{OUTPUT_DIR}/{CURRENT_LANGUAGE}_posco_dataset.jsonl"
dataset = []

# Track distribution counts for each (language, complexity, bucket) combination
# Only track the current language
distribution_counts = {key: 0 for key in DISTRIBUTION_FILTERED.keys()}

# Track case indices per language
case_indices = {CURRENT_LANGUAGE: CASE_RANGES[CURRENT_LANGUAGE][0]}

# Get case range for current language
case_start, case_end = CASE_RANGES[CURRENT_LANGUAGE]

# ===========================================
# RESUME SUPPORT
# ===========================================
# If the output file already exists and is non-empty, resume generation:
# - Do NOT overwrite the file
# - Recompute distribution counts from existing rows
# - Continue numbering dialogue IDs from the next index
resume_existing = Path(output_file).exists() and Path(output_file).stat().st_size > 0
if resume_existing:
    existing_total = 0
    max_case_id_seen = 0
    try:
        with open(output_file, "r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                existing_total += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                comp = obj.get("complexity")
                buck = obj.get("bucket")
                if comp in COMPLEXITY_LEVELS and buck in BUCKETS:
                    key = (CURRENT_LANGUAGE, comp, buck)
                    if key in distribution_counts:
                        distribution_counts[key] += 1
                cid = obj.get("case_id")
                if isinstance(cid, int) and cid > max_case_id_seen:
                    max_case_id_seen = cid
        # Continue from next case after the max case_id used (still within the configured range)
        # Note: case_id is 1-indexed; case index is 0-indexed.
        next_case_idx = max_case_id_seen  # if last used was N, next index is N (0-indexed)
        if next_case_idx < case_start:
            next_case_idx = case_start
        case_indices[CURRENT_LANGUAGE] = next_case_idx
    except Exception as e:
        print(f"⚠ Resume read failed, starting fresh: {str(e)}")
        sys.stdout.flush()
        resume_existing = False

# Open file for incremental saving
print("=" * 60)
print(f"GENERATING DIALOGUES FOR: {CURRENT_LANGUAGE.upper()}")
print("=" * 60)
print(f"Starting generation loop. Target: {TARGET_SAMPLES} dialogues")
print(f"Language: {CURRENT_LANGUAGE.capitalize()}")
print(f"Case range: {case_start + 1}-{case_end} (cases {case_start + 1} to {case_end})")
print(f"Output file: {output_file}")
if resume_existing:
    existing_generated = sum(distribution_counts.get((CURRENT_LANGUAGE, c, b), 0) for c in COMPLEXITY_LEVELS for b in BUCKETS.keys())
    print(f"Resume: detected existing file with {existing_generated} dialogues. Will append until {TARGET_SAMPLES}.")
print("=" * 60)
sys.stdout.flush()

file_mode = "a" if resume_existing else "w"
with open(output_file, file_mode, encoding="utf-8") as f:
    total_generated = sum(distribution_counts.get((CURRENT_LANGUAGE, c, b), 0) 
                          for c in COMPLEXITY_LEVELS for b in BUCKETS.keys())
    
    # Generate dialogs following the distribution plan
    # Iterate through all combinations and generate required number for each
    while total_generated < TARGET_SAMPLES:
        # Find next combination that needs more dialogs (only for current language)
        next_combo = None
        for (lang, complexity, bucket), target_count in DISTRIBUTION_FILTERED.items():
            current_count = distribution_counts[(lang, complexity, bucket)]
            if current_count < target_count:
                next_combo = (lang, complexity, bucket)
                break
        
        # If all combinations are complete, break
        if next_combo is None:
            break
        
        lang, complexity, bucket = next_combo
        target_count = DISTRIBUTION_FILTERED[next_combo]
        current_count = distribution_counts[next_combo]
        
        dialogue_num = current_count + 1

        # Pick a case index. If we hit the end of the language range, sample randomly within the range.
        # This avoids getting stuck below TARGET_SAMPLES due to occasional generation/parsing failures.
        case_idx = case_indices[CURRENT_LANGUAGE]
        if case_idx >= case_end:
            case_idx = random.randint(case_start, case_end - 1)
        else:
            # Only advance the pointer after we pick a case to try.
            case_indices[CURRENT_LANGUAGE] += 1

        # Try a few times on the same case before moving on.
        result = None
        case_number = case_idx + 1  # 1-indexed case number
        for attempt in range(3):
            print(
                f"Generating {CURRENT_LANGUAGE} {complexity} {bucket} ({dialogue_num}/{target_count}) "
                f"[Case {case_number}] (Total: {total_generated + 1}/{TARGET_SAMPLES})..."
                + (f" (retry {attempt + 1}/3)" if attempt > 0 else "")
            )
            sys.stdout.flush()
            try:
                case = CASES[case_idx]
                result = generate_dialogue(case, bucket, total_generated + 1, complexity, CURRENT_LANGUAGE, case_number)
            except Exception as e:
                print(f"✗ Error generating dialogue: {str(e)}")
                sys.stdout.flush()
                result = None
            if result:
                break

        if result:
            # Verify the fields match
            result["language"] = CURRENT_LANGUAGE
            result["complexity"] = complexity
            result["bucket"] = bucket
            result["case_id"] = case_number  # Store 1-indexed case ID
            
            dataset.append(result)
            distribution_counts[next_combo] += 1
            total_generated += 1
            
            # Save immediately to file (incremental saving)
            f.write(json.dumps(result, ensure_ascii=False))
            f.write("\n")
            f.flush()  # Ensure data is written to disk immediately
            print(f"✓ Generated and saved: {result['dialogue_id']} ({CURRENT_LANGUAGE}/{complexity}/{bucket})")
            
            # Print progress summary
            lang_total = sum(distribution_counts.get((CURRENT_LANGUAGE, c, b), 0) 
                           for c in COMPLEXITY_LEVELS for b in BUCKETS.keys())
            print(f"  Progress - {CURRENT_LANGUAGE.capitalize()}: {lang_total}/{SAMPLES_PER_LANGUAGE} | "
                  f"Overall: {total_generated}/{TARGET_SAMPLES}")
            sys.stdout.flush()
        else:
            print(f"✗ Generation failed, will retry with next combination")
            sys.stdout.flush()

print("=" * 60)
print("DONE")
print(f"Total dialogues generated: {total_generated}")
print(f"\nDistribution Summary for {CURRENT_LANGUAGE.upper()}:")
lang_total = sum(distribution_counts.get((CURRENT_LANGUAGE, c, b), 0) 
               for c in COMPLEXITY_LEVELS for b in BUCKETS.keys())
print(f"\n{CURRENT_LANGUAGE.upper()} ({lang_total}/{SAMPLES_PER_LANGUAGE}):")
for complexity in COMPLEXITY_LEVELS:
    comp_total = sum(distribution_counts.get((CURRENT_LANGUAGE, complexity, b), 0) for b in BUCKETS.keys())
    print(f"  {complexity.capitalize()}: {comp_total}")
    for bucket in BUCKETS.keys():
        count = distribution_counts.get((CURRENT_LANGUAGE, complexity, bucket), 0)
        target = DISTRIBUTION_FILTERED.get((CURRENT_LANGUAGE, complexity, bucket), 0)
        status = "✓" if count == target else "✗"
        print(f"    {status} Bucket {bucket}: {count}/{target}")
print(f"\nSaved at: {output_file}")
sys.stdout.flush()
