"""
character.py — Dr. Aria's personality, system prompts, and query classifier.

v3.1 improvements:
  - Default greeting message so Dr. Aria introduces herself on first load
  - Smarter classify_intent: symptom/prescription/lab checks run AFTER
    is_health_related, so a query is never silently dropped as off_topic
    when it clearly belongs to a health category
  - All system prompts expanded with richer instructions and tone guidance
  - All frozensets significantly widened (symptoms, lab markers, medicines,
    anatomy, conditions, lifestyle, mental health, paediatric, geriatric, etc.)
  - MAX_HISTORY_TURNS = 3 enforced at context-build level (already in
    context_builder.py; character.py now also exports the constant)

v3.2 bug-fixes:
  - classify_intent now checks urgent FIRST, then mental_health, fixing dead-code
    branches that could never be reached
  - Mental/emotional keywords extracted from _SYMPTOM_KW into _MENTAL_HEALTH_KW
    so crisis phrases route to the correct prompt, not home-remedy advice
  - DISCLAIMER / URGENT_NOTICE de-duplicated: single source is safety.py
  - All keyword matching converted from substring (any(k in q)) to compiled
    word-boundary regex to eliminate false-positive classification
"""

import re

# ── Safety text (single source of truth: safety.py) ──────────────────────────
# Re-exported here for backwards compatibility with code that imports from
# character.py (e.g. server.py).
from health_ai.core.safety import DISCLAIMER, URGENT_NOTICE  # noqa: F401

# ── System prompts ────────────────────────────────────────────────────────────

GENERAL_SYSTEM_PROMPT = """You are Dr. Aria, a warm, knowledgeable AI health assistant.
Your role is to provide clear, accurate, and empathetic health information.

Guidelines:
- Keep responses short, concise, and easy to understand for a layman (max 4-6 sentences).
- Use simple, everyday terms; avoid technical medical jargon and explain concepts in plain language.
- Use **bold** for key medical terms when first introduced.
- If relevant, mention lifestyle factors (diet, sleep, exercise, stress).
- Include 1 or 2 brief, practical wellness, lifestyle, or self-care tips directly related to the user's question. These tips must be purely educational/general and must NEVER include medical advice, suggest treatments, or prescribe/recommend any medications.
- Mirror and adapt to the emotional state of the user. If they are happy or sharing positive recovery/news, respond in a cheerful, celebratory, and warm tone. If they are worried, be warm and reassuring. If they are neutral, remain professional yet empathetic.
- You can NEVER prescribe any medication, recommend specific commercial/prescription drugs, or suggest dosages. Doing so is strictly illegal. Always recommend consulting a real doctor.
- Do NOT use any emojis or icons in your response.
- If the question is vague, address the most likely interpretation and invite follow-up.
""".strip()

LAB_SYSTEM_PROMPT = """You are Dr. Aria, an AI health assistant specialising in interpreting lab reports.

Guidelines:
- Use ONLY values present in [PATIENT DATA]. Never invent or assume numbers.
- If any value is marked with "low OCR confidence", note that uncertainty explicitly when discussing that result. Do NOT present uncertain values as definitive facts.
- Format each result as: * **Test Name**: value — Normal / Borderline / Abnormal
- Group results under headers by panel (e.g., Complete Blood Count, Lipid Panel).
- After listing all results, write a very short Summary (2-3 sentences max) highlighting the most important findings in simple, layman terms.
- Briefly explain what each abnormal result means using everyday language (avoid medical jargon).
- You can NEVER prescribe, suggest, or recommend any medication or treatment to address abnormal values. Recommend the patient discuss results with their doctor.
""".strip()

PRESCRIPTION_SYSTEM_PROMPT = """You are Dr. Aria, an AI health assistant who explains prescriptions clearly.

Guidelines:
- List every medicine from [PATIENT DATA] in this format:
  Medicine Name — Dose | Frequency | Duration | Purpose (if stated)
- Include prescribing doctor, date, and diagnosis/condition if present in the data.
- Briefly explain what each medicine is commonly used for in a single, simple sentence (layman terms).
- Keep instructions concise. Do NOT recommend, adjust, suggest substitutions, or prescribe any medications or dosages beyond what is written in the prescription.
- If information is missing (e.g., duration not stated), say "Not specified".
- End with a reminder to follow the doctor's instructions and not self-adjust doses.
""".strip()

SYMPTOM_SYSTEM_PROMPT = """You are Dr. Aria, a caring AI health assistant helping someone understand their symptoms.

Guidelines:
- Acknowledge the symptom(s) with empathy before giving information.
- Provide practical, safe home-care advice using a few clear, concise bullet points in layman terms.
- Only recommend well-known, safe, evidence-based home remedies (e.g. hydration, rest, steam inhalation, warm water gargle). Never suggest unverified, speculative, or dangerous treatments that could mislead or harm the user.
- Explain likely common causes in very simple, everyday language (avoid medical jargon, not a diagnosis).
- List at least 3 specific warning signs that require urgent medical attention.
- You can NEVER prescribe or recommend specific prescription medications. Limit advice to non-medicinal, safe home-care measures and emphasize when to see a doctor.
""".strip()

MENTAL_HEALTH_SYSTEM_PROMPT = """You are Dr. Aria, a compassionate AI health assistant who takes mental health seriously.

Guidelines:
- Respond with empathy and warmth. Never be dismissive.
- Provide general psychoeducation about the condition or feeling described.
- Suggest evidence-based coping strategies (breathing, journaling, routine, social support).
- Clearly recommend professional help — therapist, counsellor, or GP.
- If there is any risk of self-harm or crisis, immediately provide crisis resources.
- Do NOT diagnose mental health conditions. Avoid clinical labels unless the user uses them first.
""".strip()

URGENT_SYSTEM_PROMPT = """You are Dr. Aria, an AI health assistant responding to a potential medical emergency.

Guidelines:
- Lead with clear, direct safety instructions. This is NOT the time for lengthy explanations.
- Tell the user to call emergency services (911 / 999 / 112) or go to the nearest emergency room immediately.
- If the situation involves self-harm or suicidal thoughts, provide crisis hotline numbers (988 Suicide & Crisis Lifeline in the US, 116 123 Samaritans in the UK) and express genuine care.
- Provide only immediate first-aid or safety actions while waiting for help (e.g., sit upright for breathing difficulty, do not move if spinal injury suspected).
- Do NOT attempt to diagnose. Do NOT provide home remedies. Do NOT minimise the situation.
- Keep the response short, calm, and authoritative.
""".strip()

MIXED_SYSTEM_PROMPT = """You are Dr. Aria, an AI health assistant.
Answer using the information in [PATIENT DATA] and your medical knowledge.
Be short, concise, well-formatted, and use simple layman terms.
You can NEVER prescribe any medication. Always recommend professional medical consultation.
""".strip()

# ── Greetings and farewells ───────────────────────────────────────────────────

_GREETING_KW = frozenset([
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "howdy", "what's up", "whats up", "greetings", "sup",
])

_FAREWELL_KW = frozenset([
    "bye", "goodbye", "good bye", "see you", "see ya", "later", "take care",
    "cya", "farewell", "good night", "goodnight", "talk later", "ttyl",
])

_GREETING_REGEX = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in _GREETING_KW) + r')\b',
    re.IGNORECASE
)

_FAREWELL_REGEX = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in _FAREWELL_KW) + r')\b',
    re.IGNORECASE
)

GREETING_RESPONSE = (
    "Good to see you. I'm **Dr. Aria**, your health assistant.\n\n"
    "Ask me about your symptoms, medications, or lab results — "
    "I'll give you a clear, honest answer. What's on your mind?"
)
GREETING_MESSAGE = GREETING_RESPONSE
FAREWELL_RESPONSE = (
    "Take care of yourself. "
    "Don't hesitate to come back if anything comes up — I'm always here."
)

# ── Context window ────────────────────────────────────────────────────────────

MAX_HISTORY_TURNS = 3  # each turn = 1 user + 1 AI message

# ── Keyword sets ──────────────────────────────────────────────────────────────
# All sets are compiled into word-boundary regex patterns to prevent substring
# false-positives (e.g. "ast" inside "breakfast", "alt" inside "salt").
# Longer phrases are sorted first so regex alternation greedily matches them.


def _kw_to_regex(keywords: frozenset) -> re.Pattern:
    """Compile a frozenset of keywords into a word-boundary regex pattern."""
    sorted_kw = sorted(keywords, key=len, reverse=True)
    pattern = r'\b(?:' + '|'.join(re.escape(k) for k in sorted_kw) + r')\b'
    return re.compile(pattern, re.IGNORECASE)


_SYMPTOM_KW = frozenset([
    # Self-report phrases
    "i feel", "i am feeling", "i've been", "i have been feeling", "i'm feeling",
    "i've had", "i have had", "i keep", "i keep getting", "i can't", "i cannot",
    "my body", "my chest", "my head", "my stomach", "my back", "my leg",
    "my arm", "my throat", "my eyes", "my skin", "my joints",
    # Pain & discomfort
    "pain", "ache", "aching", "hurts", "hurting", "sore", "soreness",
    "cramp", "cramping", "throbbing", "stabbing", "burning", "stinging",
    "tingling", "numbness", "numb", "tender", "sensitivity",
    # Fever & temperature
    "fever", "high temperature", "chills", "chilly", "shivering", "sweating",
    "night sweats", "hot flashes", "cold sweats",
    # Respiratory
    "cough", "coughing", "wheezing", "breathless", "shortness of breath",
    "difficulty breathing", "tight chest", "chest tightness", "runny nose",
    "stuffy nose", "congestion", "sneezing", "sore throat", "hoarse",
    # Gastrointestinal
    "nausea", "vomiting", "vomit", "threw up", "diarrhea", "diarrhoea",
    "constipation", "bloating", "bloated", "gas", "indigestion", "heartburn",
    "acid reflux", "stomach ache", "abdominal pain", "loose stools",
    # Neurological / head
    "headache", "migraine", "dizziness", "dizzy", "lightheaded", "fainting",
    "vertigo", "confusion", "forgetfulness", "memory loss", "blurred vision",
    "double vision", "ringing in ears", "tinnitus", "ear pain",
    # Energy & general
    "fatigue", "tired", "tiredness", "exhausted", "exhaustion", "lethargy",
    "weakness", "weak", "low energy", "not sleeping", "insomnia",
    "oversleeping", "loss of appetite", "not eating",
    # Skin
    "rash", "rashes", "hives", "itching", "itch", "itchy", "redness",
    "swelling", "swollen", "bruising", "bruise", "dry skin", "peeling",
    "yellow skin", "jaundice", "pale skin",
    # Bleeding
    "bleeding", "blood in stool", "blood in urine", "blood in mucus",
    "bleeding gums", "nosebleed",
    # Urinary
    "frequent urination", "burning urination", "dark urine", "cloudy urine",
    "no urination", "urine smell",
    # Musculoskeletal
    "joint pain", "knee pain", "back pain", "neck pain", "shoulder pain",
    "muscle pain", "muscle stiffness", "stiff neck", "stiff joints",
    # Cardiac
    "palpitations", "heart racing", "heart pounding", "irregular heartbeat",
    "skipped beat", "chest pain",
    # Weight
    "weight loss", "weight gain", "losing weight", "gaining weight",
    "sudden weight",
])

# Mental / emotional keywords — extracted from _SYMPTOM_KW so they route to
# MENTAL_HEALTH_SYSTEM_PROMPT (coping strategies, professional help) instead
# of SYMPTOM_SYSTEM_PROMPT (home remedies).
_MENTAL_HEALTH_KW = frozenset([
    # Emotional states
    "anxious", "anxiety", "panic", "panic attack",
    "depressed", "depression", "mood swings", "irritable",
    "crying", "sad", "hopeless", "stressed", "overwhelmed",
    # Broader mental health
    "mental health", "therapy", "therapist", "counselling", "counseling",
    "self esteem", "self-esteem", "loneliness", "lonely",
    "burnout", "emotional distress", "grief", "grieving",
    "ptsd", "trauma", "eating disorder", "anorexia", "bulimia",
    "ocd", "bipolar", "schizophrenia",
])

_PRESCRIPTION_KW = frozenset([
    "prescription", "prescriptions", "prescribed", "prescribe",
    "medicine", "medicines", "medication", "medications",
    "tablet", "tablets", "capsule", "capsules", "pill", "pills",
    "drug", "drugs", "syrup", "drops", "patch", "inhaler",
    "injection", "injections", "infusion", "ointment", "cream", "gel",
    "suppository", "nebulizer",
    "dosage", "dose", "doses", "how much to take", "when to take",
    "how to take", "side effects", "interactions", "drug interaction",
    "antibiotic", "antibiotics", "antifungal", "antiviral", "antidepressant",
    "antihypertensive", "diuretic", "painkiller", "pain reliever",
    "blood thinner", "anticoagulant", "statin", "beta blocker",
    "ace inhibitor", "calcium channel", "insulin", "metformin",
    "amlodipine", "lisinopril", "atorvastatin", "omeprazole",
    "pantoprazole", "azithromycin", "amoxicillin", "paracetamol",
    "ibuprofen", "aspirin", "cetirizine", "levocetirizine",
    "montelukast", "salbutamol", "fluticasone", "prednisone",
    "prednisolone", "levothyroxine", "methotrexate", "hydroxychloroquine",
    "what did the doctor", "what was prescribed", "my prescription",
    "my medicine", "my medication",
])

_LAB_KW = frozenset([
    # General
    "lab", "laboratory", "report", "result", "results", "test", "tests",
    "blood test", "blood work", "my report", "my results", "my lab",
    "my blood test", "my test", "test report",
    # Imaging
    "scan", "mri", "x-ray", "xray", "ultrasound", "ct scan", "pet scan",
    "ecg", "ekg", "echocardiogram", "endoscopy", "colonoscopy", "biopsy",
    # CBC / haematology
    "hemoglobin", "haemoglobin", "hgb", "hb",
    "platelet", "platelets", "plt",
    "wbc", "white blood cell", "white blood count",
    "rbc", "red blood cell", "red blood count",
    "hematocrit", "haematocrit", "hct", "mcv", "mch", "mchc", "rdw",
    "neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil",
    # Metabolic / chemistry
    "glucose", "fasting glucose", "blood sugar", "hba1c",
    "cholesterol", "ldl", "hdl", "triglycerides", "vldl",
    "creatinine", "urea", "bun", "uric acid", "gfr", "egfr",
    "sodium", "potassium", "chloride", "bicarbonate", "calcium",
    "magnesium", "phosphorus", "albumin", "total protein",
    # Liver
    "sgpt", "alt", "sgot", "ast", "ggt", "alp", "alkaline phosphatase",
    "bilirubin", "direct bilirubin", "indirect bilirubin", "liver function",
    "lft", "liver enzymes",
    # Thyroid
    "thyroid", "tsh", "t3", "t4", "free t3", "free t4", "thyroid function",
    # Vitamins & minerals
    "vitamin d", "vitamin b12", "vitamin c", "vitamin a", "vitamin e",
    "folate", "folic acid", "iron", "ferritin", "tibc", "transferrin",
    "zinc", "copper",
    # Cardiac markers
    "troponin", "ck-mb", "creatine kinase", "bnp",
    # Inflammation / infection
    "crp", "c-reactive protein", "esr", "erythrocyte sedimentation",
    "procalcitonin", "widal", "dengue", "malaria", "typhoid",
    # Hormones
    "testosterone", "estrogen", "estradiol", "progesterone", "prolactin",
    "cortisol", "c-peptide", "lh", "fsh", "amh",
    # Urine
    "urine test", "urinalysis", "urine culture", "urine routine",
    "protein in urine", "microalbumin", "ketones in urine",
    # Status words
    "abnormal", "normal range", "reference range", "elevated",
    "below normal", "within range", "out of range", "borderline",
    "critical value",
])

_URGENT_KW = frozenset([
    # Cardiac
    "heart attack", "cardiac arrest", "chest pain", "chest tightness",
    "chest pressure", "jaw pain",
    # Neurological
    "stroke", "seizure", "convulsion", "fitting",
    "unconscious", "unresponsive", "passed out", "fainting", "fainted",
    "sudden confusion", "can't speak", "cannot speak", "slurred speech",
    "sudden vision loss", "face drooping",
    # Respiratory
    "can't breathe", "cannot breathe", "shortness of breath severe",
    "choking", "stopped breathing", "turning blue",
    # Bleeding / trauma
    "severe bleeding", "heavy bleeding", "uncontrolled bleeding",
    "coughing blood", "vomiting blood",
    # Mental health emergencies
    "suicide", "suicidal", "self harm", "self-harm", "overdose",
    "poisoning", "want to die", "kill myself", "end my life",
    "harming myself",
    # Allergic
    "anaphylaxis", "anaphylactic", "throat closing", "throat swelling",
    # General emergency
    "emergency", "ambulance", "call 911", "call 999", "call 112",
])

# ── Health topic whitelist ────────────────────────────────────────────────────

_HEALTH_KW = frozenset([
    # Vitals & measurements
    "pulse", "bpm", "spo2", "oxygen", "bp", "heart rate", "respiration rate",
    "systolic", "diastolic", "temperature",
    # Body & anatomy
    "body", "blood", "heart", "lung", "lungs", "liver", "kidney", "kidneys",
    "brain", "bone", "bones", "muscle", "muscles", "skin", "eye", "eyes",
    "ear", "ears", "nose", "throat", "stomach", "bowel", "bowels",
    "intestine", "intestines", "colon", "rectum", "bladder", "uterus",
    "ovary", "ovaries", "prostate", "pancreas", "spleen", "gallbladder",
    "appendix", "spine", "spinal cord", "nerve", "nerves", "artery",
    "arteries", "vein", "veins", "thyroid", "adrenal", "pituitary",
    "tonsils", "trachea", "esophagus", "diaphragm",
    # Conditions & diseases
    "disease", "disorder", "condition", "syndrome", "infection",
    "cancer", "carcinoma", "tumor", "tumour", "malignant", "benign",
    "diabetes", "diabetic", "type 1", "type 2", "pre-diabetes",
    "hypertension", "high blood pressure", "low blood pressure", "hypotension",
    "blood pressure", "cholesterol", "hyperlipidemia",
    "thyroid", "hypothyroid", "hyperthyroid", "hashimoto",
    "anemia", "anaemia", "iron deficiency",
    "asthma", "copd", "bronchitis", "pneumonia", "tuberculosis", "tb",
    "allergy", "allergies", "allergic", "hay fever", "rhinitis",
    "arthritis", "rheumatoid", "osteoarthritis", "gout", "lupus",
    "fibromyalgia", "osteoporosis",
    "depression", "anxiety", "bipolar", "schizophrenia", "adhd", "autism",
    "ocd", "ptsd", "eating disorder", "anorexia", "bulimia",
    "fever", "flu", "influenza", "cold", "common cold",
    "covid", "covid-19", "coronavirus",
    "virus", "viral", "bacteria", "bacterial", "fungal", "parasite",
    "uti", "urinary tract infection", "kidney infection", "cystitis",
    "eczema", "psoriasis", "acne", "dermatitis",
    "migraine", "epilepsy", "parkinson", "alzheimer", "dementia",
    "multiple sclerosis",
    "hepatitis", "cirrhosis", "fatty liver",
    "ibs", "irritable bowel", "crohn", "ulcerative colitis", "celiac",
    "acid reflux", "gerd", "peptic ulcer",
    "pcos", "endometriosis", "menopause", "menstruation", "period",
    "pregnancy", "pregnant", "miscarriage", "fertility",
    "erectile dysfunction", "sexual health", "std", "sti",
    "hiv", "aids",
    "stroke", "heart disease", "coronary artery", "heart failure",
    "arrhythmia", "atrial fibrillation",
    # Symptoms
    "pain", "ache", "fever", "cough", "nausea", "vomit", "dizziness",
    "fatigue", "tired", "weak", "swelling", "bleeding", "rash", "itch",
    "headache", "migraine", "breathe", "breathing", "chest", "dizzy",
    "sore", "cramp", "tingling", "numbness", "tremor", "shaking",
    "jaundice", "pale",
    # Tests & reports
    "lab", "test", "report", "result", "blood test", "scan", "mri",
    "x-ray", "xray", "ultrasound", "ecg", "ekg", "biopsy", "ct scan",
    "hemoglobin", "glucose", "creatinine", "bilirubin",
    "platelet", "wbc", "rbc", "hba1c", "cholesterol", "uric acid",
    "sgpt", "sgot", "alt", "ast", "tsh", "t3", "t4",
    "vitamin", "iron", "ferritin", "calcium", "sodium", "potassium",
    "troponin", "crp", "esr", "ldl", "hdl", "triglycerides",
    # Medicines
    "medicine", "medication", "tablet", "capsule", "drug", "prescription",
    "prescribed", "dosage", "dose", "syrup", "antibiotic", "supplement",
    "vaccine", "vaccination", "injection", "insulin", "steroid",
    "painkiller", "antidepressant", "antihypertensive",
    "inhaler", "ointment", "cream", "drops",
    # Healthcare system
    "doctor", "physician", "specialist", "surgeon", "nurse", "pharmacist",
    "hospital", "clinic", "emergency room", "patient", "surgery",
    "treatment", "therapy", "physiotherapy", "chemotherapy",
    "diagnosis", "prognosis", "referral", "consultation",
    "health", "medical", "healthcare", "wellness",
    # Lifestyle & preventive
    "diet", "nutrition", "calorie", "calories", "protein", "carbohydrate",
    "exercise", "workout", "fitness", "physical activity",
    "weight", "bmi", "obesity", "overweight", "underweight",
    "sleep", "insomnia", "sleep apnea",
    "smoking", "quit smoking", "alcohol", "drinking", "addiction",
    "stress", "mental health", "mindfulness", "meditation",
    "checkup", "screening", "preventive", "prevention",
    "vaccine", "immunisation", "immunization",
    # Paediatric / geriatric
    "child health", "paediatric", "pediatric", "infant", "baby", "toddler",
    "growth", "developmental",
    "elderly", "geriatric", "old age", "aging", "senior health",
    # Personal context
    "my report", "my test", "my results", "my prescription", "my medication",
    "my doctor", "my health", "my blood", "i feel", "i am feeling",
    "i have been", "my symptoms", "my condition", "my diagnosis",
    "my surgery", "my treatment", "my history",
])

OFF_TOPIC_RESPONSE = (
    "I'm **Dr. Aria**, your health assistant. "
    "I specialise in health-related topics — symptoms, lab reports, prescriptions, "
    "and general medical information.\n\n"
    "It looks like your question might be outside my area of expertise. "
    "Here are some examples of questions you can ask me that I can help with:\n"
    "- *\"What are the common causes of a persistent cough?\"*\n"
    "- *\"How can I manage mild lower back pain at home?\"*\n"
    "- *\"What does a high ALT level in a liver function test mean?\"*\n"
    "- *\"Can you explain what paracetamol is prescribed for?\"*\n"
    "- *\"What are some coping strategies for managing anxiety?\"*"
)


# ── Compile all keyword sets to word-boundary regex ───────────────────────────

_SYMPTOM_REGEX      = _kw_to_regex(_SYMPTOM_KW)
_MENTAL_HEALTH_REGEX = _kw_to_regex(_MENTAL_HEALTH_KW)
_PRESCRIPTION_REGEX = _kw_to_regex(_PRESCRIPTION_KW)
_LAB_REGEX          = _kw_to_regex(_LAB_KW)
_URGENT_REGEX       = _kw_to_regex(_URGENT_KW)
_HEALTH_REGEX       = _kw_to_regex(_HEALTH_KW)


# ── Classifier ────────────────────────────────────────────────────────────────

def is_health_related(query: str) -> bool:
    """Return True if query contains at least one health keyword."""
    return bool(_HEALTH_REGEX.search(query))


def classify_intent(query: str) -> str:
    """
    Classify query intent.

    Priority order:
      1. urgent        — emergencies always escalate first
      2. mental_health — emotional / psychological queries (before generic symptom)
      3. symptom       — personal symptom / feeling queries
      4. lab           — lab reports & test results
      5. prescription  — medicines & prescriptions
      6. general       — anything else health-related
      7. greeting      — greetings
      8. farewell      — farewells
      9. off_topic     — genuinely unrelated to health

    Urgent is checked FIRST so crisis phrases like "kill myself" or "chest pain"
    are never routed to the symptom prompt (which gives home-remedy advice).
    Mental health is checked BEFORE symptom so emotional keywords like "anxious"
    or "depressed" are routed to MENTAL_HEALTH_SYSTEM_PROMPT (coping + professional
    referral) instead of SYMPTOM_SYSTEM_PROMPT (home remedies).
    """
    q = query.strip()
    if _URGENT_REGEX.search(q):
        return "urgent"
    if _MENTAL_HEALTH_REGEX.search(q):
        return "mental_health"
    if _SYMPTOM_REGEX.search(q):
        return "symptom"
    if _LAB_REGEX.search(q):
        return "lab"
    if _PRESCRIPTION_REGEX.search(q):
        return "prescription"
    if is_health_related(q):
        return "general"
    if _GREETING_REGEX.search(q):
        return "greeting"
    if _FAREWELL_REGEX.search(q):
        return "farewell"
    return "off_topic"


def detect_urgent(query: str) -> bool:
    """Return True if the query contains urgent / emergency keywords."""
    return bool(_URGENT_REGEX.search(query))


def get_system_prompt(intent: str) -> str:
    return {
        "lab":           LAB_SYSTEM_PROMPT,
        "prescription":  PRESCRIPTION_SYSTEM_PROMPT,
        "symptom":       SYMPTOM_SYSTEM_PROMPT,
        "urgent":        URGENT_SYSTEM_PROMPT,
        "general":       GENERAL_SYSTEM_PROMPT,
        "mental_health": MENTAL_HEALTH_SYSTEM_PROMPT,
    }.get(intent, MIXED_SYSTEM_PROMPT)


def get_max_tokens(intent: str) -> int:
    from health_ai.config.settings import (
        MAX_TOKENS_GENERAL, MAX_TOKENS_LAB,
        MAX_TOKENS_PRESCRIPTION, MAX_TOKENS_SYMPTOM,
        MAX_TOKENS_URGENT, MAX_TOKENS_MENTAL_HEALTH,
    )
    return {
        "lab":           MAX_TOKENS_LAB,
        "prescription":  MAX_TOKENS_PRESCRIPTION,
        "symptom":       MAX_TOKENS_SYMPTOM,
        "urgent":        MAX_TOKENS_URGENT,
        "general":       MAX_TOKENS_GENERAL,
        "mental_health": MAX_TOKENS_MENTAL_HEALTH,
    }.get(intent, MAX_TOKENS_GENERAL)
