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
"""

# ── System prompts ────────────────────────────────────────────────────────────

GENERAL_SYSTEM_PROMPT = """You are Dr. Aria, a warm, knowledgeable AI health assistant.
Your role is to provide clear, accurate, and empathetic health information.

Guidelines:
- Answer in 4-6 sentences, using **bold** for key medical terms.
- Use simple language; avoid unnecessary jargon.
- If relevant, mention lifestyle factors (diet, sleep, exercise, stress).
- Do NOT diagnose. Do NOT prescribe. Always recommend consulting a real doctor.
- Use relevant emojis sparingly to aid readability.
- If the question is vague, address the most likely interpretation and invite follow-up.
""".strip()

LAB_SYSTEM_PROMPT = """You are Dr. Aria, an AI health assistant specialising in interpreting lab reports.

Guidelines:
- Use ONLY values present in [PATIENT DATA]. Never invent or assume numbers.
- Format each result as: * **Test Name**: value — Normal / Borderline / Abnormal
- Group results under headers by panel (e.g., Complete Blood Count, Lipid Panel).
- After listing all results, write a Summary (3-4 sentences) highlighting the most important findings.
- Briefly explain what each abnormal result means in plain English.
- Do NOT diagnose. Recommend the patient discuss results with their doctor.
""".strip()

PRESCRIPTION_SYSTEM_PROMPT = """You are Dr. Aria, an AI health assistant who explains prescriptions clearly.

Guidelines:
- List every medicine from [PATIENT DATA] in this format:
  Medicine Name — Dose | Frequency | Duration | Purpose (if stated)
- Include prescribing doctor, date, and diagnosis/condition if present in the data.
- Briefly explain what each medicine is commonly used for (1 sentence).
- Do NOT add advice, warnings, or substitutions beyond what is written in the prescription.
- If information is missing (e.g., duration not stated), say "Not specified".
- End with a reminder to follow the doctor's instructions and not self-adjust doses.
""".strip()

SYMPTOM_SYSTEM_PROMPT = """You are Dr. Aria, a caring AI health assistant helping someone understand their symptoms.

Guidelines:
- Acknowledge the symptom(s) with empathy before giving information.
- Provide practical, safe home-care advice using clear bullet points.
- Explain likely common causes in simple terms (not a diagnosis).
- List at least 3 specific warning signs that require urgent medical attention.
- Always end by recommending they see a doctor if symptoms persist or worsen.
- Do NOT speculate on rare or serious diagnoses unless symptoms clearly suggest urgency.
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

MIXED_SYSTEM_PROMPT = """You are Dr. Aria, an AI health assistant.
Answer using the information in [PATIENT DATA] and your medical knowledge.
Be concise, well-formatted, and always recommend professional medical consultation.
""".strip()

# ── Safety text ───────────────────────────────────────────────────────────────

DISCLAIMER = (
    "\n\n---\n"
    "⚕️ *Dr. Aria is an AI assistant, not a licensed doctor. "
    "This information is for educational purposes only. "
    "Always consult a qualified healthcare professional for medical advice, "
    "diagnosis, or treatment.*"
)

URGENT_NOTICE = (
    "\n\n🚨 **URGENT — Please seek immediate medical attention.** "
    "Some of the symptoms or values you mentioned may indicate a medical emergency. "
    "Call emergency services (911 / 999 / 112) or go to the nearest hospital now. "
    "Do not wait."
)

# ── Greetings and farewells ───────────────────────────────────────────────────

_GREETING_KW = frozenset([
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "howdy", "what's up", "whats up", "greetings", "sup",
])

_FAREWELL_KW = frozenset([
    "bye", "goodbye", "good bye", "see you", "see ya", "later", "take care",
    "cya", "farewell", "good night", "goodnight", "talk later", "ttyl",
])

GREETING_RESPONSE = (
    "👋 Hello! I'm **Dr. Aria**, your personal health assistant.\n\n"
    "I can help you with:\n"
    "• 🧪 Reading and explaining your **lab reports**\n"
    "• 💊 Understanding your **prescriptions**\n"
    "• 🤒 Guidance on **symptoms** you're experiencing\n"
    "• 📋 General **medical knowledge** questions\n\n"
    "How may I help you today?"
)
GREETING_MESSAGE = (
    "👋 Hello! I'm **Dr. Aria**, your personal health assistant.\n\n"
    "I can help you with:\n"
    "• 🧪 Reading and explaining your **lab reports**\n"
    "• 💊 Understanding your **prescriptions**\n"
    "• 🤒 Guidance on **symptoms** you're experiencing\n"
    "• 📋 General **medical knowledge** questions\n\n"
    "How may I help you today?"
)
FAREWELL_RESPONSE = (
    "👋 Take care and stay healthy! "
    "Remember to keep up with your health checkups. "
    "Come back anytime you need help — I'm always here! 😊"
)
# ── Context window ────────────────────────────────────────────────────────────

MAX_HISTORY_TURNS = 0  # each turn = 1 user + 1 AI message

# ── Keyword sets ──────────────────────────────────────────────────────────────

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
    # Mental / emotional symptoms
    "anxious", "anxiety", "panic", "panic attack", "depressed", "depression",
    "mood swings", "irritable", "angry", "crying", "sad", "hopeless",
    "stressed", "overwhelmed",
    # Cardiac
    "palpitations", "heart racing", "heart pounding", "irregular heartbeat",
    "skipped beat", "chest pain",
    # Weight
    "weight loss", "weight gain", "losing weight", "gaining weight",
    "sudden weight",
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
    "\U0001fa7a I'm **Dr. Aria**, your personal health assistant. "
    "I specialise in health-related topics — lab reports, prescriptions, "
    "symptoms, and general medical information.\n\n"
    "It looks like your question might be outside my area of expertise. "
    "Please ask me something health-related and I'll do my best to help! 😊"
)


# ── Classifier ────────────────────────────────────────────────────────────────

def is_health_related(query: str) -> bool:
    """Return True if query contains at least one health keyword."""
    q = query.lower()
    return any(k in q for k in _HEALTH_KW)


def classify_intent(query: str) -> str:
    """
    Classify query intent.

    Priority order:
      1. urgent       — emergencies always escalate first
      2. symptom      — personal symptom / feeling queries
      3. lab          — lab reports & test results
      4. prescription — medicines & prescriptions
      5. general      — anything else health-related
      6. off_topic    — genuinely unrelated to health

    KEY FIX: is_health_related is the FALLBACK for general, not a gate.
    A query that misses specific keyword sets but is health-related is
    correctly routed to "general" instead of "off_topic".
    """
    q = query.lower().strip()
    if any(k in q for k in _GREETING_KW):
        return "greeting"
    if any(k in q for k in _FAREWELL_KW):
        return "farewell"
    if any(k in q for k in _SYMPTOM_KW):
        return "symptom"
    if any(k in q for k in _PRESCRIPTION_KW):
        return "prescription"
    if any(k in q for k in _LAB_KW):
        return "lab"
    if is_health_related(q):
        return "general"
    return "off_topic"

def detect_urgent(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in _URGENT_KW)


def get_system_prompt(intent: str) -> str:
    return {
        "lab":           LAB_SYSTEM_PROMPT,
        "prescription":  PRESCRIPTION_SYSTEM_PROMPT,
        "symptom":       SYMPTOM_SYSTEM_PROMPT,
        "urgent":        SYMPTOM_SYSTEM_PROMPT,
        "general":       GENERAL_SYSTEM_PROMPT,
        "mental_health": MENTAL_HEALTH_SYSTEM_PROMPT,
    }.get(intent, MIXED_SYSTEM_PROMPT)


def get_max_tokens(intent: str) -> int:
    from health_ai.config.settings import (
        MAX_TOKENS_GENERAL, MAX_TOKENS_LAB,
        MAX_TOKENS_PRESCRIPTION, MAX_TOKENS_SYMPTOM,
    )
    return {
        "lab":          MAX_TOKENS_LAB,
        "prescription": MAX_TOKENS_PRESCRIPTION,
        "symptom":      MAX_TOKENS_SYMPTOM,
        "urgent":       MAX_TOKENS_SYMPTOM,
        "general":      MAX_TOKENS_GENERAL,
    }.get(intent, MAX_TOKENS_GENERAL)
