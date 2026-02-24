import os

# --- ARCHITECTURAL CONSTANTS ---
PERCEPTION_BACKEND = os.getenv("TM_PERCEPTION_BACKEND", "openrouter").strip().lower()
OLLAMA_URL = os.getenv("TM_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("TM_OLLAMA_MODEL", "llama3.2:1b")
GEMINI_MODEL = os.getenv("TM_GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_MODEL_CANDIDATES = [
    token.strip() for token in os.getenv(
        "TM_GEMINI_MODEL_CANDIDATES",
        "gemini-2.0-flash-lite,gemini-2.0-flash,gemini-flash-lite-latest,gemini-flash-latest"
    ).split(",") if token.strip()
]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

# Groq (FREE with generous limits - faster than others)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("TM_GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MIN_INTERVAL_SEC = float(os.getenv("TM_GROQ_MIN_INTERVAL", "0.1"))

# OpenRouter (FREE models - no credit card required!)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("TM_OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MIN_INTERVAL_SEC = float(os.getenv("TM_OPENROUTER_MIN_INTERVAL", "0.2"))

# DeepInfra (FREE - no credit card, truly free tier)
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", os.getenv("DEEPINFRA_TOKEN", "")).strip()
DEEPINFRA_MODEL = os.getenv("TM_DEEPINFRA_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEEPINFRA_MIN_INTERVAL_SEC = float(os.getenv("TM_DEEPINFRA_MIN_INTERVAL", "0.2"))

# Together AI (FREE alternative - $25 free credit, no expiry)
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", os.getenv("TOGETHER_AI_KEY", "")).strip()
TOGETHER_MODEL = os.getenv("TM_TOGETHER_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_MIN_INTERVAL_SEC = float(os.getenv("TM_TOGETHER_MIN_INTERVAL", "0.1"))

# HuggingFace Inference API (DEPRECATED - most free models removed)
HF_API_KEY = os.getenv("HF_API_KEY", os.getenv("HUGGINGFACE_API_KEY", "")).strip()
HF_MODEL = os.getenv("TM_HF_MODEL", "microsoft/Phi-3-mini-4k-instruct")
HF_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model}"
HF_MIN_INTERVAL_SEC = float(os.getenv("TM_HF_MIN_INTERVAL", "0.5"))

PERCEPTION_DEBUG = os.getenv("TM_PERCEPTION_DEBUG", "1") == "1"
PERCEPTION_TIMEOUT_SEC = int(os.getenv("TM_PERCEPTION_TIMEOUT", os.getenv("TM_OLLAMA_TIMEOUT", "30")))
PERCEPTION_MAX_RETRIES = int(os.getenv("TM_PERCEPTION_RETRIES", os.getenv("TM_OLLAMA_RETRIES", "2")))
GEMINI_MIN_INTERVAL_SEC = float(os.getenv("TM_GEMINI_MIN_INTERVAL", "3"))
GEMINI_RATE_LIMIT_COOLDOWN_SEC = float(os.getenv("TM_GEMINI_COOLDOWN", "20"))
MAX_STRING_TOKEN_LEN = int(os.getenv("TM_MAX_STRING_TOKEN_LEN", "28"))
MAX_STRING_TOKEN_UNDERSCORES = int(os.getenv("TM_MAX_STRING_TOKEN_UNDERSCORES", "3"))
