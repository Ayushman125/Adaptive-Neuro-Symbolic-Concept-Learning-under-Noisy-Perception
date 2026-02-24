import json
import time

import requests

from .config import (
    DEEPINFRA_API_KEY,
    DEEPINFRA_MIN_INTERVAL_SEC,
    DEEPINFRA_MODEL,
    DEEPINFRA_URL,
    GEMINI_API_KEY,
    GEMINI_MIN_INTERVAL_SEC,
    GEMINI_MODEL,
    GEMINI_URL_TEMPLATE,
    GROQ_API_KEY,
    GROQ_MIN_INTERVAL_SEC,
    GROQ_MODEL,
    GROQ_URL,
    HF_API_KEY,
    HF_MIN_INTERVAL_SEC,
    HF_MODEL,
    HF_URL_TEMPLATE,
    OLLAMA_MODEL,
    OLLAMA_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_MIN_INTERVAL_SEC,
    OPENROUTER_MODEL,
    OPENROUTER_URL,
    PERCEPTION_BACKEND,
    PERCEPTION_TIMEOUT_SEC,
    TOGETHER_API_KEY,
    TOGETHER_MIN_INTERVAL_SEC,
    TOGETHER_MODEL,
    TOGETHER_URL,
)


def call_perception_backend(machine, prompt):
    if PERCEPTION_BACKEND == "gemini":
        return call_gemini(machine, prompt)
    if PERCEPTION_BACKEND == "groq":
        return call_groq(machine, prompt)
    if PERCEPTION_BACKEND in {"openrouter", "router"}:
        return call_openrouter(machine, prompt)
    if PERCEPTION_BACKEND in {"deepinfra", "deep"}:
        return call_deepinfra(machine, prompt)
    if PERCEPTION_BACKEND in {"together", "togetherai"}:
        return call_together(machine, prompt)
    if PERCEPTION_BACKEND in {"huggingface", "hf"}:
        return call_huggingface(machine, prompt)
    return call_ollama(machine, prompt)


def throttle_perception(machine):
    now = time.time()
    elapsed = now - machine._last_perception_call_ts
    if elapsed < GEMINI_MIN_INTERVAL_SEC:
        wait_seconds = GEMINI_MIN_INTERVAL_SEC - elapsed
        machine._log_perception(f"throttling requests, waiting {wait_seconds:.1f}s")
        time.sleep(wait_seconds)
    machine._last_perception_call_ts = time.time()


def active_gemini_model(machine):
    if not machine.gemini_model_candidates:
        return GEMINI_MODEL
    return machine.gemini_model_candidates[machine._gemini_model_index % len(machine.gemini_model_candidates)]


def rotate_gemini_model(machine):
    if not machine.gemini_model_candidates:
        return
    machine._gemini_model_index = (machine._gemini_model_index + 1) % len(machine.gemini_model_candidates)


def call_ollama(machine, prompt):
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}
    res = requests.post(OLLAMA_URL, json=payload, timeout=PERCEPTION_TIMEOUT_SEC)
    res.raise_for_status()
    body = res.json()
    raw = body.get("response", "{}")
    if not isinstance(raw, str):
        raw = json.dumps(raw)
    return raw, res.status_code, body.get("done", None)


def call_gemini(machine, prompt):
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable")

    throttle_perception(machine)
    model_name = active_gemini_model(machine)
    url = GEMINI_URL_TEMPLATE.format(model=model_name, api_key=GEMINI_API_KEY)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "maxOutputTokens": 256,
        },
    }
    res = requests.post(url, json=payload, timeout=PERCEPTION_TIMEOUT_SEC)
    res.raise_for_status()
    body = res.json()

    candidates = body.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    if not parts:
        raise RuntimeError("Gemini returned empty content parts")

    raw = parts[0].get("text", "{}")
    if not isinstance(raw, str):
        raw = json.dumps(raw)

    finish_reason = candidates[0].get("finishReason", None)
    machine._log_perception(f"gemini_model='{model_name}'")
    return raw, res.status_code, finish_reason


def call_groq(machine, prompt):
    if not GROQ_API_KEY:
        raise RuntimeError(
            "Missing GROQ_API_KEY environment variable.\n"
            "Get your free API key from: https://console.groq.com/keys\n"
            "Then set: $env:GROQ_API_KEY='your_key_here'"
        )

    now = time.time()
    if hasattr(machine, "_last_groq_call_ts"):
        elapsed = now - machine._last_groq_call_ts
        if elapsed < GROQ_MIN_INTERVAL_SEC:
            wait_seconds = GROQ_MIN_INTERVAL_SEC - elapsed
            time.sleep(wait_seconds)
    machine._last_groq_call_ts = time.time()

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    json_prompt = prompt + "\n\nIMPORTANT: Respond with ONLY a valid JSON object, no other text."

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a feature extraction assistant. Always respond with valid JSON only.",
            },
            {
                "role": "user",
                "content": json_prompt,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }

    res = requests.post(GROQ_URL, json=payload, headers=headers, timeout=PERCEPTION_TIMEOUT_SEC)
    res.raise_for_status()
    body = res.json()

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError("Groq returned no choices")

    message = choices[0].get("message", {})
    raw = message.get("content", "{}")

    if not isinstance(raw, str):
        raw = json.dumps(raw)

    machine._log_perception(f"groq_model='{GROQ_MODEL}'")
    return raw, res.status_code, "completed"


def call_openrouter(machine, prompt):
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY environment variable.\n"
            "Get your free API key from: https://openrouter.ai/keys\n"
            "No credit card required!\n"
            "Then set: $env:OPENROUTER_API_KEY='your_key_here'"
        )

    now = time.time()
    if hasattr(machine, "_last_openrouter_call_ts"):
        elapsed = now - machine._last_openrouter_call_ts
        if elapsed < OPENROUTER_MIN_INTERVAL_SEC:
            wait_seconds = OPENROUTER_MIN_INTERVAL_SEC - elapsed
            time.sleep(wait_seconds)
    machine._last_openrouter_call_ts = time.time()

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/thinking-machine",
        "X-Title": "Tenenbaum Thinking Machine",
    }

    json_prompt = prompt + "\n\nRespond with ONLY a JSON object, no other text."

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": json_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }

    res = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=PERCEPTION_TIMEOUT_SEC)
    res.raise_for_status()
    body = res.json()

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError("OpenRouter returned no choices")

    message = choices[0].get("message", {})
    raw = message.get("content", "{}")

    if not isinstance(raw, str):
        raw = json.dumps(raw)

    machine._log_perception(f"openrouter_model='{OPENROUTER_MODEL}'")
    return raw, res.status_code, "completed"


def call_deepinfra(machine, prompt):
    if not DEEPINFRA_API_KEY:
        raise RuntimeError(
            "Missing DEEPINFRA_API_KEY environment variable.\n"
            "Get your free API key from: https://deepinfra.com/dash/api_keys\n"
            "No credit card required!\n"
            "Then set: $env:DEEPINFRA_API_KEY='your_key_here'"
        )

    now = time.time()
    if hasattr(machine, "_last_deepinfra_call_ts"):
        elapsed = now - machine._last_deepinfra_call_ts
        if elapsed < DEEPINFRA_MIN_INTERVAL_SEC:
            wait_seconds = DEEPINFRA_MIN_INTERVAL_SEC - elapsed
            time.sleep(wait_seconds)
    machine._last_deepinfra_call_ts = time.time()

    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json",
    }

    json_prompt = prompt + "\n\nRespond with ONLY a JSON object, no other text."

    payload = {
        "model": DEEPINFRA_MODEL,
        "messages": [
            {"role": "user", "content": json_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }

    res = requests.post(DEEPINFRA_URL, json=payload, headers=headers, timeout=PERCEPTION_TIMEOUT_SEC)
    res.raise_for_status()
    body = res.json()

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError("DeepInfra returned no choices")

    message = choices[0].get("message", {})
    raw = message.get("content", "{}")

    if not isinstance(raw, str):
        raw = json.dumps(raw)

    machine._log_perception(f"deepinfra_model='{DEEPINFRA_MODEL}'")
    return raw, res.status_code, "completed"


def call_together(machine, prompt):
    if not TOGETHER_API_KEY:
        raise RuntimeError(
            "Missing TOGETHER_API_KEY environment variable.\n"
            "Get your free API key from: https://api.together.xyz/settings/api-keys\n"
            "Free tier: $25 credit (no expiry!)\n"
            "Then set: $env:TOGETHER_API_KEY='your_key_here'"
        )

    now = time.time()
    if hasattr(machine, "_last_together_call_ts"):
        elapsed = now - machine._last_together_call_ts
        if elapsed < TOGETHER_MIN_INTERVAL_SEC:
            wait_seconds = TOGETHER_MIN_INTERVAL_SEC - elapsed
            time.sleep(wait_seconds)
    machine._last_together_call_ts = time.time()

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": TOGETHER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a feature extraction assistant. Respond ONLY with valid JSON, no other text.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }

    res = requests.post(TOGETHER_URL, json=payload, headers=headers, timeout=PERCEPTION_TIMEOUT_SEC)
    res.raise_for_status()
    body = res.json()

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError("Together AI returned no choices")

    message = choices[0].get("message", {})
    raw = message.get("content", "{}")

    if not isinstance(raw, str):
        raw = json.dumps(raw)

    machine._log_perception(f"together_model='{TOGETHER_MODEL}'")
    return raw, res.status_code, "completed"


def call_huggingface(machine, prompt):
    if not HF_API_KEY:
        raise RuntimeError(
            "Missing HF_API_KEY environment variable.\n"
            "Get your free API key from: https://huggingface.co/settings/tokens\n"
            "Then set: export HF_API_KEY='your_key_here'"
        )

    now = time.time()
    if hasattr(machine, "_last_hf_call_ts"):
        elapsed = now - machine._last_hf_call_ts
        if elapsed < HF_MIN_INTERVAL_SEC:
            wait_seconds = HF_MIN_INTERVAL_SEC - elapsed
            time.sleep(wait_seconds)
    machine._last_hf_call_ts = time.time()

    url = HF_URL_TEMPLATE.format(model=HF_MODEL)
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    formatted_prompt = f"""[INST] {prompt}

IMPORTANT: Respond with ONLY a valid JSON object, no other text. [/INST]
"""

    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "temperature": 0.2,
            "max_new_tokens": 256,
            "return_full_text": False,
        },
        "options": {
            "wait_for_model": True,
        },
    }

    res = requests.post(url, json=payload, headers=headers, timeout=PERCEPTION_TIMEOUT_SEC)
    res.raise_for_status()
    body = res.json()

    if isinstance(body, list) and len(body) > 0:
        raw = body[0].get("generated_text", "{}")
    elif isinstance(body, dict):
        raw = body.get("generated_text", body.get("text", "{}"))
    else:
        raw = "{}"

    if not isinstance(raw, str):
        raw = json.dumps(raw)

    machine._log_perception(f"huggingface_model='{HF_MODEL}'")
    return raw, res.status_code, "completed"
