"""
Gemini API adapter for CityLens (Google AI Studio free tier).
Set GOOGLE_API_KEY or GEMINI_API_KEY. Get key at https://aistudio.google.com/apikey
"""
import os
import base64
import time
import requests


GEMINI_MODEL_NAMES = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-8b", "gemini"]


def _session_to_gemini_parts(session):
    parts = []
    if not session or session[0].get("role") != "user":
        return parts
    content = session[0].get("content", [])
    if isinstance(content, str):
        return [content]
    for item in content:
        if item.get("type") == "text":
            parts.append(item["text"])
        elif item.get("type") == "image_url":
            url = (item.get("image_url") or {}).get("url")
            if not url:
                continue
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                b64 = base64.b64encode(resp.content).decode("utf-8")
                mime = (resp.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip()
                if not mime or mime == "application/octet-stream":
                    mime = "image/jpeg"
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})
            except Exception as e:
                print(f"Warning: could not fetch image: {e}")
    return parts


def get_response_mllm_api_gemini(session, model_name, temperature=0, max_tokens=1000, max_retries=5):
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GOOGLE_API_KEY or GEMINI_API_KEY (free at aistudio.google.com/apikey)")
        return None
    try:
        import google.generativeai as genai
    except ImportError:
        print("Error: pip install google-generativeai")
        return None
    genai.configure(api_key=api_key)
    parts = _session_to_gemini_parts(session)
    if not parts:
        return None
    gemini_model = "gemini-1.5-flash" if model_name == "gemini" else model_name
    retries = 0
    while retries < max_retries:
        try:
            model = genai.GenerativeModel(gemini_model)
            config = {"temperature": temperature, "max_output_tokens": max_tokens}
            response = model.generate_content(parts, generation_config=config)
            if response and response.text:
                return response.text.strip()
            return None
        except Exception as e:
            retries += 1
            print(f"Error calling Gemini API: {e}. Retry {retries}/{max_retries}...")
            if retries < max_retries:
                time.sleep(2 ** retries)
    return None
