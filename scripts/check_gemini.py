"""
scripts/check_gemini.py

One-off connection test for the Gemini API.
Confirms that the API key, the model name in .env and the google-genai SDK all work.

It makes exactly ONE generation request, which uses 1 request of your free daily quota.
Listing the models first does not generate any text.

Run from the project root:
    python scripts/check_gemini.py
"""

import os
import sys
import time

import httpx2  # HTTP library used inside google-genai; we only need its error types
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Read .env (in the current folder) into environment variables.
# override=True: values in this project's .env win over anything already set
# in your shell (for example a key exported in ~/.zshrc for another project).
load_dotenv(override=True)

# This project uses only GEMINI_API_KEY. Remove GOOGLE_API_KEY from THIS
# process (your shell is not changed) so the SDK can't pick up a key that
# belongs to a different project.
os.environ.pop("GOOGLE_API_KEY", None)

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

# Fail early with a clear message. Never print the key itself.
if not api_key:
    sys.exit("GEMINI_API_KEY is missing from .env")
if not model:
    sys.exit("GEMINI_MODEL is missing from .env")

# Without a timeout, a request to an overloaded server can hang for minutes.
# HttpOptions.timeout is in MILLISECONDS: 30_000 ms = 30 seconds.
TIMEOUT_MS = 30_000
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=TIMEOUT_MS),
)

# --- 1) Can this key see the model we configured? ---------------------------
print("Checking that your key can see the model...")
try:
    # The API returns names like "models/gemini-3.8-flash"; strip the prefix.
    available = [m.name.removeprefix("models/") for m in client.models.list()]
except errors.APIError as e:
    # Typical causes: invalid key (400/403) or no network.
    sys.exit(f"Could not list models (HTTP {e.code} {e.status}): {e.message}")

if model not in available:
    print(f"'{model}' is NOT available to this key.")
    print("Gemini models this key can see:")
    for name in sorted(n for n in available if n.startswith("gemini")):
        print("  -", name)
    sys.exit(1)

print(f"OK: '{model}' is available.")

# --- 2) One small text generation request -----------------------------------
# Server errors (5xx, e.g. 503 "high demand") are usually temporary, so we
# retry a LIMITED number of times, waiting longer each time ("exponential
# backoff"). Client errors (4xx) are never retried: a bad key or an empty
# quota won't fix itself in a few seconds.
MAX_ATTEMPTS = 3
WAIT_SECONDS = [5, 15]  # wait before attempt 2 and attempt 3

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"Sending one test request (attempt {attempt} of {MAX_ATTEMPTS})...")
    start = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model,
            contents="Reply with exactly this sentence: MultiDoc AI connection OK",
        )
        break  # success: leave the retry loop
    except errors.ClientError as e:
        # 4xx = problem on our side: bad key, quota used up (429), wrong model (404).
        sys.exit(f"Request rejected (HTTP {e.code} {e.status}): {e.message}")
    except (errors.ServerError, httpx2.TransportError) as e:
        # 5xx = problem on Google's side.
        # TransportError = no usable reply: timeout or network failure.
        if isinstance(e, httpx2.TimeoutException):
            print(f"  No reply within {TIMEOUT_MS // 1000} s (timeout).")
        elif isinstance(e, httpx2.TransportError):
            print(f"  Network problem: {type(e).__name__}. Check your internet connection.")
        else:
            print(f"  Gemini server error (HTTP {e.code} {e.status}): {e.message}")
        if attempt == MAX_ATTEMPTS:
            sys.exit(
                "Gave up after 3 attempts. Try again in a few minutes, or set "
                "GEMINI_MODEL to another free model in .env."
            )
        wait = WAIT_SECONDS[attempt - 1]
        print(f"  Waiting {wait} s before retrying...")
        time.sleep(wait)

elapsed = time.perf_counter() - start

print("Model reply:", response.text)
print(f"Latency: {elapsed:.2f} s")

usage = response.usage_metadata
if usage:
    print(
        "Tokens:"
        f" prompt={usage.prompt_token_count},"
        f" response={usage.candidates_token_count},"
        f" thinking={usage.thoughts_token_count or 0},"
        f" total={usage.total_token_count}"
    )
