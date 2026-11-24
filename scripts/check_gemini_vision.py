"""
scripts/check_gemini_vision.py

Checks that the Gemini model in .env can actually SEE an image,
not just read text about one.

How: we draw a small bar chart whose answer exists ONLY in the picture
(no numbers are written on the bars), send the image bytes to Gemini,
and check whether its answer matches what we drew.

Makes ONE generation request (plus retries only on server or network errors).

Run from the project root:
    python scripts/check_gemini_vision.py
"""

import io
import os
import sys
import time
from pathlib import Path

import httpx2  # HTTP library used inside google-genai; we only need its error types
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from PIL import Image, ImageDraw, ImageFont

# The "ground truth": what the chart shows. The tallest bar is Mangoes = 8.
CATEGORIES = [("Apples", 3), ("Mangoes", 8), ("Bananas", 5)]
Y_MAX = 10
EXPECTED_LABEL = "mango"
EXPECTED_VALUE = "8"

OUTPUT_IMAGE = Path("data/test/vision_test_chart.png")  # data/ is gitignored


def draw_chart() -> bytes:
    """Draw a simple bar chart with Pillow and return it as PNG bytes."""
    width, height = 600, 420
    left, right, top, bottom = 70, 570, 60, 360  # edges of the plotting area

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=18)

    draw.text((left, 15), "Fruit boxes sold this week", fill="black", font=font)

    # Horizontal gridlines with y-axis labels 0, 2, 4, ... 10
    for value in range(0, Y_MAX + 1, 2):
        y = bottom - (bottom - top) * value / Y_MAX
        draw.line([(left, y), (right, y)], fill=(220, 220, 220))
        draw.text((left - 35, y - 10), str(value), fill="black", font=font)

    # Axes
    draw.line([(left, top), (left, bottom)], fill="black", width=2)
    draw.line([(left, bottom), (right, bottom)], fill="black", width=2)

    # Bars, with the category name under each bar (no value written on the bar)
    slot = (right - left) / len(CATEGORIES)
    for i, (label, value) in enumerate(CATEGORIES):
        x0 = left + i * slot + slot * 0.25
        x1 = left + (i + 1) * slot - slot * 0.25
        y0 = bottom - (bottom - top) * value / Y_MAX
        draw.rectangle([x0, y0, x1, bottom], fill=(66, 133, 244))
        draw.text((x0, bottom + 10), label, fill="black", font=font)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# --- 1) Make the test image -------------------------------------------------
png_bytes = draw_chart()
OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_IMAGE.write_bytes(png_bytes)
print(f"Test chart saved to {OUTPUT_IMAGE} ({len(png_bytes) // 1024} KB)")

# --- 2) Same safe setup as check_gemini.py ----------------------------------
load_dotenv(override=True)
os.environ.pop("GOOGLE_API_KEY", None)  # use only this project's key

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")
if not api_key:
    sys.exit("GEMINI_API_KEY is missing from .env")
if not model:
    sys.exit("GEMINI_MODEL is missing from .env")

TIMEOUT_MS = 30_000
client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))

# --- 3) Send image + question ------------------------------------------------
# "contents" is a list of parts. The image goes in as raw bytes with its MIME
# type, so the model receives the actual pixels, not a text description.
contents = [
    types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
    "Look at this bar chart. Which category has the tallest bar, and what is "
    "its approximate value on the y-axis? Answer in one short sentence.",
]

MAX_ATTEMPTS = 3
WAIT_SECONDS = [5, 15]

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"Sending image to '{model}' (attempt {attempt} of {MAX_ATTEMPTS})...")
    start = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=contents)
        break
    except errors.ClientError as e:
        if e.code == 400:
            print("  HTTP 400 can mean this model does not accept image input.")
        sys.exit(f"Request rejected (HTTP {e.code} {e.status}): {e.message}")
    except (errors.ServerError, httpx2.TransportError) as e:
        if isinstance(e, httpx2.TimeoutException):
            print(f"  No reply within {TIMEOUT_MS // 1000} s (timeout).")
        elif isinstance(e, httpx2.TransportError):
            print(f"  Network problem: {type(e).__name__}. Check your internet connection.")
        else:
            print(f"  Gemini server error (HTTP {e.code} {e.status}): {e.message}")
        if attempt == MAX_ATTEMPTS:
            sys.exit("Gave up after 3 attempts. Try again in a few minutes.")
        wait = WAIT_SECONDS[attempt - 1]
        print(f"  Waiting {wait} s before retrying...")
        time.sleep(wait)

elapsed = time.perf_counter() - start
answer = (response.text or "").strip()

print("Model reply:", answer)
print(f"Latency: {elapsed:.2f} s")

usage = response.usage_metadata
if usage:
    print(f"Tokens: prompt={usage.prompt_token_count} (includes the image), total={usage.total_token_count}")

# --- 4) Did it really see the chart? -----------------------------------------
if EXPECTED_LABEL in answer.lower() and EXPECTED_VALUE in answer:
    print("RESULT: PASS - the model read the chart correctly (Mangoes, about 8).")
else:
    print("RESULT: CHECK - the answer doesn't clearly say 'Mangoes' and '8'.")
    print(f"Open {OUTPUT_IMAGE} and compare it with the reply above.")
