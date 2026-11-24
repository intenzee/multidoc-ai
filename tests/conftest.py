"""
Test setup. Runs BEFORE any backend module is imported:
  * DATA_DIR points to a throw-away folder, so tests never touch your real data;
  * the LLM is switched off (empty key), so tests never use Gemini quota.
"""

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="multidoc-tests-")
os.environ["DATA_DIR"] = _tmp
os.environ["GEMINI_API_KEY"] = ""
os.environ["CAPTION_IMAGES"] = "false"
os.environ.pop("GOOGLE_API_KEY", None)
