import os
import tempfile

# Keep tests away from the real Inspect response cache and quiet the progress display.
os.environ.setdefault("INSPECT_CACHE_DIR", tempfile.mkdtemp(prefix="arena-inspect-cache-"))
os.environ["INSPECT_DISPLAY"] = "none"
os.environ.pop("ARENA_TRACE", None)
