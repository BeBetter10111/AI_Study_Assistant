"""Upload every supported file in a folder to the API in parallel.

Usage (from the host, backend must be running):
    pip install requests
    python bulk_upload.py ./tai-lieu --email you@example.com --password 'yourpassword'
Optional: --api http://localhost:8000/api  --workers 4  --no-wait
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ALLOWED = {".pdf", ".docx", ".pptx", ".md"}
DONE = {"embedded", "failed"}


def get_token(api, email, password):
    body = {"email": email, "password": password}
    r = requests.post(f"{api}/auth/login", json=body, timeout=30)
    if r.status_code != 200:  # first run: create the account
        requests.post(f"{api}/auth/register", json=body, timeout=30)
        r = requests.post(f"{api}/auth/login", json=body, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def upload(api, headers, path):
    with path.open("rb") as fh:
        r = requests.post(f"{api}/documents/upload", headers=headers,
                          files={"file": (path.name, fh)}, timeout=600)
    if r.status_code != 201:
        return path.name, None, f"HTTP {r.status_code}: {r.text[:120]}"
    return path.name, r.json()["document_id"], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--email", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--api", default="http://localhost:8000/api")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-wait", action="store_true", help="do not wait for chunking to finish")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.folder).rglob("*") if p.suffix.lower() in ALLOWED)
    if not files:
        sys.exit("No .pdf/.docx/.pptx/.md files found.")
    headers = {"Authorization": f"Bearer {get_token(args.api, args.email, args.password)}"}

    print(f"Uploading {len(files)} file(s)...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda p: upload(args.api, headers, p), files))

    pending = {}
    for name, doc_id, err in results:
        print(("FAIL  " if err else "OK    ") + name + (f"  {err}" if err else ""))
        if doc_id:
            pending[doc_id] = name
    if args.no_wait or not pending:
        return

    print("Waiting for chunking/embedding...")
    final = {}
    while pending:
        time.sleep(3)
        for doc_id, name in list(pending.items()):
            st = requests.get(f"{args.api}/documents/{doc_id}/status", timeout=30).json().get("status")
            if st in DONE:
                final[name] = st
                del pending[doc_id]
                print(f"{st.upper():9} {name}  ({len(final)}/{len(final) + len(pending)})")
    ok = sum(1 for s in final.values() if s == "embedded")
    print(f"Done: {ok} embedded, {len(final) - ok} failed.")


if __name__ == "__main__":
    main()