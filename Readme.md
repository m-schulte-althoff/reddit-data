(i) data-extraction-hugging-face.py:

Strict extractor for the exact Reddit window requested from
https://huggingface.co/datasets/open-index/arctic

Window interpreted literally as:
- 18 months before 2022-11-30  -> start = 2021-05-30 00:00:00 UTC
- 12 months after 2022-11-30   -> end   = 2023-11-30 23:59:59.999999 UTC
Implemented as:
    created_at >= 2021-05-30 00:00:00 UTC
    created_at <  2023-12-01 00:00:00 UTC

Outputs:
- arctic_exact_20210530_20231130/comments.parquet
- arctic_exact_20210530_20231130/submissions.parquet
- arctic_exact_20210530_20231130/manifest.json

Behavior:
- Verifies that every required month exists in the Hugging Face mirror.
- Aborts with a clear error if any month/type pair is missing.
- Otherwise exports exactly the requested rows.

Install:
    pip install duckdb huggingface_hub


---

(ii) data-extraction-arctic-shift.py:

Exact non-Hugging-Face fallback extractor for Reddit data from Arctic Shift.

What it does
------------
1. Connects to the Arctic Shift 2005-06 .. 2023-12 bundle torrent via its info hash.
2. Selectively downloads only these monthly files:
      comments/RC_2021-05.zst .. comments/RC_2023-11.zst
      submissions/RS_2021-05.zst .. submissions/RS_2023-11.zst
3. Streams the .zst JSONL files line-by-line.
4. Keeps only records with:
      2021-05-30 00:00:00 UTC <= created_utc < 2023-12-01 00:00:00 UTC
5. Writes exact filtered raw records to:
      output/comments_20210530_20231130.jsonl.zst
      output/submissions_20210530_20231130.jsonl.zst
      output/manifest.json

Why JSONL.zst output?
---------------------
This preserves the raw Arctic Shift records exactly, without forcing a schema during extraction.
You can convert to Parquet later if you want.

Install
-------
pip install libtorrent zstandard orjson