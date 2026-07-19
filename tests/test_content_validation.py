"""Tests for manual validation sampling of content proxies."""

from __future__ import annotations

from pathlib import Path

from src.content_validation import analyse_content_validation_sample, create_content_validation_sample
from tests.conftest import write_zst_jsonl


def test_content_validation_sample_is_stratified_and_coding_ready(tmp_path: Path) -> None:
    comments_path = tmp_path / "comments.zst"
    submissions_path = tmp_path / "submissions.zst"
    records = [
        {
            "id": "general-pre",
            "subreddit": "askreddit",
            "body": "Why does this happen?",
            "created_utc": 1661990400,
        },
        {
            "id": "health-post",
            "subreddit": "depression",
            "body": "I have symptoms and need support.",
            "created_utc": 1667433600,
        },
    ]
    write_zst_jsonl(comments_path, records)
    write_zst_jsonl(submissions_path, [
        {
            "id": "general-post",
            "subreddit": "askreddit",
            "title": "Question",
            "selftext": "Anyone have experience?",
            "created_utc": 1667433600,
        },
    ])

    result = create_content_validation_sample(
        [comments_path],
        [submissions_path],
        per_stratum=1,
        tables_dir=tmp_path / "tables",
    )

    assert len(result.sample) == 3
    assert set(result.sample["manual_question"]) == {""}
    assert result.coverage["sampled_records"].eq(1).all()
    assert result.table_paths["sample"].exists()

    sample = result.sample.copy()
    sample["manual_question"] = sample["proxy_question"]
    sample["manual_experience"] = sample["proxy_experience"]
    sample["manual_support"] = sample["proxy_support"]
    sample["manual_medical"] = sample["proxy_medical"]
    sample.to_csv(result.table_paths["sample"], index=False)
    report = analyse_content_validation_sample(tables_dir=tmp_path / "tables")

    assert report["n_coded"].gt(0).all()
    assert (tmp_path / "tables" / "content-validation-report.csv").exists()
    assert (tmp_path / "tables" / "content-validation-coding-coverage.csv").exists()