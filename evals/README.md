# Search quality evaluation

This suite is an opt-in live baseline for the gateway. It measures successful
retrieval, latency, source diversity, likely primary-source coverage, declared
date coverage, page extraction, citation URLs, and a small number of expected
domain/content signals. It does not use an LLM to grade answers and does not
claim that a retrieved passage proves a factual claim.

Run it from a machine or container that can reach the gateway:

```console
python evaluate_search_quality.py --base-url http://search-gateway:8080
```

Run one case while tuning:

```console
python evaluate_search_quality.py --base-url http://search-gateway:8080 \
  --case docker-compose-install
```

Use `--output report.json` to retain a report. The default concurrency is one
so the baseline measures the normal pipeline without intentionally saturating
the VPS. Live evaluation is deliberately excluded from automated tests because
search indexes, sites, dates, and network conditions change.
