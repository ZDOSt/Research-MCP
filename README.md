# Private Search Gateway

This project is a deterministic, self-hosted search and page-retrieval backend
for AI frontends. It exposes a fast SearXNG-compatible discovery endpoint, a
bounded integrated search route, and Firecrawl-compatible scrape/search routes.
The gateway discovers sources with a small set of search engines, opens pages
only when the selected route asks for it, extracts their actual contents,
reranks evidence locally, and returns source URLs and page-derived text.

Search results also carry deterministic evidence metadata: an inferred source
type and tier, an authority score, normalized page-declared dates, version
markers, and a stable citation ID/URL. These are ranking and coverage aids, not
claims that the gateway has proved source ownership or verified every claim.

It does not call a paid search API or an internal language model. The frontend's
model receives the retrieved evidence and writes the answer. This keeps the
service private, predictable, and usable by any frontend that accepts a custom
SearXNG or JSON search provider.

## What runs

- `search-gateway`: the only client-facing service, on internal port `8080`
- `searxng`: web, technical, news, image, and research discovery
- `reranker`: local `BAAI/bge-reranker-base` relevance ranking
- `crawl4ai`: JavaScript-aware crawling for difficult pages
- `web-runner`: isolated Crawl4AI and Playwright control over a Unix socket
- `pdf-runner`: network-isolated PDF extraction
- `safe-egress`: blocks private-network and metadata destinations for browsers
- `redis`: response cache and stale-result fallback

The stack publishes no host ports. A frontend reaches it over a shared Docker
network at:

```text
http://search-gateway:8080/search
```

Every container attached to that shared network can call the gateway. Use a
dedicated shared network if other unrelated containers should not have access.

## Requirements

- 64-bit Linux VPS
- Docker Engine and Docker Compose v2.24.4 or newer
- About 10 GB free disk for images, Chromium, and the reranker model
- 16 GB RAM recommended for the complete stack

The supplied ceilings total about 10.5 GB, excluding shared memory and normal
Docker overhead. They are limits, not reservations, but leave useful headroom
on a 16 GB host. The first build is slow because it downloads Chromium, the
Crawl4AI image, and the reranker model.

## Clean installation

Create the Docker network once if it does not already exist:

```console
docker network inspect docker-stacks_app-network >/dev/null 2>&1 || \
  docker network create docker-stacks_app-network
```

Clone and configure the project:

```console
git clone https://github.com/ZDOSt/Research-MCP.git
cd Research-MCP
cp .env.example .env
chmod 600 .env
```

Generate three different secrets:

```console
openssl rand -hex 32
openssl rand -hex 32
openssl rand -hex 32
```

Edit `.env` and replace `SEARXNG_SECRET`, `CRAWL4AI_API_TOKEN`, and
`FIRECRAWL_API_KEY` with three different values. Change
`CLIENT_DOCKER_NETWORK` only if your frontend uses a different external Docker
network.

Validate and start the complete stack:

```console
docker compose config --quiet
docker compose up -d --build --wait
docker compose ps
```

No `ports:` entries are needed. Do not add one unless you intentionally want to
expose the gateway outside Docker.

## Verify it

Run a health check from the gateway container:

```console
docker compose exec -T search-gateway python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/healthz').read().decode())"
```

Run a real search from any container on the shared network. Replace
`your-frontend-container` with `anythingllm`, `librechat`, or another container
name:

```console
docker exec your-frontend-container sh -lc \
  "wget -qO- 'http://search-gateway:8080/search?q=how+to+install+docker+compose&format=json' | head -c 1000"
```

The response should contain `results`, source URLs, search snippets, and
diagnostics. This endpoint intentionally does not crawl pages, so it returns
quickly and is suitable for standard SearXNG integrations.

## Frontend setup

Use the following base URL wherever the frontend asks for a SearXNG URL:

```text
http://search-gateway:8080
```

If it asks for the complete search path, use:

```text
http://search-gateway:8080/search
```

AnythingLLM requires the complete discovery path even though its field is
labeled `SearXNG API Base URL`. Configure it as:

```text
http://search-gateway:8080/search
```

Use `http://search-gateway:8080/integrated/search` in that field instead when
you specifically want AnythingLLM's search request to include a bounded crawl.

The standard request is:

```http
GET /search?q=your+question&format=json
```

Supported query parameters include:

- `language=auto`
- `time_range=day|week|month|year`
- `categories=general,it,news,science,images`
- `max_results=1..8`
- `mode=auto|quick|balanced|deep`

When no category is supplied, the gateway infers useful SearXNG categories from
the request. `auto` uses quick mode for simple lookups and balanced mode for
technical questions and recommendations. The discovery route never crawls.

### Integrated search and Firecrawl compatibility

For a frontend that has one combined search/scraper setting, use the bounded
integrated route:

```text
http://search-gateway:8080/integrated/search
```

It performs SearXNG discovery, crawls at most the configured number of top
pages, and returns page-derived `content`. Its default timeout and crawl budget
are intentionally smaller than `/v1/research`.

For LibreChat, Open WebUI, or LobeChat Firecrawl settings, use this API base:

```text
http://search-gateway:8080
```

The gateway implements `POST /v2/scrape` and `POST /v2/search`. Set the
frontend's Firecrawl API key to the same value as `FIRECRAWL_API_KEY` in the
gateway `.env`. The scraper accepts the common Markdown request and returns
`success`, `data.markdown`, and `data.metadata.sourceURL`. The Firecrawl routes
require a Bearer token and use the existing URL validation, direct extraction,
Crawl4AI, Playwright, and PDF isolation controls.

The gateway also implements a Jina-compatible `POST /v1/rerank` endpoint backed
by the stack's local BGE reranker. It uses the same Bearer credential as the
Firecrawl routes. This adapter lets LibreChat include relevant passages from
scraped pages in the model-visible Web Search result without a hosted reranking
service. If the local model is unavailable or exceeds its bounded deadline, the
adapter returns lexical fallback passages instead of an empty result.

Queries containing an explicit HTTP or HTTPS URL bypass SearXNG discovery. The
supplied URL is returned as the deterministic direct discovery result and
then passes through the same authenticated Firecrawl-compatible scraper. This
prevents direct-page requests from depending on whether a search engine happens
to index the supplied URL.

The `/v1/research` and `/integrated/search` responses include an
`evidence_summary` with independent-domain coverage, likely primary-source
coverage, date and extraction coverage, version context, and explicit warnings
when evidence is thin. Source classification is based on transparent domain,
path, and query-affinity heuristics. It never represents itself as claim-level
verification or proof that two domains are organizationally independent.

Use these internal Docker URLs:

| Frontend | Search setting | Scraper setting | Reranker setting |
| --- | --- | --- | --- |
| AnythingLLM | `http://search-gateway:8080/search` or `/integrated/search` | Use the integrated route when its separate scraper cannot be changed | Included in integrated search |
| LibreChat | SearXNG base `http://search-gateway:8080` | Firecrawl base `http://search-gateway:8080` | Jina URL `http://search-gateway:8080/v1/rerank` |
| Open WebUI | SearXNG query URL `http://search-gateway:8080/search?q=<query>&format=json` | `FIRECRAWL_API_BASE_URL=http://search-gateway:8080` | Configure separately in the frontend |
| LobeChat | Configure its preferred search provider separately | `FIRECRAWL_URL=http://search-gateway:8080/v2` | Configure separately in the frontend |

For Open WebUI select the Firecrawl web loader and set `FIRECRAWL_API_KEY`.
For LobeChat include Firecrawl in `CRAWLER_IMPLS` and set the same key. For
LibreChat select SearXNG as the search provider and Firecrawl as the scraper;
permit the private `search-gateway` address in its web-search allowlist.

Examples:

```http
POST /v2/scrape
Authorization: Bearer <FIRECRAWL_API_KEY>
Content-Type: application/json

{"url":"https://example.com","formats":["markdown"]}
```

```http
POST /v2/search
Authorization: Bearer <FIRECRAWL_API_KEY>
Content-Type: application/json

{"query":"how to install Docker Compose","limit":5}
```

```http
POST /v1/rerank
Authorization: Bearer <FIRECRAWL_API_KEY>
Content-Type: application/json

{
  "model": "jina-reranker-v2-base-multilingual",
  "query": "Docker Engine Ubuntu installation commands",
  "documents": ["first passage", "second passage"],
  "top_n": 5,
  "return_documents": true
}
```

For direct integrations, the full bounded research endpoint is also available:

```http
POST /v1/research
Content-Type: application/json

{
  "query": "What are the recommended settings for an AW3426DW?",
  "mode": "balanced",
  "max_results": 5,
  "language": "auto",
  "categories": []
}
```

## Updating

From the repository directory on the VPS:

```console
git pull --ff-only
docker compose config --quiet
docker compose up -d --build --remove-orphans --wait
docker compose ps
```

You do not need to run `docker compose down` for a normal update. Existing Redis
cache and reranker downloads remain in named volumes.

## Operations

Useful commands:

```console
docker compose ps
docker compose logs --tail=200 search-gateway searxng reranker
docker compose logs --tail=200 crawl4ai web-runner safe-egress pdf-runner
docker compose restart search-gateway
docker compose down
docker compose up -d --wait
```

### Repeatable quality baseline

The opt-in live evaluation suite covers installation guides, errors, hardware
settings, recommendations, version-sensitive questions, direct URLs,
multi-source comparisons, gaming, academic material, and current news. It is
not run automatically and adds no latency to normal gateway requests.

From inside the running gateway container:

```console
docker compose exec -T search-gateway python evaluate_search_quality.py \
  --base-url http://127.0.0.1:8080
```

Run one case while tuning with `--case docker-compose-install`, or retain the
JSON report with `--output /tmp/search-quality-report.json`. See
`evals/README.md` for the measured fields and limitations.

`docker compose down` preserves named volumes. `docker compose down -v` deletes
the cache and downloaded reranker model and should be used only for a deliberate
full reset.

## Limitations

This can approach hosted search tools for documentation, troubleshooting,
product settings, games, current information, and general research, but it
cannot guarantee the same coverage as commercial providers. Keyless engines may
rate-limit datacenter IPs, some sites block all automated browsers, and no
single VPS has the proprietary search indexes used by Google, Brave, or paid
answer engines. The gateway compensates with multiple discovery providers,
concurrent extraction, local reranking, bounded browser fallbacks, caching, and
honest partial results rather than inventing an answer.
