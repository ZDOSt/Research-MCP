# Private Search Gateway

This project is a deterministic, self-hosted search and page-retrieval backend
for AI frontends. It exposes a SearXNG-compatible search endpoint, discovers
sources with a small set of search engines, opens the strongest pages, extracts
their actual contents, follows a bounded number of relevant same-site links,
reranks evidence locally, and returns cited source URLs and page-derived text.

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

Generate two different secrets:

```console
openssl rand -hex 32
openssl rand -hex 32
```

Edit `.env` and replace `SEARXNG_SECRET` and `CRAWL4AI_API_TOKEN` with those
values. Change `CLIENT_DOCKER_NETWORK` only if your frontend uses a different
external Docker network.

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

The response should contain `results`, source URLs, extracted `content`, and
diagnostics. Search snippets are used only as a clearly labeled fallback when a
site blocks extraction or the request deadline is reached.

## Frontend setup

Use the following base URL wherever the frontend asks for a SearXNG URL:

```text
http://search-gateway:8080
```

If it asks for the complete search path, use:

```text
http://search-gateway:8080/search
```

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
technical questions and recommendations.

For direct integrations, a richer JSON endpoint is also available:

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
