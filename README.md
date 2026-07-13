# Research MCP

Research MCP gives any Streamable HTTP MCP client a private set of high-level
tools for web research, focused URL investigation, read-only GitHub research,
and persistent semantic memory. The gateway authenticates and queues work;
separate workers perform search, bounded page extraction, browser automation,
ingestion, retrieval, and reranking.

## Run with Docker Compose

Requirements: Docker Engine with Compose v2.24.4 or newer, a 64-bit Linux host,
and outbound internet access for search and page retrieval. The configured
memory ceilings total about 15 GB for the base stack, so a 16 GB VPS is the
practical default; each optional Crawl4AI or reranker profile adds up to 4 GB.
The first build downloads Chromium and the first research request downloads the
configured embedding model.

All settings have runnable defaults, so the base stack starts with one command:

```console
docker compose up -d
```

The MCP endpoint is `http://127.0.0.1:8001/mcp`. It uses streamable HTTP. Only
that gateway port is published; Redis, Qdrant, SearXNG, and workers remain on
private Docker networks.

When `MCP_AUTH_TOKEN` or `MCP_AUTH_TOKENS_JSON` is set, MCP clients must send an
`Authorization: Bearer TOKEN` header on gateway requests. An empty policy
leaves only the loopback endpoint unauthenticated for local development; the
gateway refuses an unauthenticated non-loopback bind by default.

For a durable non-local installation, create `.env` from `.env.example`, set a
random `SEARXNG_SECRET` and `MCP_AUTH_TOKEN`, review the resource limits, and
terminate TLS at a reverse proxy in front of the gateway.

```console
cp .env.example .env
docker compose up -d --build --wait
docker compose ps
```

PowerShell equivalent:

```powershell
Copy-Item .env.example .env
docker compose up -d --build --wait
docker compose ps
```

## Publish to GitHub and deploy to a VPS

Create an empty private GitHub repository first. Then establish a repository
boundary inside this project directory before adding files. This matters if a
parent directory is already a Git repository.

```console
cd Research-MCP-main
git init
git branch -M main
git rev-parse --show-toplevel
git add .
git commit -m "Initial private research MCP"
git remote add origin git@github.com:YOUR_ACCOUNT/research-mcp.git
git push -u origin main
```

The `git rev-parse` output must be the `Research-MCP-main` directory itself.
`.env`, model data, vector data, Redis data, and artifacts are ignored and must
never be committed.

On an Ubuntu or Debian VPS, install Git and Docker Engine with the Compose
plugin from Docker's official repository, then verify `docker compose version`
reports v2.24.4 or newer. Clone and configure the service:

```console
git clone git@github.com:YOUR_ACCOUNT/research-mcp.git
cd research-mcp
cp .env.example .env
chmod 600 .env
echo "SEARXNG_SECRET=$(openssl rand -hex 32)"
echo "MCP_AUTH_TOKEN=$(openssl rand -base64 48)"
echo "CRAWL4AI_API_TOKEN=$(openssl rand -hex 32)"
```

Put the generated values in `.env` as `SEARXNG_SECRET`, `MCP_AUTH_TOKEN`, and
`CRAWL4AI_API_TOKEN`; use a different value for each credential.
Keep `MCP_BIND_ADDRESS=127.0.0.1`, and add the public hostname to the strict Host
allowlist, for example
`MCP_ALLOWED_HOSTS=127.0.0.1:*,localhost:*,[::1]:*,mcp-gateway:*,research.example.com`.
Entries must match the HTTP `Host` header exactly; the narrow `:*` suffix allows
any port only for the named host. Add that suffix to the public hostname too if
clients connect on a non-default HTTPS port.
Browser-based MCP clients that send an `Origin` header also require that exact
origin in `MCP_ALLOWED_ORIGINS`; server-to-server clients may leave it empty.
Review the resource limits, and then start:

```console
docker compose config --quiet
docker compose up -d --build --wait
docker compose ps
docker compose logs --tail=100 mcp-gateway research-worker
```

Do not publish port 8001 directly to the internet. Put Caddy, Nginx, Traefik,
or another TLS reverse proxy on the same VPS and proxy a dedicated HTTPS name
to `127.0.0.1:8001`. For example, a host Caddy configuration is:

```caddyfile
research.example.com {
    reverse_proxy 127.0.0.1:8001
}
```

Allow only SSH, HTTP, and HTTPS through the VPS firewall. The remote MCP URL is
then `https://research.example.com/mcp`; the client must still send the bearer
token. The proxy must allow long-lived streaming responses and retain the
`Authorization` header. To update later:

```console
git pull --ff-only
docker compose up -d --build --wait
```

Back up the Qdrant, Redis, and artifact named volumes before destructive Docker
maintenance. `docker compose down` preserves them; `docker compose down
--volumes` permanently deletes them.

## Architecture

```text
MCP client -> mcp-gateway -> Redis queue -> research-worker
                                      |-> SearXNG -> public web
                                      |-> direct bounded fetch -> public web
                                      |-> Unix socket -> isolated PDF parser (no network)
                                      |-> Unix socket -> isolated web-runner
                                      |                    |-> safe-egress -> public web
                                      |                    `-> isolated Crawl4AI (optional)
                                      |                                      `-> safe-egress
                                      |-> Qdrant + FastEmbed
                                      `-> job artifacts volume
```

The default stack includes:

| Service | Role | Host exposure |
| --- | --- | --- |
| `mcp-gateway` | MCP protocol, validation, queue submission, result polling | `127.0.0.1:8001` |
| `research-worker` | Search, crawl, extraction, browser automation, RAG | None |
| `redis` | Durable job queue and result state | None |
| `qdrant` | Persistent vector memory | None |
| `searxng` | Private metasearch with JSON output enabled | None |
| `pdf-runner` | Resource-bounded PDF parsing in a network-less container | None |
| `web-runner` | Network-isolated Playwright/Crawl4AI control process | None |
| `safe-egress` | DNS-pinned, public-only SOCKS5 broker for the browser sandbox | None |

Redis, Qdrant, and the optional reranker join only the internal backend network.
SearXNG performs search on a separate control network. The worker reaches the
browser and PDF parser only through separate Unix sockets. The PDF parser has no
network interface. Playwright and optional Crawl4AI run only on an internal
sandbox network and reach public targets through `safe-egress`, which rejects
private, loopback, link-local, and metadata destinations after DNS resolution.
SearXNG uses a file-backed settings mount so its root filesystem remains
read-only across Compose implementations; `SEARXNG_SECRET` overrides the
checked-in placeholder at runtime.
The gateway joins a separate edge network for its published port. Named volumes
retain Redis state, Qdrant vectors and snapshots, embedding models, and research
artifacts across replacement.

## MCP tools

The gateway exposes the following tools. `namespace` values create logical
partitions in Qdrant memory. They also become an MCP authorization boundary
when a token policy restricts its `namespaces` patterns.

| Tool | Purpose and important behavior |
| --- | --- |
| `research_web` | Run open-ended research. Modes are `quick`, `balanced`, `deep`, `technical`, `academic`, `local_only`, and `web_only`. It can request cross-source corroboration, include existing memory, and request planner synthesis. With the Compose Redis backend it queues the work, waits up to `MCP_SYNC_JOB_WAIT_SECONDS` (60 seconds by default), then returns either the result or a job ID while longer work continues. An exact same-owner request already queued or running is coalesced and returns that job ID immediately. |
| `investigate_url` | Investigate one public HTTP(S) URL with crawl, rendered-browser, scrolling, clicking, and network-data fallbacks. Modes are `auto`, `targeted`, `balanced`, and `exhaustive`; optional flags control ingestion, raw text, and diagnostics. |
| `start_research` | Queue durable `research_web` work and return a job ID immediately. Use this for clients with short tool-call timeouts. It requires `JOB_BACKEND=redis`, which Compose configures. |
| `research_job` | Operate on a durable job with `action=status`, `result`, or `cancel`. Status and result calls use bounded long polling (`wait_seconds`, default 15) to reduce model/tool polling loops. `include_full_result=false` returns compact Redis metadata instead of loading the complete artifact. Cancellation is cooperative and may not be instantaneous. |
| `get_research_artifact` | Read a returned relative `artifact_path` as bounded text. `max_chars` is clamped to 1,000-250,000 characters. |
| `query_memory` | Search Qdrant memory in a namespace, with `top_k` clamped to 1-30, and return reranked evidence. |
| `ingest_text` | Store supplied text in a namespace. Common credentials and tokens are redacted by default. Under token policies, `redact_secrets=false` requires the separate `memory:write:unredacted` scope. |
| `manage_sources` | `list`, inspect `stats`, or `delete` an ingested source within a namespace. Deletion removes its Qdrant chunks, not job artifact files. |
| `github_research` | Read-only GitHub API access. `search` searches `issues`, `code`, or `repositories`; `inspect` returns repository metadata and a prioritized recursive file tree; `read` returns a file or directory listing at an optional ref. |

For `github_research`, `repository` accepts `owner/name` or a GitHub repository
URL. Anonymous access is rate-limited. Set a least-privilege `GITHUB_TOKEN` for
private repositories, higher limits, and API operations that require
authentication. `GITHUB_MAX_FILE_CHARS` bounds returned file content, while
`max_results` is bounded to 30 search results or 1,000 inspected files. Set
`GITHUB_API_URL` only to a trusted GitHub Enterprise API because the configured
token is sent to that origin.

When `GITHUB_TOKEN` is configured, `GITHUB_ALLOWED_REPOSITORIES` is mandatory
and credentialed requests fail closed outside that allowlist. Repository-less
search is allowed only when the allowlist contains the exact global `*` entry.
Prefer explicit `owner/repository` entries. The same rule applies independently
to each multi-token client policy.

## Authentication and authorization

`MCP_AUTH_TOKEN` keeps the simple single-client configuration. It grants the
normal research, redacted memory-write, memory-delete, artifact-read, and
GitHub-read scopes. Restrict it with comma-separated
`MCP_ALLOWED_NAMESPACES` and `GITHUB_ALLOWED_REPOSITORIES` values.

For multiple clients, set `MCP_AUTH_TOKENS_JSON` to a one-line JSON object keyed
by bearer token. When it is present, it replaces `MCP_AUTH_TOKEN`. Example:

```dotenv
MCP_AUTH_TOKENS_JSON='{"alice-long-random-token":{"client_id":"alice","scopes":["research","memory:write","memory:delete","artifacts:read","github:read"],"namespaces":["alice-*"],"github_repositories":["owner/docs"]},"automation-long-random-token":{"client_id":"automation","scopes":["research","memory:write"],"namespaces":["automation"],"github_repositories":[]}}'
```

`client_id` is the stable job owner. Namespace patterns use shell-style globs
and are case-sensitive; repository patterns are case-insensitive. Every token
must have `research`. Optional scopes are:

- `memory:write`: ingest text after secret redaction.
- `memory:write:unredacted`: explicitly permit `redact_secrets=false`.
- `memory:delete`: delete vector-memory sources.
- `artifacts:read`: read artifacts owned by that client's jobs.
- `github:read`: use the read-only GitHub connector, still constrained by the
  token and server repository allowlists.

Durable jobs are tagged with the authenticated `client_id`; other clients
cannot inspect, retrieve, cancel, or read their artifacts. Jobs created before
ownership was enabled are denied while authentication is active. The temporary
`MCP_ALLOW_LEGACY_UNOWNED_JOBS=true` migration switch weakens this isolation
and should be removed after old jobs expire.

Generate tokens with a cryptographically secure random generator and protect
`.env` with restrictive filesystem permissions. Static bearer tokens are not a
replacement for TLS.

## Verification semantics

`verify=true` asks the pipeline to favor distinct source-owner domains and
report lexical/topical overlap across retrieved evidence. It does not perform
claim-level entailment, prove that apparently separate sites are independent,
or establish factual truth. The returned `verification` object states this
explicitly. Planner synthesis validates that every `[E#]` citation points to
available evidence, but it also does not automatically validate factual
entailment. Clients should treat these features as corroboration aids and keep
uncertainty visible when evidence is incomplete or conflicts.

## Durable jobs and artifacts

The Compose deployment uses Redis jobs and a shared artifact volume. A typical
asynchronous client flow is:

1. Call `start_research`, retain its `job_id`, and return control to the user.
2. In a later assistant turn, call `research_job(action="result", job_id=...)`.
   The call waits up to `wait_seconds`; if it remains nonterminal, honor
   `retry_after_seconds` instead of polling again in the same turn.
3. Set `include_full_result=false` when only compact Redis metadata is needed.
4. When a result or source includes `artifact_path`, call
   `get_research_artifact` to read a bounded copy later.

`research_web` and `investigate_url` use the same queue under Compose, but wait
for up to `MCP_SYNC_JOB_WAIT_SECONDS` (60 seconds by default) before returning a
nonterminal job ID. The worker continues the same full-depth job in the
background; this timeout only bounds how long the original MCP call stays open.
Exact active requests are coalesced by authenticated owner, job kind, and
canonical payload, so a model retry does not create duplicate physical work.
Workers
hold random per-attempt leases and heartbeat active jobs. Stale leases are
recovered both on startup and every `JOB_STALE_RECOVERY_INTERVAL_SECONDS`; only
the current lease owner may publish a result. Duplicate queue entries therefore
cannot execute the same lease concurrently. Queue payloads are capped by
`JOB_MAX_PAYLOAD_BYTES`, and queue admission is capped by `JOB_MAX_QUEUED` (`0`
disables the admission cap). A job that repeatedly loses a stale lease is
terminally failed after `JOB_MAX_ATTEMPTS` claims instead of retrying forever.
Before an ingesting attempt can dispatch, its compensation record is written to
Redis and confirmed on the local AOF with `WAITAOF`. The worker fails closed if
that confirmation does not arrive within
`JOB_INGESTION_WAITAOF_TIMEOUT_MS` (default 5000 ms). Setting the timeout to `0`
disables the fsync confirmation and weakens recovery from a Redis host crash.

Each successful attempt writes its full JSON result atomically under
`ARTIFACT_DIR/<job_id>/result-<lease-prefix>.json`; source snapshots may be
stored beside it. A worker that loses its lease cannot attach that artifact to
the job result, preventing an obsolete attempt from overwriting a newer one.
Compose fixes `ARTIFACT_DIR` to `/data/artifacts` and mounts the `artifacts`
named volume into both gateway and workers. Paths returned to clients are
relative, validated paths rather than host filesystem paths.

Redis job metadata expires `JOB_RESULT_TTL_SECONDS` after a terminal state
(default 30 days; `0` disables expiry). Artifact directories are independently
pruned when their newest file is older than `ARTIFACT_RETENTION_SECONDS`
(default 30 days), with scans every `ARTIFACT_CLEANUP_INTERVAL_SECONDS` (default
one hour). A retention value of `0` disables worker pruning. Queued and running
job directories are protected from cleanup. Unless metadata expiry is disabled,
`JOB_RESULT_TTL_SECONDS` must be at least `ARTIFACT_RETENTION_SECONDS` so an
authenticated client never retains a path after its ownership record expires.
Qdrant memory has a separate lifecycle and is not removed by artifact cleanup.
Ingestion is committed in two phases. Workers immediately revoke all chunks
from a cancelled, failed, or lease-lost attempt when Qdrant is available; if it
is not, the fsynced Redis compensation ledger retries the idempotent revocation
after recovery. Successful completion atomically disarms that ledger entry.
Attempt tombstones are permanent and are never removed by ordinary Qdrant
history pruning, so delayed writes cannot reactivate an abandoned attempt.
Lifecycle repair runs every
`QDRANT_LIFECYCLE_REPAIR_INTERVAL_SECONDS` (default one hour), resumes bounded
`QDRANT_LIFECYCLE_REPAIR_MAX_POINTS` scans across passes, and removes inactive
history after `QDRANT_HISTORY_RETENTION_SECONDS` (default 30 days; `0` disables
history pruning). Set the repair interval to `0` only when another process owns
lifecycle maintenance.

## Optional standalone RAG API

`api.py` exposes the Qdrant-backed RAG operations as HTTP routes for a split
deployment. It is not published by the default Compose stack. If it is run with
Uvicorn, set `RESEARCH_API_TOKEN` and send the same bearer token from the
consumer; management routes refuse unauthenticated access by default.
`RESEARCH_API_ALLOW_UNAUTHENTICATED=true` is intended only for a network-isolated
development service. Consumers set `USE_RESEARCH_API_RAG=true`,
`RESEARCH_API_URL`, `RESEARCH_API_TOKEN`, and bounded request, metadata, ingest,
and response limits. `RESEARCH_API_TOTAL_TIMEOUT_SECONDS` bounds the entire
streamed response, not only individual socket operations. Public remote API
URLs must use HTTPS unless `RESEARCH_API_ALLOW_INSECURE_HTTP=true` is explicitly
set for a trusted endpoint; private Docker service URLs may use HTTP. Leased
workers use the same token to invalidate an entire opaque ingestion attempt when
a remote request is cancelled or its outcome is uncertain; the API persists an
attempt tombstone so an in-flight late commit cannot become retrievable.

## Connect MCP clients

The Compose endpoint uses Streamable HTTP at
`http://127.0.0.1:8001/mcp`. If either authentication setting is non-empty,
configure the client to send its assigned token on every MCP request:

```text
Authorization: Bearer YOUR_MCP_AUTH_TOKEN
```

For LibreChat, add a Streamable HTTP server to `librechat.yaml` (store the token
through LibreChat's secret or environment interpolation facility in a real
deployment):

```yaml
mcpServers:
  research-mcp:
    type: streamable-http
    url: http://127.0.0.1:8001/mcp
    timeout: 300000
    requiresOAuth: false
    headers:
      Authorization: "Bearer YOUR_MCP_AUTH_TOKEN"
```

For an OpenWebUI release with native MCP support, add an external MCP server in
its admin connections/tools UI, select **Streamable HTTP**, use the same URL,
and add the same `Authorization` header. Some OpenWebUI releases expose only
OpenAPI tool servers; those require an MCP-to-OpenAPI bridge such as `mcpo`.
Keep that bridge private and configure the bearer header between it and this
gateway.

### Container-to-container clients

When LibreChat or OpenWebUI runs in a container, `127.0.0.1` refers to that
client container, not the Docker host. The optional
`docker-compose.client-network.yml` override attaches only `mcp-gateway` to an
existing external Docker network. Redis, Qdrant, workers, search, and browser
services remain on the base stack's private networks.

Set these values in `.env`:

```dotenv
COMPOSE_FILE=docker-compose.yml:docker-compose.client-network.yml
COMPOSE_PATH_SEPARATOR=:
MCP_CLIENT_NETWORK=the-existing-client-network
MCP_CLIENT_ALIAS=research-mcp
MCP_AUTH_TOKEN=a-long-random-secret
```

The named network must already exist and the MCP client must also join it. The
alias must be unique on that network. Keep `MCP_BIND_ADDRESS=127.0.0.1` so the
base deployment remains safe if the override is later removed. The override
removes the host port publication and exposes the gateway only through its
Docker network alias. `COMPOSE_FILE` makes every subsequent `docker compose`
command retain both files, including profile and scaling commands.
`COMPOSE_PATH_SEPARATOR` keeps that setting portable across host operating
systems. Start the stack with:

```console
docker compose config --quiet
docker compose up -d --build --wait
```

The override makes authentication mandatory and assigns unique backend aliases
for gateway database connections. This prevents an unrelated `redis`, `qdrant`,
or `reranker` on the shared client network from being selected by Docker DNS.
It also removes every host port publication; only containers on the external
client network can reach the gateway. Do not attach any other Research MCP
service to that external network.

Configure the client for Streamable HTTP at
`http://research-mcp:8001/mcp`, or replace the hostname with the value of
`MCP_CLIENT_ALIAS`. Plain HTTP is appropriate only for a trusted, same-host
Docker network; use a TLS reverse proxy across hosts or untrusted networks.
Client tool-call timeouts should exceed `MCP_SYNC_JOB_WAIT_SECONDS`. Clients
with shorter limits should use `start_research`, return the job ID, and call
`research_job` in a later turn; they should not busy-poll within one model turn.

LibreChat also blocks private MCP destinations by default. Allow the exact
private socket rather than disabling SSRF protection:

```yaml
mcpSettings:
  allowedAddresses:
    - "research-mcp:8001"
```

LibreChat ignores `allowedAddresses` when `mcpSettings.allowedDomains` is also
configured. In that case, make the equivalent narrow entry in the authoritative
domain allowlist or remove that setting and use the exact address above.

Without the override, the portable base deployment remains available at
`http://127.0.0.1:8001/mcp` for host-local clients and reverse proxies. Remote
clients should use a TLS reverse proxy instead of a shared Docker network. Leave
`COMPOSE_FILE` and `COMPOSE_PATH_SEPARATOR` unset for that base deployment.

## Optional profiles

Crawl4AI improves structured extraction. The reranker improves final evidence
ordering. Both are optional because Playwright/direct HTTP and Qdrant vector
order are built-in fallbacks.

The Crawl4AI image requires its separate `CRAWL4AI_API_TOKEN` in order to bind
beyond its own loopback interface. Compose passes that token only to Crawl4AI
and the isolated `web-runner`, which sends it as a bearer credential. Replace
the placeholder in `.env` before enabling the profile on a VPS. Compose builds
`CRAWL4AI_DERIVED_IMAGE` from the pinned upstream image and overlays only its
localhost pinning proxy. That proxy resolves and pins public destinations, then
tunnels the pinned IP through `safe-egress`; the Crawl4AI container remains on
the internal-only browser network. Always include `--build` after changing the
upstream image pin or overlay.

```console
# Start either profile
docker compose --profile crawl4ai up -d --wait
docker compose --profile reranker up -d --wait

# Start both
docker compose --profile crawl4ai --profile reranker up -d --wait
```

The reranker profile downloads `RERANKER_MODEL` on first start and can require
several gigabytes of memory and disk. The default CPU image is amd64-only; leave
this profile disabled on arm64 hosts unless you configure a compatible image or
external reranking service.

## Operations

Inspect readiness and logs:

```console
docker compose ps
docker compose logs -f mcp-gateway research-worker
```

Scale workers when direct crawling, embedding, or ingestion jobs queue faster
than they complete:

```console
docker compose up -d --scale research-worker=3
```

Each worker processes up to `RESEARCH_SOURCE_CONCURRENCY` sources concurrently.
The default is `2`; valid values are `1` through `4`. Increase it only when the
worker has enough CPU and memory for concurrent extraction, embedding, and
ingestion.

For ordinary static pages, direct extraction receives a short
`DIRECT_FIRST_HEDGE_SECONDS` head start. Results must pass the conservative
`DIRECT_FIRST_MIN_CONTENT_CHARS` quality gate; thin, blocked, or dynamic pages
must also contain enough primary page content after navigation chrome is removed.
Thin, blocked, boilerplate-only, or dynamic pages still escalate to Crawl4AI and
the rendered-browser fallback. Research source fallbacks reuse their first crawl
result instead of fetching the same URL twice.

Workers share one serialized `web-runner`, so worker replicas do not increase
rendered-browser concurrency. Scale only after budgeting `WORKER_MEMORY_LIMIT`
and `WORKER_CPUS` for every replica; scaling the browser tier requires a separate
socket/routing design rather than this command.

Stop containers without deleting research state:

```console
docker compose down
```

Deleting named volumes permanently removes queue state, vector memory, cached
models, and artifacts:

```console
docker compose down --volumes
```

## Configuration

Compose reads `.env` automatically. `.env.example` documents image pins,
gateway binding and synchronous wait timing, queue timing,
embedding/vector settings, optional backends,
GitHub/planner credentials, browser safety switches, and resource bounds.
Important invariants:

- `VECTOR_SIZE` must match the selected `EMBEDDING_MODEL`.
- `MCP_BIND_ADDRESS` defaults to loopback. `MCP_AUTH_TOKEN` enables static bearer
  authentication, but the application does not provide TLS. Require the token
  and terminate TLS at a reverse proxy before publishing to an untrusted
  network.
- Redis uses `noeviction` so queued work is never silently evicted. Monitor its
  volume and memory, especially with long result TTLs.
- Keep `RESEARCH_BROWSER_DISABLE_SANDBOX=false` and
  `RESEARCH_BROWSER_IGNORE_HTTPS_ERRORS=false` unless a controlled environment
  has a documented compatibility requirement.
- `RESEARCH_BROWSER_SANDBOX_MODE=auto` first launches Chromium with its native
  sandbox. Only a recognized host denial of Chromium's user namespace permits a
  compatibility retry without that inner sandbox; the dedicated web runner
  remains non-root, read-only, capability-free, seccomp-filtered,
  resource-limited, isolated behind `safe-egress`, and subject to Docker's
  AppArmor profile where AppArmor is available. Set the mode to `required` after
  loading a host AppArmor policy that grants `userns`. `disabled` and the
  legacy `RESEARCH_BROWSER_DISABLE_SANDBOX=true` switch are explicit emergency
  overrides. Do not add `SYS_ADMIN` or change host-wide user-namespace policy.
- Successful job artifacts are pruned after `ARTIFACT_RETENTION_SECONDS` by
  default. Set it to `0` only when another process owns artifact lifecycle, and
  keep the Redis result TTL at least as long as artifact retention.
- Treat `.env` as a secret when it contains `MCP_AUTH_TOKEN`, `GITHUB_TOKEN`, or
  `PLANNER_API_KEY`; it is excluded from the container build context.
- Image defaults are exact tags. For a controlled production rollout, mirror
  them and optionally replace tags with digest-pinned image references.
- The supplied Crawl4AI profile is deliberately inside the browser sandbox. An
  external Crawl4AI deployment requires an equivalent public-only egress and
  authentication design. The reranker URL may point to an external service.
- `SAFE_EGRESS_ALLOWED_PORTS` applies to both direct fetches and browser traffic.
  Keep it limited to required web ports. Use `SAFE_EGRESS_DENY_CIDRS` to block
  otherwise-public addresses owned by this VPS, its provider control plane, or
  other administrative networks that web research must never contact.

## Limitations

- Static token policies are not a replacement for TLS or a full identity
  provider. Multi-token policies do enforce namespace, job-owner, artifact, and
  GitHub-repository boundaries; the legacy single token can also restrict its
  allowed namespaces. A wildcard namespace policy intentionally grants access
  to every namespace.
- URL investigation accepts only public `http` and `https` targets. Loopback,
  private, link-local, metadata-service, credential-bearing, and non-HTTP URLs
  are rejected, including hostnames with any non-public DNS answer. This
  intentionally prevents intranet and local-development URL research.
- Extraction is heuristic. Authentication walls, CAPTCHAs, paywalls, unusual
  shadow-DOM or iframe interactions, and aggressive anti-automation can leave
  evidence incomplete. The system does not bypass access controls.
- Chromium work is serialized in the shared isolated `web-runner` and is
  memory-heavy. Additional workers improve non-browser throughput but do not
  multiply browser capacity in the supplied topology.
- Durable cancellation is cooperative. Redis job metadata and artifact files
  have independent expiry, and deleting the named volumes is irreversible.
- GitHub access is read-only and subject to GitHub API permissions and rate
  limits. Recursive trees and large files can be truncated; it does not clone a
  repository or expose arbitrary Git history.
- Planner synthesis is disabled unless an OpenAI-compatible planner endpoint,
  model, credentials, and `PLANNER_ENABLE_SYNTHESIS=true` are configured.

## Native Python dependencies

The container installs the repository's `requirements.txt`. The queued
gateway/worker implementation also requires the Redis Python client declared by
that file. Playwright's Chromium and operating-system libraries are installed
in the runtime image. SearXNG, Qdrant, Crawl4AI, and the reranker are network
services, not Python packages embedded in the application container.
