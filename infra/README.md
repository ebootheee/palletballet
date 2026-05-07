# PalletBallet — home-server deployment

Runs the FastAPI service behind a Cloudflare Tunnel, with Watchtower auto-pulling new images from GHCR on every push to `main`.

```
                 boothe.io           palletballet-api.boothe.io
                     │                          │
                     ▼                          ▼
              Cloudflare Worker         Cloudflare Tunnel
              (bootheio-main,                   │
               serves Astro page)               ▼
                                       home server (Docker):
                                         ┌──────────────────┐
                                         │ cloudflared      │
                                         │ palletballet     │ ← FastAPI :8000
                                         │ watchtower       │ ← pulls GHCR
                                         └──────────────────┘
```

## One-time setup

### 1. Create the tunnel

In Cloudflare Zero Trust → **Networks** → **Tunnels** → **Create a tunnel**:

- Connector: `Cloudflared`
- Name: `palletballet`
- Save → copy the token shown in the Docker install snippet.

In **Public Hostname**:

- Subdomain: `palletballet-api`
- Domain: `boothe.io`
- Type: `HTTP`
- URL: `palletballet:8000`

(Cloudflare auto-creates the DNS CNAME for `palletballet-api.boothe.io`.)

### 2. Make the GHCR package public (one-time)

After the first GitHub Actions build succeeds, the image lives at `ghcr.io/ebootheee/palletballet`. By default, GitHub creates new packages as **private**, so the server can't pull anonymously. Fix in one click:

- Go to <https://github.com/users/ebootheee/packages/container/palletballet/settings>
- Scroll to "Danger Zone" → Change visibility → **Public**

(Alternatively, run `docker login ghcr.io` on the server with a PAT — but public is simpler for an open-source demo.)

### 3. Bring up the stack on the server

```bash
git clone https://github.com/ebootheee/palletballet.git
cd palletballet/infra
cp .env.example .env
# paste TUNNEL_TOKEN into .env
docker compose up -d
```

### 4. Verify

```bash
curl https://palletballet-api.boothe.io/healthz
# {"status":"ok","version":"0.1.0"}
```

## How updates flow

1. `git push` to `main` on this repo
2. GitHub Actions runs tests, then builds and pushes `ghcr.io/ebootheee/palletballet:latest`
3. Watchtower (running on the server, polling every 5 min) sees the new digest, pulls it, recreates the `palletballet` container
4. The `cloudflared` and `watchtower` containers are pinned to whatever you started — they don't auto-update (only containers with the `com.centurylinklabs.watchtower.enable=true` label do, which is just `palletballet`)

## Manual operations

```bash
docker compose pull palletballet && docker compose up -d palletballet   # force update now
docker compose logs -f palletballet                                      # tail API logs
docker compose logs -f cloudflared                                       # tail tunnel logs
docker compose down                                                      # stop everything
```

## Rate limiting

The Cloudflare WAF rule for `palletballet-api.boothe.io` is configured at the zone level (Security → WAF → Rate limiting rules). Default: **60 requests / minute / IP**, action: managed challenge. Tune in dashboard.

## CORS

The API allows `https://boothe.io`, `https://www.boothe.io`, and `http://localhost:4321` by default. Override with the `ALLOWED_ORIGINS` env var (comma-separated) in `docker-compose.yml`.
