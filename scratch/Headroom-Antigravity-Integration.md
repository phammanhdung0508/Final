# Antigravity CLI ↔ Headroom Transparent Proxy Integration

## The Solution
You can force Antigravity (`agy`) to route its API traffic transparently through the local Headroom proxy by overriding its internal base URL using the `CLOUD_CODE_URL` environment variable.

### Command
To run a single command:
```bash
CLOUD_CODE_URL=http://127.0.0.1:8787 agy --prompt "Your prompt here"
```

To enable it persistently for your shell:
```bash
export CLOUD_CODE_URL=http://127.0.0.1:8787
```

## How It Works (The Investigation)
1. **The TLS Tunnel Problem:** By default, `agy` connects to `https://daily-cloudcode-pa.googleapis.com` using a TLS `CONNECT` tunnel. Standard proxy variables like `HTTPS_PROXY` fail because Headroom cannot terminate the TLS for the `daily` endpoint (resulting in a 404).
2. **The `CLOUD_CODE_URL` Override:** Through binary inspection, we discovered the hidden `CLOUD_CODE_URL` environment variable. Setting this changes `agy`'s base endpoint to plain HTTP directed at Headroom.
3. **Native Interception:** Because the traffic is now unencrypted HTTP sent to Headroom's port, Headroom successfully intercepts routes like `/v1internal:streamGenerateContent`.
4. **Automatic Upstream Routing:** Headroom recognizes the request as `client: antigravity` and `provider: gemini`, performs its structural token compression, and transparently proxies the request upstream to the production `cloudcode-pa.googleapis.com` servers. 
5. **Authentication:** The auth tokens `agy` generates for the daily environment are fully valid for the production endpoint, meaning you don't need any custom API keys.

You now have true API-level context compression for Antigravity, bypassing the token limits of the previous MCP tool approach!
