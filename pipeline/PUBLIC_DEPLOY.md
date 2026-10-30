# Exposing the Dashboard Publicly (Jetson → Internet)

This gets you a public HTTPS link to your live dashboard using **Cloudflare
Tunnel** — no port forwarding, no router config, no static IP needed, works
behind CGNAT. It's free and the URL doesn't change on restart if you set up
a named tunnel (steps below).

## Why not raw port forwarding
Port forwarding exposes your Jetson's port directly to the internet with no
HTTPS and (until now) no auth — a real target for scanners/bots within
hours of going live. Cloudflare Tunnel instead makes an outbound connection
from the Jetson to Cloudflare, so nothing is listening on your router at
all, and you get HTTPS automatically.

## 1. Set dashboard credentials (do this first — non-negotiable once public)

On the Jetson, before running the dashboard:
```bash
export DASHBOARD_USER=admin
export DASHBOARD_PASS=<pick-something-strong>
```
Add these two lines to `~/.bashrc` so they persist across reboots, or set
them in the systemd service file if you set that up (see the earlier
README's optional systemd section).

Without these set, the dashboard has NO auth — fine for local testing,
not once it's tunneled.

## 2. Install cloudflared on the Jetson

```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb
cloudflared --version
```

## 3. Quick option — temporary URL (no account needed)

Good for testing today:
```bash
# in one terminal, run your dashboard as usual
python3 app.py --models-dir ../models --device 0

# in a second terminal, tunnel port 5000
cloudflared tunnel --url http://localhost:5000
```
This prints a random `https://something.trycloudflare.com` URL — share
that link. It changes every time you restart the tunnel, so it's fine for
a demo today but not a permanent link.

## 4. Permanent option — named tunnel with your own URL (free Cloudflare account)

1. Sign up free at https://dash.cloudflare.com and add any domain you own
   (or get a free one — Cloudflare doesn't require a paid domain for this).
2. Authenticate:
   ```bash
   cloudflared tunnel login
   ```
3. Create a named tunnel:
   ```bash
   cloudflared tunnel create glass-defect-dashboard
   ```
4. Route it to a subdomain:
   ```bash
   cloudflared tunnel route dns glass-defect-dashboard dashboard.yourdomain.com
   ```
5. Create `~/.cloudflared/config.yml`:
   ```yaml
   tunnel: glass-defect-dashboard
   credentials-file: /home/<user>/.cloudflared/<tunnel-id>.json

   ingress:
     - hostname: dashboard.yourdomain.com
       service: http://localhost:5000
     - service: http_status:404
   ```
6. Run it:
   ```bash
   cloudflared tunnel run glass-defect-dashboard
   ```
   Now `https://dashboard.yourdomain.com` is a stable public link to your
   live dashboard, protected by the Basic Auth you set in step 1.

7. (Optional) Run the tunnel as a service so it survives reboots:
   ```bash
   sudo cloudflared service install
   sudo systemctl enable --now cloudflared
   ```

## 5. GitHub repo — for the code, not the live feed

GitHub itself can't run your Flask app or serve a live camera — it only
hosts files. So the repo holds the code + setup instructions; the actual
live link is the Cloudflare Tunnel URL from above. Put the tunnel URL in
your repo's README so visitors know where to click.

```bash
cd glass-defect-pipeline
git init
git add .
git commit -m "Hybrid PatchCore + YOLO glass defect dashboard"
```

Before pushing, create `.gitignore` (see included file) so you don't
accidentally commit model weight files (too large for a normal GitHub
repo — use Git LFS or a release asset if you want weights in the repo
too) or your dataset zips.

Then on GitHub: create a new repo, then:
```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

Add a line near the top of your repo's README like:
```
**Live dashboard:** https://dashboard.yourdomain.com
```

## Security reminders
- Rotate `DASHBOARD_PASS` periodically since it's now internet-facing.
- Cloudflare's free tier includes basic DDoS protection, which helps here.
- If this ever needs to be more than a demo (e.g. actual factory line
  monitoring accessible to your team), look at Cloudflare Access instead
  of Basic Auth — it adds proper login/identity in front of the tunnel
  with per-person access control, still free for small teams.
