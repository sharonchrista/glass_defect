# Pushing to GitHub + Enabling GitHub Pages

Same pattern as `lidar-drainage-pipeline`: public code repo + a GitHub
Pages landing page that links to wherever the actual live thing is running
(there, presumably a video/demo; here, your Cloudflare Tunnel URL).

## 1. Create the repo on GitHub

Go to github.com → New repository → name it `glass-defect-pipeline` →
Public → don't initialize with a README (you already have one).

## 2. Push your local project

```bash
cd glass-defect-pipeline

# Use the public-facing README as the main one GitHub shows
mv README.md SETUP.md
mv GITHUB_README.md README.md

git init
git add .
git commit -m "Hybrid PatchCore + YOLO glass defect detection pipeline"
git branch -M main
git remote add origin https://github.com/sharonchrista/glass-defect-pipeline.git
git push -u origin main
```

Your `.gitignore` already excludes model weights, dataset zips, and
Cloudflare credentials — double check `git status` before committing shows
nothing sensitive staged.

## 3. Enable GitHub Pages

1. On the repo page: Settings → Pages
2. Under "Build and deployment" → Source: **Deploy from a branch**
3. Branch: **main**, folder: **/docs**
4. Save

GitHub will build and publish within a minute or two at:
```
https://sharonchrista.github.io/glass-defect-pipeline/
```

## 4. Point the landing page at your live dashboard

Edit `docs/index.html`, find this line:
```html
<a class="btn" href="https://dashboard.yourdomain.com" target="_blank" rel="noopener">
```
Replace the URL with your actual Cloudflare Tunnel URL from
`PUBLIC_DEPLOY.md` (either the temporary `trycloudflare.com` one for a
quick demo, or your permanent named-tunnel domain). Commit and push —
GitHub Pages updates automatically within a minute.

```bash
git add docs/index.html
git commit -m "Update live dashboard link"
git push
```

## Result

- `https://github.com/sharonchrista/glass-defect-pipeline` — code, public
- `https://sharonchrista.github.io/glass-defect-pipeline/` — landing page
  with a button to the live dashboard
- The actual dashboard itself keeps running wherever it always does — on
  the Jetson, behind your Cloudflare Tunnel, with Basic Auth
