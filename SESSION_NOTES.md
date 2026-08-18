# Session Notes — VGG19 Transfer Learning Portfolio

Continuation notes for picking this project up in a new session (different machine/agent).
Written 2026-08-18.

## What this project is

`vgg19-transfer-learning-cifar10/` is a feature-extraction transfer learning exercise:
a pretrained VGG19 (ImageNet weights) with its convolutional base frozen, plus a new
classifier trained on top, to classify CIFAR-10 (10 classes).

**Source**: adapted from the [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course/unit0/welcome/TableOfContents)
(repo: `johko/computer-vision-course`, MIT License, Copyright (c) 2023 Johannes Kolbe).
Verified by fetching the repo's `LICENSE` file directly — MIT permits reuse/redistribution
with the original notice preserved, so publishing this as a portfolio piece is fine as
long as attribution + LICENSE are kept (already done, see repo root `LICENSE`/`README.md`).

## Where things live

- **Working notebook (actually being run)**: on the deep learning GPU server, at
  `~/coding/learning/vgg19_transfer/transfer_learning_vgg19.ipynb`. Windows-side this is
  reached via a local symlink at
  `C:\Users\devuser\my-github-repository\en683_lpr\vgg19_transfer` → SSHFS `Z:\coding\learning\vgg19_transfer`.
- **Portfolio repo (this repo)**: `~/coding/dl-portfolio/` on the same server, git remote
  `origin` = `https://github.com/leeyunhome/dl-portfolio.git`, branch `main`.
  GitHub Pages is enabled (Settings → Pages → Deploy from branch, `main` / root),
  live at `https://leeyunhome.github.io/dl-portfolio/`.
- Portfolio repo intentionally does **not** live inside `en683_lpr` — that repo mixes in
  company production LPR work (server IPs, OCR training data, product binaries), so it
  was kept fully separate to avoid exposing anything company-confidential.
- Server access: SSH to `deeplearning@10.10.237.222` (key-based, no password). See the
  `dl-server` skill for connection details, SSHFS mount caveats, etc.
- A Python venv with `jupyter`/`nbconvert` (used to render the notebook to HTML) exists
  at `~/coding/learning/yolov8_training/venv` on the server — borrowed for this purpose
  even though it's a different project's venv.

## Trial-and-error log (things that weren't obvious)

1. **CIFAR-10 download is slow** — the `torchvision.datasets.CIFAR10(download=True)`
   call pulls from `cs.toronto.edu`, which was slow from this network (~35+ min for
   ~170MB). Not a code problem, just external bandwidth.
2. **`root='./data'` is relative to the kernel's cwd, not the notebook's own folder.**
   The Jupyter kernel's working directory turned out to be `~/coding/learning/`, so the
   downloaded data landed in `~/coding/learning/data/`, not
   `~/coding/learning/vgg19_transfer/data/`. Caused confusion when checking "did it
   download yet" by looking in the wrong directory.
3. **Known unfixed bug, left intentionally**: in the "change the number of classes" cell,
   ```python
   pre_vgg19.classifier[6].out_features = NUM_CLASSES
   ```
   only overwrites the attribute — it does **not** resize the actual `nn.Linear(4096, 1000)`
   weight matrix. The correct fix would be:
   ```python
   pre_vgg19.classifier[6] = nn.Linear(4096, NUM_CLASSES)
   ```
   This is called out in a code comment but left as-is to match the original tutorial.
   **Open question for whoever continues this**: decide whether to actually apply the fix
   before running real training — as-is, the model still has a 1000-way output head, which
   will train "successfully" (loss goes down) but isn't really doing correct 10-class
   classification, and inference (`classes[preds[0]]`) can `IndexError` if the argmax lands
   outside 0-9.
4. **Comment/code mismatch**: a comment says "Display the first 8 images from the batch"
   but the code slices `inputs[:5]` / `labels[:5]` — only 5 images are actually shown.
5. **Inference cell's image path was Kaggle-only** (`/kaggle/input/validation-data/cat.jpg`),
   changed to a local relative path `img_path="cat.jpg"` — **still needs an actual local
   image file placed next to the notebook** before that cell will run.
6. Verified (via `plot_class_distribution` + bar charts, both text `Counter` output and
   the rendered PNGs) that CIFAR-10 is perfectly class-balanced: 5000 images/class in
   train, 1000 images/class in validation. No re-balancing needed.

## Current execution status (verified directly from the notebook's saved JSON, not assumed)

Done and executed with real outputs:
- Imports, device selection (`cuda`)
- CIFAR-10 download + `train_set`/`validation_set` creation
- Class distribution check (text + 2 bar chart PNGs, both balanced)
- `batch_size=128` DataLoaders (`train_loader`, `validation_loader`) built
- Batch shape check (`[128, 3, 224, 224]`)
- Sample image visualization (5 images with class-name titles)

**Not yet executed** (defined in code but `execution_count=None`, no outputs):
- `learning_rate = 1e-4` and all helper functions (`get_lr`, `metrics_batch`,
  `loss_batch`, `loss_epoch`, `train_val`, `plot_out`)
- Loading `pre_vgg19 = models.vgg19(weights='IMAGENET1K_V1')`
- Freezing conv base / classifier head change (the bug above)
- The actual `train_val(...)` training call (10 epochs planned, Adam lr=1e-4,
  `CosineAnnealingLR`)
- Learning curve plot, and the custom-image inference cell

So: **the model has not been trained yet.** Everything up through data loading/visualization
is done; the training loop itself is next.

## Portfolio repo setup (already done)

- Confirmed MIT license of the source course material (see above) → safe to publish.
- Created `~/coding/dl-portfolio/` with:
  - `README.md` (project list + license attribution)
  - `LICENSE` (verbatim MIT text, copyright preserved: Johannes Kolbe 2023)
  - `.gitignore` (`data/`, `*.pt`, `venv/`, `__pycache__/`, etc.)
  - `vgg19-transfer-learning-cifar10/` — copy of the working notebook + its own `README.md`
- `git init`, `remote add origin https://github.com/leeyunhome/dl-portfolio.git`, `branch -M main`
- User set `git config user.name "leeyunhome"` and `user.email "leeyunhome@users.noreply.github.com"`
  (noreply email chosen deliberately to avoid publicly exposing a real email in commit history)
- First commit ("Add VGG19 transfer learning CIFAR-10 exercise") pushed successfully via
  PAT auth (`credential.helper=store` was already set globally on this machine, so the PAT
  only needs to be entered once).
- Enabled GitHub Pages: Settings → Pages → Source "Deploy from a branch" → `main` / `(root)`.
- First version of the page was plain (unstyled `README.md` render) — added:
  - `_config.yml` with `theme: jekyll-theme-cayman` for a nicer default GitHub Pages theme
  - `jupyter nbconvert --to html transfer_learning_vgg19.ipynb` (using the borrowed
    yolov8_training venv) to produce `vgg19-transfer-learning-cifar10/transfer_learning_vgg19.html`,
    a standalone rendered view of the notebook's current outputs, linked from both READMEs

**Pending as of this note**: the theme + HTML export changes were written to disk on the
server but **not yet committed/pushed** — still need:
```bash
cd ~/coding/dl-portfolio
git add .
git commit -m "Add GitHub Pages theme and rendered notebook HTML"
git push
```

## Next steps for whoever continues this

1. Run the pending `git add`/`commit`/`push` above if not already done.
2. Decide on the classifier-head bug (see #3 above) before training — fix it or knowingly
   keep it for tutorial-fidelity, but either way training hasn't started yet.
3. Continue notebook execution from the `learning_rate` cell onward: helper functions →
   load VGG19 → freeze conv base → run `train_val` (10 epochs) → `plot_out` learning curves.
4. Drop a real local image next to the notebook and fix `img_path="cat.jpg"` before running
   the inference cell.
5. Once training/inference actually produce results, re-run
   `jupyter nbconvert --to html transfer_learning_vgg19.ipynb --output transfer_learning_vgg19.html`
   in `~/coding/dl-portfolio/vgg19-transfer-learning-cifar10/`, copy the updated
   `.ipynb`/`.html` over from the working copy in `~/coding/learning/vgg19_transfer/`, and
   commit/push again so the public GitHub Pages site reflects the finished results.
