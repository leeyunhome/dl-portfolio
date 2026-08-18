# Session Notes — VGG19 Transfer Learning Portfolio

Continuation notes for picking this project up in a new session (different machine/agent).
Written 2026-08-18. **Status: training complete, results in, portfolio content finalized.**
Only remaining step is committing/pushing (see bottom).

## What this project is

`vgg19-transfer-learning-cifar10/` is a feature-extraction transfer learning exercise:
a pretrained VGG19 (ImageNet weights) with its convolutional base frozen, plus a new
classifier trained on top, to classify CIFAR-10 (10 classes).

**Source**: adapted from the [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course/unit0/welcome/TableOfContents)
(repo: `johko/computer-vision-course`, MIT License, Copyright (c) 2023 Johannes Kolbe).
Verified by fetching the repo's `LICENSE` file directly — MIT permits reuse/redistribution
with the original notice preserved, so publishing this as a portfolio piece is fine as
long as attribution + LICENSE are kept (see repo root `LICENSE`/`README.md`).

## Result

Best checkpoint (lowest val loss, epoch 5): **89.98% validation accuracy** after 10 epochs.
Peak val accuracy during the run: 90.12% (epoch 8). Held-out inference on a real CIFAR-10
cat image correctly predicted "cat". Full numbers and a rendered chart are in
`vgg19-transfer-learning-cifar10/README.md` and `loss_accuracy_history.png`.

## Where things live — IMPORTANT: there were two parallel copies of this notebook

This project ended up with **two independent notebook copies being run in parallel**
against the same remote kernel, which caused confusion mid-session:

1. `~/coding/learning/vgg19_transfer/transfer_learning_vgg19.ipynb` on the DL server —
   reached from Windows via a symlink at
   `C:\Users\devuser\my-github-repository\en683_lpr\vgg19_transfer` → SSHFS `Z:\coding\learning\vgg19_transfer`.
   This was the *first* copy worked on, got the classifier-bug fix applied early, but
   **training was never actually run in this copy** — it was abandoned once the second
   copy (below) turned out to be the one the user was actually executing cells in.
2. `C:\Users\devuser\my-github-repository\en683_lpr\references\lecture_ipynb\transfer_learning_vgg19_ipynb의_사본.ipynb`
   — a genuinely separate **local file on the Windows machine** (not a symlink), opened in
   VS Code with the same remote kernel (`YOLOv8 Training (venv)`, i.e. the same GPU server,
   see below). **This is the copy that was actually trained to completion** and is now the
   canonical source — it was copied into the portfolio repo (see below).

**Lesson for next time**: before doing repeated cell-execution work across a session, confirm
which single file is actually connected to the live kernel (check the kernel picker / execution
counts), rather than assuming a symlinked "working copy" is the one being run.

- **Portfolio repo**: `~/coding/dl-portfolio/` on the DL server (`deeplearning@10.10.237.222`,
  key-based SSH, see the `dl-server` skill), git remote `origin` =
  `https://github.com/leeyunhome/dl-portfolio.git`, branch `main`. GitHub Pages enabled
  (Settings → Pages → Deploy from branch, `main` / root), live at
  `https://leeyunhome.github.io/dl-portfolio/`. Kept fully separate from `en683_lpr`
  (that repo mixes in company production LPR work — server IPs, OCR training data, product
  binaries — so this portfolio repo intentionally excludes all of it).
- The notebook's kernel (`YOLOv8 Training (venv)`, `~/coding/learning/yolov8_training/venv`
  on the server) has `torch 2.9.1+cu128`, `torchvision`, CUDA, and (separately) `jupyter`/
  `nbconvert` — borrowed from a different project's venv but works fine for this purpose.
  No `kaleido` in this venv, so Plotly figures can't be exported to static PNG directly —
  worked around by re-plotting the same numbers with matplotlib instead (see below).

## Bugs found and fixed (already applied in the final notebook)

1. **Classifier head never actually resized.** `pre_vgg19.classifier[6].out_features = NUM_CLASSES`
   only overwrites the attribute — the underlying `nn.Linear` weight matrix stayed
   `(1000, 4096)`. Fixed: `pre_vgg19.classifier[6] = nn.Linear(4096, NUM_CLASSES)`.
2. **Title/image rendered as two disconnected figures** in `visualize_model_predictions`
   (title drawn on a separate `plt.subplot()` figure from the one `imshow()` creates
   internally). Fixed by passing the title into `imshow(img, title=...)` instead.
3. **`CosineAnnealingLR(T_max=5)` used for a 10-epoch run.** LR bottoms out at epoch 5 then
   rises back up symmetrically for epochs 6-10 (verified by hand-computing the cosine
   formula against the logged `current_lr` values — they matched exactly). This is why
   train/val loss both tick back up after epoch 6 — a scheduler mismatch, not overfitting.
   Not fixed (kept as a documented finding in the README) since training was already done
   by the time this was noticed; `T_max=10` would be the fix for a future run.
4. Kaggle-only inference image path replaced with a real held-out CIFAR-10 test image
   (label=cat) saved server-side and referenced by absolute path.
5. Comment/code mismatch noted (a comment said "first 8 images", code only showed 5) —
   informational only, not fixed (cosmetic, doesn't affect results).

## Portfolio repo contents (as of this note)

```
dl-portfolio/
├── .gitignore
├── LICENSE                              (verbatim MIT, Johannes Kolbe 2023)
├── README.md                            (project list + license attribution)
├── _config.yml                          (theme: jekyll-theme-cayman)
├── SESSION_NOTES.md                     (this file)
└── vgg19-transfer-learning-cifar10/
    ├── README.md                        (full write-up: result, bugs, setup)
    ├── transfer_learning_vgg19.ipynb    (final, trained, from the local 사본 copy)
    ├── transfer_learning_vgg19.html     (nbconvert export — note: Plotly output
    │                                     doesn't render in this export, see below)
    └── loss_accuracy_history.png        (matplotlib re-plot of the exact same
                                           loss/accuracy numbers, used because (a)
                                           kaleido isn't installed to export the
                                           Plotly figure directly, and (b) nbconvert's
                                           HTML exporter doesn't render the Plotly
                                           mimetype either — this PNG is the reliable
                                           fallback embedded directly in the README)
```

## Known rough edge: Plotly chart doesn't render in the nbconvert HTML export

`jupyter nbconvert --to html` prints:
```
UserWarning: Your element with mimetype(s) dict_keys(['application/vnd.plotly.v1+json'])
is not able to be represented.
```
So the interactive loss/accuracy chart is silently missing from
`transfer_learning_vgg19.html`. Worked around by regenerating the same chart with
matplotlib (exact same numbers, hardcoded from the executed notebook's output JSON) and
embedding that PNG directly in the project README instead of relying on the HTML export.
If someone wants the *interactive* Plotly chart to actually work on the Pages site, options:
try a different nbconvert template, or `pip install kaleido` and export it as its own
static image, or embed the raw Plotly JSON + plotly.js CDN script by hand.

## Remaining step

Everything above is written to disk on the server (`~/coding/dl-portfolio/`) but not all
committed/pushed yet. Check `git status` there — as of this note:
```
modified:   vgg19-transfer-learning-cifar10/README.md
modified:   vgg19-transfer-learning-cifar10/transfer_learning_vgg19.html
modified:   vgg19-transfer-learning-cifar10/transfer_learning_vgg19.ipynb
new file:   vgg19-transfer-learning-cifar10/loss_accuracy_history.png
```
Just needs:
```bash
cd ~/coding/dl-portfolio
git add .
git commit -m "Add final training results and bug-fix writeup"
git push
```
After that, the portfolio is done — the user has a job interview around 2026-08-25/08-28
this was being prepared for (see the `project-job-interview-and-portfolio-deadline` memory
if working from Claude's memory system).
