# Learning Notes — VGG19 Transfer Learning

Concepts covered while working through this project (Hugging Face Computer Vision
Course exercise, ported to a local GPU server), kept as a study log alongside the code.

## Transfer learning

- **Feature extraction** (used here): freeze a pretrained model's convolutional base
  (`requires_grad = False` on all `features` parameters) and train only a new
  classifier head on top. The base's learned representations (edges, textures, shapes
  from ImageNet) are reused as-is; only the final decision layer adapts to the new task.
- Contrast with **fine-tuning**: unfreezing some/all of the base and training it too,
  usually with a smaller learning rate. Not used in this exercise.
- Swapping a classifier head correctly means **replacing the layer object**
  (`nn.Linear(in, out)`), not just changing its `.out_features` attribute — the latter
  only edits metadata, not the actual weight tensor shape.

## Data pipeline

- `Dataset` (e.g. `torchvision.datasets.CIFAR10`) + `DataLoader` split: the dataset
  defines how to fetch and transform one sample; the loader batches, shuffles, and
  parallelizes fetching across `num_workers` worker processes.
- `pin_memory=True` avoids an extra pageable→pinned memory copy before the CPU→GPU
  DMA transfer, which matters because **CPU↔GPU data transfer (over PCIe) is a classic
  bottleneck in heterogeneous (CPU+GPU) computing** — GPU compute throughput far
  outpaces PCIe bandwidth, so anything that speeds up the transfer path helps keep the
  GPU fed.
- `shuffle=True` for training (avoids the model learning from a fixed sample order),
  `shuffle=False` for validation (order doesn't affect evaluation, and keeps runs
  comparable).
- Batch size trade-off: a larger batch improves **wall-clock** throughput per step
  (better GPU parallelism) but doesn't automatically mean faster convergence in terms
  of epochs/steps — gradient noise from smaller batches can act as an implicit
  regularizer, and a bigger batch usually needs the learning rate scaled up too, which
  wasn't done automatically just by raising `batch_size`.
- `root='./data'` in `download=True` datasets resolves relative to the **kernel's
  working directory**, not the notebook file's location — caused real confusion when
  checking whether a download had landed in the expected folder.

## Tensors and shapes

- Image batch tensor: `[batch, channels, height, width]` (PyTorch convention, channels
  before spatial dims) — `[128, 3, 224, 224]` here.
- Label batch tensor: `[batch]` — just one scalar class index per sample, no spatial
  structure needed, hence 1-D vs. the image tensor's 4-D.
- `transpose((1, 2, 0))` reorders axes from PyTorch's `(C, H, W)` to matplotlib's
  expected `(H, W, C)` — the numbers are axis *indices* being permuted, not pixel
  values.

## Normalization and statistics

- Normalizing with per-channel `mean`/`std` centers pixel values and scales them to a
  consistent range, stabilizing and speeding up training.
- Standard deviation, from scratch: deviation from the mean → squared (to remove sign)
  → averaged (= variance) → square-rooted (back to original units) = std. A small std
  means values cluster tightly around the mean; a large std means they're spread out.

## Learning rate scheduling

- `CosineAnnealingLR(T_max=N)` decreases the LR along a cosine curve down to `eta_min`
  at step `N` — but **without restarts, it doesn't stay low**: past `T_max` it rises
  back up symmetrically, mirroring the descent. Setting `T_max` shorter than the total
  number of training epochs (5 vs. 10 here) caused the LR to climb back up in the
  second half of training, visibly degrading loss/accuracy — confirmed by hand-computing
  the cosine formula against the logged `current_lr` values.
- Distinguishing "loss got worse because of overfitting" vs. "loss got worse because of
  a scheduler artifact" mattered here — the actual cause was the latter, not the model
  memorizing training data.

## Model selection in a training loop

- `train_val`'s checkpointing selects the **best model by lowest validation loss**, not
  by highest validation accuracy — these don't always agree exactly (here, the
  highest-accuracy epoch and the lowest-loss epoch were two epochs apart).

## Debugging habits that paid off

- Not trusting tutorial code/comments at face value surfaced three real bugs: the
  classifier-head resize that silently didn't do anything, a plotting bug that split a
  title and image across two disconnected figures, and the LR scheduler/epoch-count
  mismatch. All three were confirmed by checking actual values (tensor shapes, logged
  learning rates), not just by reading the code.

## Licensing and publishing

- Before publishing adapted course material, checked the source repository's actual
  `LICENSE` file (MIT here) rather than assuming — MIT permits reuse/redistribution as
  long as the original copyright notice is kept, which is why this repo carries the
  original license text.
- Keeping this portfolio repo fully separate from the company work repo (`en683_lpr`)
  avoided any risk of exposing company-internal infrastructure or product details.

## Tooling notes

- `jupyter nbconvert --to html` doesn't render Plotly (`application/vnd.plotly.v1+json`)
  outputs without extra setup (e.g. `kaleido`) — worked around by re-plotting the same
  numbers with matplotlib as a static PNG for reliable rendering on GitHub Pages.
- Running the "same" notebook from two different file copies against the same remote
  Jupyter kernel can silently diverge into two separate execution histories — worth
  double-checking which file is actually wired to the live kernel before trusting
  "it's already been run" assumptions.
