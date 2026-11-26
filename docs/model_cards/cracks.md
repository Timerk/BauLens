# Model card: BauLens crack classifier

| | |
| --- | --- |
| Task | Binary classification of a concrete surface **image patch**: crack / no crack |
| Architecture | torchvision ResNet18, ImageNet weights `ResNet18_Weights.IMAGENET1K_V1`, new one-logit head |
| Input | RGB image, resized (not cropped) to 128 × 128 px, ImageNet mean/std normalisation |
| Output | P(crack) and a label from a threshold chosen on the validation split (stored in `metadata.json`) |
| Formats | PyTorch state dict `model.pt`, ONNX `model.onnx` (opset 18, dynamic batch axis) |
| Code | `baulens.cracks` (`baulens cracks split / train / evaluate / export / benchmark`) |
| Results | [docs/results/cracks.md](../results/cracks.md) |
| Licence | Code MIT. Trained on Özgenel (2019), CC BY 4.0; weights are not distributed |

## Intended use

Triage support for screening **patch images** of concrete surfaces for visible surface cracks:
sorting photos so that a qualified inspector looks at likely cracks first, or highlighting
regions (Grad-CAM) for review. The output is a screening signal for a person, not a finding.

**Out of scope:** structural assessment or any safety decision; crack width, length, depth or
orientation measurement; crack type or cause (shrinkage, settlement, corrosion); whole-structure
photos, drone imagery or scenes with objects, edges, joints, graffiti, vegetation or shadows that
the training patches do not contain; other materials (masonry, asphalt, steel).

## Training data

[Özgenel (2019), *Concrete Crack Images for Classification*](https://data.mendeley.com/datasets/5y9wdsg2zt/2),
CC BY 4.0: 40,000 RGB patches of 227 × 227 px (20,000 per class), cropped from 458 photos of
campus buildings at METU (Ankara). All patches have the same size and similar lighting and camera.

- Split: 70 / 15 / 15 % train / validation / test, stratified, with **duplicate groups kept in
  one split**. 3,117 files are byte-identical copies and a further set of re-encoded near-copies
  was found; 12,920 images (32 %) belong to 6,194 duplicate groups. A naive random split would
  have put 3,328 duplicate pairs across split boundaries. See the results page for the method.
- The CPU default trains on a stratified 40 % subsample of the train split (11,200 images) to fit a
  laptop CPU budget; validation and test use all 6,000 images each.
- Patches from the same source photo cannot be grouped (photo identifiers are not published),
  so test images still share photos, surfaces and lighting with training images.

## Evaluation

Test split, 6,000 images, threshold 0.954 chosen on validation (lower edge of the validation F1
plateau 0.954–0.984). 95 % CIs in brackets come from a bootstrap over detected duplicate groups.
They are conditional on this patch benchmark; source-photo-level uncertainty is unmeasured.

| Precision | Recall | F1 | ROC-AUC | ECE |
| --- | --- | --- | --- | --- |
| 0.9993 [0.9983, 1.0] | 0.9987 [0.9973, 0.9997] | 0.9990 [0.9982, 0.9997] | 0.999997 (rounded) | 0.0012 |

On the tested synthetic perturbations (2,000 test patches), contrast × 0.3 lowers recall to
0.841. Mean confidence only falls from 0.9995 to 0.9811, and ECE rises from 0.00058 to 0.03034.
Over-exposure (× 1.5) lowers recall to 0.969. Precision stays at 1.000 except at JPEG quality 10
(0.999). See [docs/results/cracks.md](../results/cracks.md) for the
calibration, the full corruption probe, the error gallery and the ONNX/CPU benchmark. Test performance is near the ceiling, as
is usual for this dataset. **It measures in-distribution performance on one homogeneous,
single-source lab-like dataset and says little about field performance.**

## Limitations

- **Patch level only.** One probability per image; no localisation beyond a coarse Grad-CAM map
  (4 × 4 activations at 128 px), no segmentation, no crack width or length.
- **Single dataset, single site.** No evidence yet for other concretes, surface finishes,
  lighting, cameras, distances or weathering. A cross-dataset test (e.g. SDNET2018, CC BY 4.0) is
  the obvious next step and is not done.
- **Shift sensitivity.** The synthetic corruption probe on the results page shows which capture
  conditions degrade the model; real shifts can be larger and unlike any of them.
- **Hard negatives are under-represented.** Joints, formwork marks, edges, stains and shadows that
  look like cracks are rare in the negatives, so false positives in the field are likely higher.
- **Residual leakage.** Duplicate grouping removes copies, not distinct patches of the same photo;
  the test score is therefore optimistic.
- **Threshold.** Chosen to maximise F1 on validation. A screening deployment would rather fix a
  minimum recall and accept more false positives; re-derive it on representative data.

## Ethical and safety notes

Missed cracks (false negatives) can delay maintenance, and false alarms cost inspection time and
can erode trust. The model must not replace a visual inspection by a qualified person or the
inspection rules for structures (in Germany, e.g. DIN 1076 for bridges). The dataset shows no
people; inference images might, so photos should be handled under the applicable data-protection
rules. The training data come from one university campus, and nothing is known about how well
the model works for structures elsewhere.
