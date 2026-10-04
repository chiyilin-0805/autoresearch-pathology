# Open-source architecture review

Snapshot date: 2026-10-04. Star counts are approximate and will change.

This review was used to choose experiments 6-10. No external repository code
was copied into HistoGuard. The implementation uses PyTorch and torchvision
building blocks already compatible with the project.

| Project | Approx. stars | Relevant idea | Decision for HistoGuard |
|---|---:|---|---|
| [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) | 11.8k | FPN, U-Net and DeepLab-style pretrained segmentation architectures | Retain the existing multiscale FPN and treat localization as a first-class output. |
| [DINOv2](https://github.com/facebookresearch/dinov2) | 13.4k | Strong self-supervised patch representations | Promising future encoder study, but pathology/domain validation and weight licensing must be checked before redistribution. |
| [MONAI](https://github.com/Project-MONAI/MONAI) | 8.7k | Medical-imaging training, transforms and segmentation evaluation | Follow its explicit separation of data protocol, training and evaluation; do not add the large dependency for the current small experiment. |
| [anomalib](https://github.com/open-edge-platform/anomalib) | 6.2k | PatchCore/PaDiM-style image anomaly detection and pixel localization | Keep as a clean-only benchmark candidate. It is not the main model because cross-patient tissue can look locally normal and same-patient sham is also pasted. |
| [CLAM](https://github.com/mahmoodlab/CLAM) | 1.7k | Attention/MIL over local pathology instances with heatmaps | Motivated experiment 8's local top-k instance aggregation. It increased sham false positives and was discarded. |
| [PatchCore inspection](https://github.com/amazon-science/patchcore-inspection) | 1.4k | Memory-bank nearest-neighbour anomaly localization | Useful as a future non-neural-head baseline; same semantic limitation as anomalib applies. |
| [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath) | 637 | Microsoft/Providence tile encoder plus spatial slide aggregation | Motivated separating local evidence from global image evidence. Its whole-slide scale is unnecessary for current 512x512 images. |
| [Google Path Foundation](https://github.com/Google-Health/path-foundation) | 30 | Pathology-specific patch embeddings trained at scale | Relevant future frozen-encoder comparison, subject to model access, license and domain evaluation. |

## What transferred successfully

The most useful common principle was to make local evidence participate in the
image-level decision. Experiment 7 therefore added the strongest FPN heatmap
responses directly to the global classifier logit. This improved both AUROC and
localization. Experiment 10 retained that coupling and replaced ResNet18 with
the official ImageNet-pretrained ConvNeXt-Tiny backbone.

The direct CLAM/GigaPath-inspired alternative in experiment 8 used an
independent deep local-instance head. It improved box IoU over the old baseline
but raised same-patient-sham FPR to 0.25, so it was not selected.

## What was deliberately not adopted

- PatchCore/PaDiM were not treated as drop-in replacements. They learn a normal
  appearance distribution, whereas this task must distinguish cross-patient
  tissue from both clean images and visually similar same-patient paste shams.
- Whole-slide transformers were not added to 512x512 inputs. Their scale and
  dependency cost are not justified by the current data.
- No foundation-model result is claimed without an actual controlled run.
- No external project or weight is represented as suitable for clinical use.

## Next credible comparison

Before another architecture sweep, collect a new case-disjoint blind cohort.
Then compare the selected ConvNeXt model against (1) a clean-only PatchCore
baseline and (2) a frozen pathology foundation encoder with the same supervised
FPN head. Run at least three seeds and report case-bootstrap confidence
intervals. The already viewed historical test must not drive further choices.
