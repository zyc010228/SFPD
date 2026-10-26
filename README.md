# SFPD

Official implementation of **Stabilizing Few-Shot Object Detection with Language-Conditioned Probabilistic Prototypes** (NeurIPS 2026).

## Overview

SFPD studies stable few-shot object detection with two core mechanisms:

- **Language-Conditioned Feature Query (LC-FQ)** uses precomputed CLIP text embeddings to provide semantic initialization for novel classes.
- **Uncertainty-Aware Prototype Distillation (UA-PD)** estimates heteroscedastic uncertainty and reduces the influence of unreliable support evidence.

The CLIP text encoder is used offline; the online detection path does not require text encoding.

## Repository status

This is an initial source release. The model components are provided first. Final experiment configurations, environment details, and pretrained checkpoints will be added after verification.

## Structure

```text
fpd/           Model and data-pipeline components
tools/         Training and evaluation entry points
```

## Dependencies

The implementation builds on:

- [MMFewShot](https://github.com/open-mmlab/mmfewshot)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [FPD](https://github.com/wangchen1801/FPD)
- [VFA](https://github.com/csuhan/VFA)
- [CLIP](https://github.com/openai/CLIP)

Please use versions compatible with the original FPD/MMFewShot environment. Exact environment and configuration files are being verified and will be documented in a subsequent update.

## Data

PASCAL VOC and MS-COCO are not redistributed in this repository. Download them from their official sources and follow the MMFewShot dataset preparation instructions.

## Citation

Citation information will be updated when the official proceedings entry is available.

## Acknowledgments

This codebase is derived from the OpenMMLab few-shot detection stack and the public FPD implementation. Portions retain their original copyright notices. We thank the authors of MMFewShot, MMDetection, FPD, VFA, and CLIP.

## License

No license is currently granted for this repository. Third-party components remain subject to their respective upstream terms.
