# ASPA: Attribute-Structured Prompt Adapter for Plant Disease GZSL

This repository provides the official PyTorch implementation of **ASPA**, an attribute-structured prompt adapter for generalized zero-shot plant disease recognition.

ASPA is built on top of CLIP and is designed to improve generalized zero-shot learning (GZSL) performance by constructing discriminative text-driven prototypes and adapting image features with a lightweight structure-preserving adapter.

## Overview

Generalized zero-shot plant disease recognition aims to classify both seen disease categories, which are available during training, and unseen disease categories, which are not used for supervised training.

The main idea of ASPA is to use CLIP as a vision-language backbone and refine the alignment between image features and text-based disease descriptions. The method includes:

- class-name anchor construction;
- prompt-based attribute representation;
- discriminative prompt selection;
- purified prototype construction;
- lightweight feature adaptation;
- calibrated GZSL inference.

## Method Pipeline

The implementation follows the pipeline below:

1. Load plant disease images from folder-based class splits.
2. Encode images using a frozen CLIP image encoder.
3. Encode class names and disease-related prompts using a frozen CLIP text encoder.
4. Build a prompt bank for all seen and unseen classes.
5. Select discriminative prompts for each class according to visual-textual similarity.
6. Construct purified class prototypes from the selected prompts.
7. Train a lightweight ASPA adapter using:
   - classification loss;
   - attribute alignment loss;
   - structure-preserving distillation loss.
8. Evaluate the model under the generalized zero-shot setting using seen accuracy, unseen accuracy, and harmonic mean.

## Repository Structure

```text
ASPA-plant-disease-gzsl/
  configs/
  model.py
  requirements.txt
  README.md
  LICENSE
  .gitignore
