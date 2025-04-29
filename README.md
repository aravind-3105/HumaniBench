# HumaniBench: Evaluations for Human-Centric Multimodal Understanding

## Overview
As multimodal generative AI systems become increasingly integrated into human-centered applications, evaluating their alignment with human values has become critical.

HumaniBench is the first comprehensive human-centric benchmark for evaluating **Multimodal Large Language Models (MLLMs)** across fairness, ethics, perception, multilingual equity, empathy, and robustness.

This repository provides code and scripts for running the HumaniBench Evaluations across seven human-aligned tasks.

- 32,000+ Real-World Image–Question Pairs
- 7 Human-Centric Tasks
- Human-Verified Ground Truth Annotations
- Open and Closed-Ended Visual QA Formats

- Paper (Preprint)
- HumaniBench: A Human-Centric Benchmark for Multimodal Large Language Models Evaluation (NeurIPS 2025 under review)



# Evaluation Tasks Overview

| Evaluation | Focus | Folder |
|:---|:---|:---|
| **Eval 1: Scene Understanding & Social Bias** | Visual reasoning + bias/toxicity analysis in images with social attributes (gender, age, occupation, etc.) | `evaluations/eval1_Scene_Understanding` |
| **Eval 2: Context Understanding** | Visual reasoning in socially and culturally rich contexts | `evaluations/eval2_Context_Understanding` |
| **Eval 3: Visual Perception (Multiple Choice QA)** | Attribute recognition via multiple-choice VQA tasks | `evaluations/eval3_Visual_Perception` |
| **Eval 4: Multilingual Visual QA** | VQA performance across 10+ languages | `evaluations/eval4_Multilingual` |
| **Eval 5: Visual Grounding** | Localizing social attributes within images (bounding box detection) | `evaluations/eval5_Visual_Grounding` |
| **Eval 6: Emotion and Empathy (WIP)** | Empathetic captioning and emotional understanding evaluation | `evaluations/eval6_Emotion` |
| **Eval 7: Robustness Evaluation** | Model resilience under image perturbations (blur, noise, compression) | `evaluations/eval7_Robustness` |

> Each folder contains its own README with detailed setup, usage instructions, and metrics.


# Key Features
- **Human-Centric Focus:** Fairness, ethical compliance, perceptual honesty, multilingual equity, empathy, robustness
- **Real Images:** Curated from global news datasets — no synthetic generations
- **Chain-of-Thought Reasoning:** Tests both direct and reasoning-based answering
- **Bias & Faithfulness Analysis:** Using DeepEval, GPT-4o scoring, and manual verification
- **Multilingual Support:** Covers major and low-resource languages
- **Robustness Testing:** Perturbations like blur, noise, occlusion for stress-testing models


# Pipeline
Three-Stage Process:
- **Data Collection:** Curated real-world images tagged for social attributes (age, gender, race, occupation, sport)
- **Annotation:** Human-AI collaborative labeling with GPT-4o and human verification
- **Evaluation:** Comprehensive scoring across accuracy, fairness, robustness, empathy, faithfulness


# Key Findings

- Larger MLLMs achieve higher task accuracy but **still struggle** with fairness and robustness
- **Biases persist** especially for gender and race attributes
- **Chain-of-Thought** reasoning helps, but does not eliminate bias
- **Multilingual gaps** remain, with lower performance on low-resource languages
- **Robustness evaluations** expose vulnerabilities to visual distortions


# Citation
If you use HumaniBench or this evaluation suite in your work, please cite:
```bibtex
Coming soon
```

# Contact

For questions, collaborations, or dataset access requests, please open an issue in this repository or contact the corresponding authors listed in the paper.


---

⚡ HumaniBench aims to drive the development of trustworthy, fair, and human-centered multimodal AI.

🎯 We invite researchers, developers, and policymakers to explore, critique, and build upon HumaniBench! 🚀
