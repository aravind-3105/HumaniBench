# HumaniBench: A Human-Centric Benchmark for Large Multimodal Models Evaluation

## Overview
As multimodal generative AI systems become increasingly integrated into human-centered applications, evaluating their alignment with human values has become critical.

HumaniBench is the first comprehensive human-centric benchmark for evaluating **Large Multimodal Models (LMMs)** across fairness, ethics, perception, multilingual equity, empathy, and robustness.

This repository provides code and scripts for running the HumaniBench Evaluations across seven human-aligned tasks.

- 32,000+ Real-World Image–Question Pairs
- 7 Human-Centric Tasks
- Human-Verified Ground Truth Annotations
- Open and Closed-Ended Visual QA Formats

- Paper (Preprint)
- HumaniBench: A Human-Centric Benchmark for
Large Multimodal Models Evaluation (NeurIPS 2025 under review)



# Evaluation Tasks Overview

| Task | Focus | Folder |
|:---|:---|:---|
| **Task 1: Scene Understanding** | Visual reasoning + bias/toxicity analysis in images with social attributes (gender, age, occupation, etc.) | `code/task1_Scene_Understanding` |
| **Task 2: Instance Identity** | Visual reasoning in socially and culturally rich contexts | `code/task2_Instance_Identity` |
| **Task 3: Multiple Choice QA** | Attribute recognition via multiple-choice VQA tasks | `code/task3_Instance_Attribute` |
| **Task 4: Multilingual Visual QA** | VQA performance across 10+ languages | `code/task4_Multilingual` |
| **Task 5: Visual Grounding** | Localizing social attributes within images (bounding box detection) | `code/task5_Visual_Grounding` |
| **Task 6: Empathetic Captioning** | Empathetic captioning and emotional understanding evaluation | `code/task6_Emotion` |
| **Task 7: Image Resilience** | Model resilience under image perturbations (blur, noise, compression) | `code/task7_Robustness_and_Stability` |

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
