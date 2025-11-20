<div align="center">

# Multi-Source Adaptive Constrained Energy Minimization for Hyperspectral Target Detection on Lightweight Platforms

**[Robin Gerster](https://github.com/RobinGerster7)<sup>1</sup>, [Peter Stütz](https://www.researchgate.net/profile/Peter-Stuetz)<sup>1</sup>**  
<sup>1</sup> University of the Bundeswehr Munich


</div>

<p align="center">
  <a href="#-update">Update</a> |
  <a href="#-overview">Overview</a> |
  <a href="#-datasets">Datasets</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-citation">Citation</a> |
  <a href="#-contact">Contact</a>
</p>

---


# 🌞 Overview
**MACEM** (Multi-Source Adaptive Constrained Energy Minimization) is a fast and deterministic method for **test-time spectrum refinement** in hyperspectral target detection. It builds on our previous work, [**TASR**](assets/readmes/tasr.md), which introduced spectrum adaptation. MACEM performs a **single, efficient update** guided by one or more labeled source images—improving speed, stability, and generalization across scenes. MACEM is the first method to explicitly support **multi-source refinement**, but it also works with a **single source**, making it both robust and flexible. It is particularly suited for detectors deployed in **unfamiliar environments** and **lightweight platforms**, such as onboard UAVs, where reliability and runtime efficiency are critical.

---


# 🛠 Testing MACEM

To test **MACEM**, make sure you configure your experiment with `MACEM()` as the generator. Below is a minimal working example:

```python
if __name__ == "__main__":
    image_size = (100, 100)
    config = DemoConfig(
        source_folder="datasets/",
        test_folder="datasets/SanDiego",
        pre_processing_transforms=[Resize(image_size, "bilinear"), MinMaxNormalize()],
        gt_transforms=[Resize(image_size, "nearest")],
        post_processing_transforms=[MinMaxNormalize()],
        detector=CEM(),
        target_spectrum_generator=MACEM(),  # Use MACEM here
        num_runs=1,
        fpr_range=(0, 1),
        rgb_indices=[79/223, 94/223, 112/223],
    )

    config.display()
    experiment = RotatingLeaveOneOutExperiment(config)
    experiment.run()
```
---

# ⭐ Citation

If TASR proves useful to your work, please consider starring this repository ⭐ and citing:

```
@article{MACEM,
  title={Multi-Source Adaptive Constrained Energy Minimization for Hyperspectral Target Detection on Lightweight Platforms},
  author={Robin Gerster, Peter Stütz},
  journal={arXiv preprint arXiv:2406.11519},
  year={2025}
}
```

---


