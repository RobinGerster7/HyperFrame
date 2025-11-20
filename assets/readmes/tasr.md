<div align="center">

# Towards Robust Hyperspectral Target Detection via Test-Time Spectrum Adaptation

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

**TASR** is the first method for **test-time domain adaptation** based on **target spectrum refinement**. It enhances hyperspectral target detection by using a **discrete genetic algorithm** to dynamically adapt to new environmental conditions. TASR is frugal, requiring only one labeled image containing the target material to be detected, making it highly efficient.



---


# 🛠 Testing TASR

To test **TASR**, use the `TASR()` target spectrum generator. TASR performs **test-time adaptation** by optimizing a refined target spectrum using a **genetic algorithm** based on pixels from the test image and a single labeled source image.

Below is a minimal working example:

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
        target_spectrum_generator=TASR(),  # Use TASR here
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
@article{TASR,
  title={Towards Robust Hyperspectral Target Detection via Test-Time Spectrum Adaptation},
  author={Robin Gerster, Peter Stütz},
  journal={arXiv preprint arXiv:2406.11519},
  year={2025}
}
```

---

