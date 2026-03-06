[README.md](https://github.com/user-attachments/files/25801969/README.md)
# GPU-Accelerated Near-Miss Detection from E-Scooter Video

**CS 5463 – HPML, Spring 2026**  
**Naveen John**  
**UTSA ScooterLab / CARE AI Lab**

---

## Project Overview

This project investigates three progressively optimized parallel approaches to accelerate an offline near-miss detection pipeline for e-scooter ride videos. The pipeline uses **YOLOv12** for object detection and **Apple Depth Pro** for monocular metric depth estimation on recorded footage from UTSA's ScooterLab fleet.

### Methods

| # | Method | Key Technique | Deadline |
|---|--------|--------------|----------|
| 1 | Baseline GPU Inference | PyTorch profiling, CUDA events | Mar 18 |
| 2 | TensorRT Optimization | FP16/INT8 quantization, layer fusion | Apr 1 |
| 3 | Multi-Stream Pipeline Parallelism | CUDA streams, batched frame processing | Apr 27 |

---

## Deliverables

- [Problem Statement (PDF)](problem_statement-onepage.pdf)
- [Bibliography & Algorithm Descriptions (DOCX)](bibliography.docx)

---

## Bibliography

### Object Detection (YOLO Family)

1. Y. Tian, Q. Ye, and D. Doermann, "YOLOv12: Attention-Centric Real-Time Object Detectors," *NeurIPS 2025*, arXiv:2502.12524. [[Paper]](https://arxiv.org/abs/2502.12524) [[Code]](https://github.com/sunsmarterjie/yolov12)
2. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," *CVPR*, 2016. [[Paper]](https://arxiv.org/abs/1506.02640)
3. A. Wang et al., "YOLOv10: Real-Time End-to-End Object Detection," arXiv:2405.14458, 2024. [[Paper]](https://arxiv.org/abs/2405.14458)
4. R. Khanam and M. Hussain, "YOLOv11: An Overview of the Key Architectural Enhancements," arXiv:2410.17725, 2024. [[Paper]](https://arxiv.org/abs/2410.17725)

### Monocular Depth Estimation

5. A. Bochkovskii et al., "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second," *ICLR 2025*, arXiv:2410.02073. [[Paper]](https://arxiv.org/abs/2410.02073) [[Code]](https://github.com/apple/ml-depth-pro)
6. L. Yang et al., "Depth Anything V2," *NeurIPS 2024*, arXiv:2406.09414. [[Paper]](https://arxiv.org/abs/2406.09414)
7. R. Ranftl, A. Bochkovskiy, and V. Koltun, "Vision Transformers for Dense Prediction," *ICCV*, 2021. [[Paper]](https://arxiv.org/abs/2103.13413)

### TensorRT and GPU Inference Optimization

8. NVIDIA Corporation, "TensorRT: Programmable Inference Accelerator," NVIDIA Developer Documentation, 2024. [[Docs]](https://developer.nvidia.com/tensorrt)
9. S. Shin and Y. Kim, "TensorRT-Based Framework and Optimization Methodology for Deep Learning Inference on Jetson Boards," *ACM Trans. Embedded Computing Systems*, vol. 21, no. 3, 2022. [[Paper]](https://dl.acm.org/doi/10.1145/3508391)
10. M. Alqahtani et al., "Accelerating Deep Learning Inference: A Comparative Analysis of Modern Acceleration Frameworks," *Electronics*, vol. 14, no. 15, 2977, 2025. [[Paper]](https://www.mdpi.com/2079-9292/14/15/2977)
11. M. Al-Qizwini et al., "Tensor-Based CUDA Optimization for ANN Inferencing Using Parallel Acceleration on Embedded GPU," *PMC/Sensors*, 2020. [[Paper]](https://pmc.ncbi.nlm.nih.gov/articles/PMC7256376/)
12. X. Zhang and B. Li, "Tennis Ball Detection Based on YOLOv5 with TensorRT," *Scientific Reports*, vol. 15, 21011, 2025. [[Paper]](https://www.nature.com/articles/s41598-025-06365-3)

### CUDA Streams and Pipeline Parallelism

13. M. Li et al., "Deep Learning and Machine Learning with GPGPU and CUDA: Unlocking the Power of Parallel Computing," arXiv:2410.05686, 2024. [[Paper]](https://arxiv.org/abs/2410.05686)
14. H. Zhou, S. Bateni, and C. Liu, "S3DNN: Supervised Streaming and Scheduling for GPU-Accelerated Real-Time DNN Workloads," *IEEE RTAS*, 2018. [[Paper]](https://ieeexplore.ieee.org/document/8394742)
15. S. Bateni et al., "Efficient CUDA Stream Management for Multi-DNN Real-Time Inference on Embedded GPUs," *Journal of Systems Architecture*, vol. 139, 2023. [[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S138376212300067X)
16. A. Patel, S. Rao, and V. Nagarajan, "Multi-Stream Scheduling of Inference Pipelines on Edge Devices – a DRL Approach," *ACM Trans. Design Automation of Electronic Systems*, 2024. [[Paper]](https://dl.acm.org/doi/10.1145/3677378)
17. PPipe: Efficient Video Analytics Serving on Heterogeneous GPU Clusters via Pool-Based Pipeline Parallelism, arXiv:2507.18748, 2025. [[Paper]](https://arxiv.org/abs/2507.18748)

### E-Scooter Safety and Near-Miss Detection

18. H. N. Kegalle et al., "Watch Out! E-scooter Coming Through!: Multimodal Sensing of Mixed Traffic Use and Conflicts Through Riders' Ego-centric Views," *Proc. ACM IMWUT*, vol. 9, no. 1, 2025. [[Paper]](https://arxiv.org/abs/2502.16755)
19. M. Tabatabaie, S. He, and X. Yang, "Naturalistic E-Scooter Maneuver Recognition with Federated Contrastive Rider Interaction Learning," *Proc. ACM IMWUT*, vol. 6, no. 4, 2022. [[Paper]](https://dl.acm.org/doi/10.1145/3570345)
20. S. He et al., "Beyond 'Taming Electric Scooters': Disentangling Understandings of Micromobility Naturalistic Riding," *Proc. ACM IMWUT*, vol. 8, no. 3, 2024. [[Paper]](https://dl.acm.org/doi/abs/10.1145/3678513)
21. J. Kim et al., "SecureRide: Detecting Safety-Threatening Behavior of E-Scooters Using Battery Information," *ACM Trans. Embedded Computing Systems*, vol. 24, no. 5s, 2025. [[Paper]](https://dl.acm.org/doi/10.1145/3758095)
22. H. Yang et al., "Safety of Micro-mobility: Analysis of E-Scooter Crashes by Mining News Reports," *Accident Analysis & Prevention*, vol. 143, 105608, 2020. [[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S0001457520317747)

### Quantization and Model Compression

23. B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," *CVPR*, 2018. [[Paper]](https://arxiv.org/abs/1712.05877)
24. P. Micikevicius et al., "Mixed Precision Training," *ICLR*, 2018. [[Paper]](https://arxiv.org/abs/1710.03740)

### GPU Profiling and Benchmarking

25. NVIDIA Corporation, "CUDA C++ Programming Guide," NVIDIA Developer Documentation, 2024. [[Docs]](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
26. NVIDIA Corporation, "Nsight Systems User Guide," NVIDIA Developer Documentation, 2024. [[Docs]](https://docs.nvidia.com/nsight-systems/)
