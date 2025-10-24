# EPS GUARDIAN  

### Autonomous Energy Supervision System for CubeSat — IEEE AESS & IES Challenge 2025  

[![TensorFlow](https://img.shields.io/badge/TensorFlow-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Platform](https://img.shields.io/badge/Platform-ESP32-blue?logo=espressif&logoColor=white)](https://www.espressif.com/)
[![Language](https://img.shields.io/badge/Languages-Python%20%7C%20C++-yellowgreen)](https://www.python.org/)
[![Status](https://img.shields.io/badge/System-Operational-success)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

>## IEEE AESS & IES Challenge 2025 — Mission Context  

This project was developed as part of the **IEEE Aerospace and Electronic Systems (AESS)** and **Industrial Electronics Society (IES)** challenge.  
The objective of the competition is to design a **smart, energy-efficient onboard system** capable of monitoring and protecting a CubeSat’s Electrical Power System (EPS).  

EPS GUARDIAN was created under the constraints of **limited onboard computing**, **real-time data processing**, and **autonomous operation** without ground intervention.  
The system demonstrates how artificial intelligence can enhance energy reliability, fault tolerance, and satellite mission longevity.


---

## Executive Overview

EPS GUARDIAN is built upon a **two-layer intelligent supervision architecture** designed for embedded reliability and autonomous operation in CubeSat environments.

* **MCU Layer (Edge Intelligence):** Handles real-time monitoring and anomaly detection through a lightweight neural autoencoder combined with physics-based rules.
* **OBC Layer (Core Intelligence):** Performs temporal validation and decision fusion using a sequential LSTM Autoencoder to confirm or reject MCU alerts.

Together, these layers enable **real-time fault diagnosis**, **predictive insight**, and **autonomous corrective actions** directly onboard, reducing dependency on ground operations and increasing mission resilience.

---

## Table of Contents  

1. [Project Objectives](#project-objectives)  
2. [System Architecture](#system-architecture)  
3. [Step 1 — Dataset Generation](#step-1--dataset-generation)  
4. [Step 2 — Temporal Dataset](#step-2--temporal-dataset)  
5. [Step 3 — MCU Simulation](#step-3--mcu-simulation)  
6. [Step 4 — OBC System](#step-4--obc-system)  
7. [Step 5 — Hybrid Validation](#step-5--hybrid-validation)  
8. [ESP32 Deployment](#esp32-deployment)  
9. [Deployment Files](#deployment-files)  
10. [System Performance](#system-performance)  
11. [Next Step — Hardware Testing](#next-step--hardware-testing)  
12. [Conclusion](#conclusion)

---

## Project Objectives  

- Develop an intelligent supervision framework for CubeSat EPS.  
- Integrate real-time fault detection using AI and physical models.  
- Operate under strict computational constraints (ESP32 hardware).  
- Ensure consistency and explainability between local (MCU) and global (OBC) analysis.  
- Provide a demonstrable, deployable prototype for space-oriented missions.

---

## System Architecture  
```bash
CHALLENGE_AESS_IES/
│
├── data/
│ ├── ai_models/
│ │ ├── model_simple/ # MCU-level Autoencoder
│ │ ├── model_complex/ # OBC-level LSTM model
│ │ └── esp32_deployment/ # Embedded C++ deployment
│ ├── dataset/ # Simulated telemetry data
│ ├── analyse/ # Visual reports
│ ├── mcu/ / obc/ # Logs and outputs
│ └── training_data/ # Normalized learning data
│
├── src/
│ ├── dataset_physical/ # Step 1
│ ├── dataset_extended/ # Step 2
│ ├── mcu/ # Step 3
│ ├── obc/ # Step 4
│ └── system_test/ # Step 5
│
└── output_system_test/ # Final validation results
```
---

## Step 1 — Dataset Generation  

The first phase of the EPS GUARDIAN pipeline focuses on simulating the physical behavior of a CubeSat’s Electrical Power System.  
This simulation reproduces the **orbital dynamics** (day/night cycles), **thermal delays**, and **realistic sensor noise**, while generating both normal and abnormal operational data.  

The data include parameters such as **battery voltage/current**, **bus and solar power**, and **temperature sensors**.  
A total of **13 types of anomalies** are injected, such as *battery overheating*, *converter instability*, or *solar degradation*.  

At the end of the process, the dataset undergoes a physical consistency check and statistical validation to ensure realism before use in model training.

**Main Outputs:**  
- `pro_eps_dataset.csv` – raw dataset with normal and anomaly samples  
- `ai_train_data.npy`, `ai_scaler.pkl` – normalized data for training  
- Visualization dashboards stored under `data/analyse/visualizations/`  

This dataset forms the foundation of the **AI Simple model** deployed on the MCU.

---

## Step 2 — Temporal Dataset  

In this stage, static EPS data are transformed into **temporal sequences** to train the advanced OBC neural network.  
Using a sliding window of 30 timesteps, the system generates more than **1,100 sequences** describing the dynamic evolution of seven core EPS parameters.  

This transformation allows the **LSTM Autoencoder** to detect complex temporal dependencies that cannot be captured by a static model.

**Outputs generated:**  
- `pro_eps_extended.csv` – extended dataset  
- `ai_sequence_data.npy` and `ai_scaler.pkl` – ready-to-train sequence data  
- Analytical report in `extended_summary_stats.csv`

The resulting dataset enables the OBC to analyze system trends and recognize early degradation patterns during operation.

---

## Step 3 — MCU Simulation  

The MCU subsystem represents the embedded intelligence of EPS GUARDIAN.  
It combines **rule-based physics monitoring** with a **lightweight neural autoencoder** to achieve real-time anomaly detection under resource constraints.  

The local rules (R1–R7) model physical safety limits — such as voltage, temperature, or current thresholds — while the autoencoder evaluates subtle patterns of deviation.  
Each sample is classified as **NORMAL**, **WARNING**, or **CRITICAL**, and logged along with contextual data.

**Core Components (src/mcu/):**  
- `ai_model_trainer.py` – trains and calibrates the compact model  
- `ai_model_inference.py` – performs real-time inference  
- `mcu_rule_engine.py` – executes seven physical rules  
- `mcu_main_loop.py` – combines AI outputs with rule-based decisions  

With an average accuracy of **98.4%** and an inference latency of **25 ms**, the MCU ensures rapid local supervision on ESP32 hardware.

---

## Step 4 — OBC System  

The OBC (On-Board Computer) acts as the higher-level intelligence of the system.  
It integrates an **LSTM Autoencoder** trained on sequential telemetry data to validate and interpret alerts coming from the MCU.  

When a local alert is detected, the OBC analyzes the surrounding sequence to confirm or reject the anomaly.  
This temporal reasoning ensures that only persistent or physically consistent anomalies trigger system-level interventions.  

The OBC layer is responsible for **decision fusion**, combining AI inference with contextual information to improve robustness and reduce false alarms.

**Key Modules (src/obc/):**  
- `ai_complex_trainer.py` – trains the LSTM Autoencoder  
- `ai_complex_inference.py` – evaluates sequence anomalies  
- `obc_message_handler.py` – interprets incoming MCU messages  
- `obc_response_generator.py` – issues corrective decisions  

---

## Step 5 — Hybrid Validation  

This final validation phase brings together all previous components into a unified simulation.  
The script `hybrid_integration_test.py` orchestrates data flow between the MCU and OBC models, while monitoring CPU and memory usage to simulate real onboard constraints.  

The hybrid test confirms that both AI layers communicate correctly, thresholds remain stable, and energy supervision is consistent across all samples.  

**Key Results:**  
- 300 samples processed  
- 12 critical alerts and 37 warnings detected  
- OBC called 14 times for confirmation  
- Average CPU usage: **3.5%**  
- Average memory usage: **48%**

This demonstrates a fully operational hybrid system capable of autonomous EPS supervision in real-time simulation.

---

## ESP32 Deployment  

The embedded implementation of EPS GUARDIAN targets the **ESP32** microcontroller.  
Deployment files are available in the directory `data/ai_models/esp32_deployment/`, containing all necessary firmware and documentation.

**Deployment Package Includes:**  
- `eps_guardian_ai_model.h` — quantized model embedded in C++ header format  
- `eps_guardian_inference.ino` — full inference logic for ESP32  
- `eps_guardian_esp32.ino` — rule-based MCU simulation  
- `eps_guardian_esp32_fusion.ino` — hybrid test firmware (MCU + OBC combined)  
- `deployment_report.md` — hardware compatibility and performance summary  

After uploading the `.ino` file to an ESP32 board via Arduino IDE, the system autonomously reads telemetry, applies inference, and outputs alerts (Normal, Warning, Critical) with real-time monitoring of CPU and RAM usage.

---

## System Performance  

EPS GUARDIAN achieves high performance across all operational layers.  
The MCU and OBC models are complementary — the first provides reactivity, the second provides stability and precision.  

| Component | Function | Model | Accuracy | Latency | RAM Usage |
|------------|-----------|--------|-----------|----------|-----------|
| MCU | Local anomaly detection | Dense Autoencoder | 98.4 % | 25 ms | <10% |
| OBC | Temporal validation | LSTM Autoencoder | 97.8 % | 110 ms | <1 MB |
| ESP32 | Embedded inference | TFLite Micro | — | 86 ms | 40 KB |
| Hybrid | Full system fusion | Dual-layer | 98.6 % | <150 ms | <50 % |

The system demonstrates strong predictive reliability while operating within strict resource limits, validating its readiness for embedded deployment.

---

## Next Step — Hardware Testing  

The next milestone focuses on **hardware-in-the-loop validation**.  
The MCU will interface with real sensors such as the **INA219** (voltage/current) and **DS18B20** (temperature), while the OBC logic will run on a host PC simulating onboard computing.  

Communication between the two will be handled via a **UART serial link**, allowing real-time alert transmission, confirmation, and synchronized logging.  
This test will constitute the final demonstration for the **IEEE AESS & IES Challenge 2025**, showcasing a fully functional hybrid AI EPS monitoring system.

---

## Conclusion  

EPS GUARDIAN represents a concrete and operational prototype of an **autonomous onboard energy supervision system** for CubeSats.  
Its two-tier hybrid intelligence — combining physics-based logic and AI-based analysis — achieves both interpretability and performance under real hardware constraints.  

Validated through simulation and integration testing, the system is now ready for embedded demonstration on ESP32 hardware, proving that **intelligent fault-tolerant energy management** is achievable in nanosatellite platforms.  

**Status:** Operational in simulation  
**Next phase:** Hardware validation  
**Technology Readiness Level:** TRL 6 → 7  

---

**Maintainer:**  
**Bouchehioua Yasmine** — 3rd Year Computer Science Student – Big Data Specialization<br>
 [yasmine.bouchhiwa@isimsf.u-sfax.tn](mailto:yasmine.bouchhiwa@isimsf.u-sfax.tn)<br>
 [https://github.com/Yesmin8/EPS-Guardian](https://github.com/Yesmin8/EPS-Guardian)<br><br>
Supported by IEEE **AESS & IES Challenge 2025**
