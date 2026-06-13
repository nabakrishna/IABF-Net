# ILLUMINATION-AWARE BEV FUSION NETWORK (IABF-NET) WITH TEMPORAL KINEMATIC REGULARIZATION

Modern Advanced Driver Assistance Systems (ADAS) and Autonomous Driving (AD) perception frameworks
heavily rely on deep learning models to execute 3D object detection, semantic segmentation, and depth
estimation. While these architectures demonstrate high structural accuracy under pristine, daytime
illumination conditions, their operational reliability degrades severely in adverse environments.

The primary breakdown occurs during low-light nocturnal driving, sudden weather transitions (such as
torrential rain or dense fog), and high-contrast exposure shifts (e.g., exiting tunnels). Visual noise, extreme
sensor attenuation, and photon scarcity introduce devastating artifacts into the camera stream. This disrupts
downstream spatial feature extractions, leading to catastrophic misalignments in 3D bounding box predictions
and depth registration.


| Component | Mechanism | Core Function |
| :--- | :--- | :--- |
| **1. Front-End Encoder** | Illumination-Guided Attention Blocks (IGAB) | Dynamically balances spatial features to correct regional exposure, lens glare, and low-lux photon deficiencies. |
| **2. Gating Layer** | Entropy-driven Signal-to-Noise Scoring Matrices | Calculates a dynamic camera confidence coefficient ($\alpha$) to track signal integrity and data quality. |
| **3. View Transformer** | Lift-Splat-Shoot / Discrete Depth Projections | Projects multi-view perspective features onto a unified orthographic 2D-to-3D Birds-Eye View (BEV) map. |
| **4. Fusion Engine** | Adaptive $\alpha$-weighted cross-modal merging | Creates a robust unified BEV map by balancing sensor influence and isolating degraded data paths. |
| **5. Downstream Heads** | Anchor-free centerpoint tracking arrays | Regresses 3D bounding box coordinates, object orientation, and velocities from the unified BEV map. |