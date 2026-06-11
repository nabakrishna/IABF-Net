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