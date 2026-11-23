# 🐕 Boston Dynamics Spot Robot Repository

Welcome to the **Spot Robot** repository! This repository provides guides, configuration steps, API setups, and example scripts for working with **Spot**—the agile, quadruped mobile robot by Boston Dynamics—utilizing the official [Boston Dynamics Spot SDK](https://github.com/boston-dynamics/spot-sdk).

---

## 📑 Table of Contents

1. [Overview & Capabilities](#-overview--capabilities)
2. [Hardware Specifications](#-hardware-specifications)
3. [Prerequisites & Environment Setup](#-prerequisites--environment-setup)
4. [Installation Guide](#-installation-guide)
5. [Connecting to Spot](#-connecting-to-spot)
6. [Quickstart: Controlling Spot via Python](#-quickstart-controlling-spot-via-python)
7. [Repository Structure](#-repository-structure)
8. [Safety & Operation Guidelines](#-safety--operation-guidelines)
9. [Official References & Links](#-official-references--links)

---

## 🌟 Overview & Capabilities

Spot is an advanced 4-legged mobile robot designed to navigate challenging terrains, perform dynamic locomotion, and integrate custom sensors or payloads. 

### Key Capabilities:
* **Terrain Adaptability:** Walks, trots, climbs stairs, and navigates obstacles autonomously.
* **360° Spatial Sensing:** Built-in stereo cameras continuously detect obstacles and map local surroundings.
* **Payload Integration:** Supports external hardware like LiDAR, thermal imaging cameras, edge compute units (Spot CORE), and the Spot Arm.
* **gRPC-based API:** Controlled programmatically via modern remote procedure calls using the official SDK.

---

## ⚙️ Hardware Specifications

| Feature | Specification |
| :--- | :--- |
| **Dimensions (Walking)** | 1100 mm (L) × 500 mm (W) × 610 mm (H) |
| **Weight** | ~33.8 kg (74.5 lbs) with battery |
| **Payload Capacity** | Up to 14 kg (30.8 lbs) |
| **Max Speed** | 1.6 m/s (approx. 3.6 mph) |
| **Operating Temp** | -20°C to 45°C (-4°F to 113°F) |
| **Protection Rating** | IP54 (Dust & Water Resistant) |
| **Battery Life** | ~90 minutes nominal |

---

## 🛠️ Prerequisites & Environment Setup

Before programming Spot, ensure your development environment satisfies the following requirements:

* **Operating System:** Ubuntu Linux (20.04 / 22.04 LTS recommended) or macOS / Windows 10+
* **Python Version:** Python 3.8+
* **Network Connection:** Wi-Fi connection to Spot's local access point or shared enterprise network.

---

## 📦 Installation Guide

1. **Clone the Repository & Spot SDK:**
   ```bash
   git clone https://github.com/boston-dynamics/spot-sdk.git
   cd spot-sdk
   ```

2. **Create & Activate a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   ```

---

## 📡 Connecting to Spot

Spot communicates via its onboard Wi-Fi network or Ethernet connection.

1. Turn on Spot and wait for the power LEDs to turn solid green.
2. Connect your development machine to Spot's Wi-Fi network (e.g., `Spot-BDXXXXXX`).
3. Verify connection by pinging Spot's default IP address:
   ```bash
   ping 192.168.80.3
   ```

---

## 🚀 Quickstart: Controlling Spot via Python

Here is a simple example script showing how to authenticate, acquire an operation lease, power on the motors, stand the robot up, and safely power down using the Python SDK.

```python
import time
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand

def main():
    # 1. Initialize the SDK and create a robot object
    sdk = bosdyn.client.create_standard_sdk('SpotQuickstartClient')
    robot = sdk.create_robot('192.168.80.3')

    # 2. Authenticate using user credentials
    robot.authenticate('user', 'password')
    robot.time_sync.wait_for_sync()

    # 3. Create clients for lease and command control
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    # 4. Acquire control lease and keep it alive
    with bosdyn.client.lease.LeaseKeepAlive(lease_client):
        # Power on motors safely
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot failed to power on!"

        print("Standing Spot up...")
        blocking_stand(command_client, timeout_sec=10)
        time.sleep(3)

        print("Powering Spot off...")
        robot.power_off(cut_off_immediately=False, timeout_sec=20)

if __name__ == '__main__':
    main()
```

---

## 📁 Repository Structure

```text
├── docs/                 # Documentation and hardware diagrams
├── scripts/              # Setup scripts and network utility tools
├── src/
│   ├── examples/         # Sample scripts (Hello Spot, WASD Teleop, Camera Streams)
│   └── custom_nodes/     # Custom robot behaviors and API extensions
├── requirements.txt      # Python dependencies
└── README.md             # Repository documentation
```

---

## ⚠️ Safety & Operation Guidelines

* **Keep Distance:** Stay at least 2 meters (6.5 feet) away from Spot when motors are enabled.
* **E-Stop Ready:** Always have an E-Stop (software or physical payload button) within reach during experimental control.
* **Payload Balance:** Ensure all mounted payloads are firmly secured and center-of-mass aligned to prevent motor overloading.

---

## 🔗 Official References & Links

* [Official Boston Dynamics Spot SDK GitHub](https://github.com/boston-dynamics/spot-sdk)
* [Boston Dynamics Developer Portal](https://dev.bostondynamics.com/)
* [Spot Python SDK Documentation](https://dev.bostondynamics.com/docs/python/understanding_spot_python)
