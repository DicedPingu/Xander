# Motorola G24 Root Plan

## 1. Overview
This document outlines the steps required to gain root access on a Motorola G24 (Android 14) device. The process involves unlocking the bootloader, flashing a custom recovery (TWRP), and installing a root solution (Magisk).

## 2. Prerequisites
- A Motorola G24 device with at least 50% battery.
- USB debugging enabled (for ADB/Fastboot).
- A computer with Android SDK Platform Tools installed.
- The correct device model code (e.g., `g24` or `g24power`).

## 3. Steps

### 3.1. Unlock Bootloader
1. Power off the device.
2. Hold **Volume Down** + **Power** to enter Fastboot mode.
3. Connect to the computer via USB.
4. Verify connection with:
   ```bash
   fastboot devices
   ```
5. Unlock the bootloader:
   ```bash
   fastboot flashing unlock
   ```
   - Confirm on the device by pressing **Volume Up**.

### 3.2. Flash TWRP Recovery
1. Download the TWRP image for Motorola G24 (e.g., from [Twrp.me](https://twrp.me/motorola/motorola-g24.html)).
2. Rename the file to `twrp.img`.
3. Flash TWRP:
   ```bash
   fastboot flash recovery twrp.img
   ```
4. Reboot into TWRP:
   ```bash
   fastboot reboot
   ```
   - Immediately press **Volume Up** + **Power** to boot into TWRP.

### 3.3. Install Magisk
1. In TWRP, go to **Install** and select the Magisk APK (downloaded from [GitHub](https://github.com/Magisk-Open-Source/Magisk/releases)).
2. Swipe to confirm installation.
3. Reboot the device.

### 3.4. Verify Root
- After reboot, check for root access using an app like **Root Checker** from the Play Store.

## 4. Notes
- The bootloader unlock may wipe data. Back up important files before proceeding.
- Some Motorola devices require a specific `edl` or `fastboot` variant; adjust commands if needed.
- If stuck, try flashing the stock firmware first, then repeat these steps.

## 5. References
- [Twrp.me – Motorola G24](https://twrp.me/motorola/motorola-g24.html)
- [Magisk GitHub](https://github.com/Magisk-Open-Source/Magisk)
- [XDA Developers – Motorola G24 Root](https://forum.xda-developers.com/motorola-g24)