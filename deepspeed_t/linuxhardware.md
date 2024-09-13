

### 1. **CPU Information**
   - **Command**: `lscpu`
     - Provides detailed information about the CPU architecture, cores, threads, model, and speed.
   - **Command**: `cat /proc/cpuinfo`
     - Lists detailed information about each CPU core.
   - **Command**: `dmidecode -t processor`
     - Provides detailed processor information including CPU flags and specifications.

### 2. **Memory Information**
   - **Command**: `free -h`
     - Shows the amount of used and free memory in the system.
   - **Command**: `dmidecode -t memory`
     - Detailed memory information including RAM type, size, speed, and manufacturer.

### 3. **GPU Information**
   - **Command**: `lspci | grep -i vga`
     - Lists GPU information.
   - **Command**: `nvidia-smi` (for NVIDIA GPUs)
     - Provides detailed information on NVIDIA GPUs, including temperature, utilization, and driver version.
   - **Command**: `glxinfo | grep "OpenGL renderer"`
     - Shows GPU details including the OpenGL renderer.

### 4. **Storage Information**
   - **Command**: `lsblk`
     - Lists block devices and their mount points.
   - **Command**: `lsscsi`
     - Lists all SCSI devices including storage drives.
   - **Command**: `df -h`
     - Shows disk space usage.
   - **Command**: `smartctl -a /dev/sda` (requires `smartmontools`)
     - Detailed S.M.A.R.T. data for storage devices.

### 5. **Motherboard and BIOS Information**
   - **Command**: `dmidecode -t baseboard`
     - Provides motherboard details including manufacturer and model.
   - **Command**: `dmidecode -t bios`
     - Displays BIOS version and details.

### 6. **Network Information**
   - **Command**: `lspci | grep -i network`
     - Shows details about network adapters.
   - **Command**: `ethtool eth0`
     - Provides detailed information about a specific network interface.

### 7. **Overall System Information**
   - **Command**: `lshw -short`
     - Lists detailed hardware configuration.
   - **Command**: `inxi -Fxz`
     - Provides an easy-to-read summary of the system's hardware and software (install with `sudo apt install inxi`).

### 8. **Power and Battery Information**
   - **Command**: `upower -i /org/freedesktop/UPower/devices/battery_BAT0`
     - Displays battery status and details (for laptops).

### 9. **PCI Devices**
   - **Command**: `lspci`
     - Lists all PCI devices with detailed information.

### 10. **USB Devices**
   - **Command**: `lsusb`
     - Lists all USB devices connected to the system.

### Tips:
- **Combine with `sudo`**: Some commands like `dmidecode` may require root permissions (`sudo`) to show detailed information.
- **Use `-v` or `--verbose`**: For even more details, many commands support verbose modes (`-v`) to provide extended information.

