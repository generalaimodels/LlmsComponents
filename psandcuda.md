

### Basic Usage

1. **Display General Information**:
   ```bash
   nvidia-smi
   ```

2. **Display GPU Utilization**:
   ```bash
   nvidia-smi --query-gpu=utilization.gpu --format=csv
   ```

3. **Display Memory Usage**:
   ```bash
   nvidia-smi --query-gpu=memory.used,memory.free --format=csv
   ```

4. **Display Driver Version**:
   ```bash
   nvidia-smi --query-gpu=driver_version --format=csv
   ```

5. **Display GPU Temperature**:
   ```bash
   nvidia-smi --query-gpu=temperature.gpu --format=csv
   ```

### Advanced Monitoring

6. **Display GPU Performance and Power Usage**:
   ```bash
   nvidia-smi --query-gpu=power.draw,power.limit --format=csv
   ```

7. **Display GPU Compute Mode**:
   ```bash
   nvidia-smi --query-gpu=compute_mode --format=csv
   ```

8. **Display GPU Memory Information (detailed)**:
   ```bash
   nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv
   ```

9. **Monitor Processes Using GPUs**:
   ```bash
   nvidia-smi pmon -c 1
   ```

10. **Display Real-Time GPU Usage**:
    ```bash
    watch -n 1 nvidia-smi
    ```

### Management

11. **Set GPU Power Limit**:
    ```bash
    nvidia-smi -i 0 -pl 100
    ```
    (Set power limit to 100 watts for GPU 0)

12. **Enable/Disable Persistence Mode**:
    ```bash
    nvidia-smi -pm 1   # Enable persistence mode
    nvidia-smi -pm 0   # Disable persistence mode
    ```

13. **Change GPU Clock Speeds**:
    ```bash
    nvidia-smi -ac 3505,875
    ```
    (Set application clocks to 3505 MHz memory and 875 MHz GPU clock)

14. **Reset GPU**:
    ```bash
    nvidia-smi --gpu-reset -i 0
    ```

15. **Display GPU Health Status**:
    ```bash
    nvidia-smi --query-gpu=health --format=csv
    ```

### Query and Logging

16. **Query All GPUs for Various Metrics**:
    ```bash
    nvidia-smi --query-gpu=all --format=csv
    ```

17. **Export GPU Information to a File**:
    ```bash
    nvidia-smi --query-gpu=all --format=csv > gpu_info.csv
    ```

18. **Display GPU Clock Information**:
    ```bash
    nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory --format=csv
    ```

19. **Show GPU Processes with Detailed Information**:
    ```bash
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    ```

20. **Monitor GPU Usage Over Time (historical data)**:
    ```bash
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv --loop=1
    ```

### Additional Tips:

- **Help and Full Documentation**:
  ```bash
  nvidia-smi --help
  ```

- **Check CUDA Version**:
  ```bash
  nvcc --version
  ```
-----

| **Command**                                                   | **Description**                                                   |
|---------------------------------------------------------------|-------------------------------------------------------------------|
| `nvidia-smi`                                                  | Display general GPU information                                   |
| `nvidia-smi --query-gpu=utilization.gpu --format=csv`        | Display GPU utilization                                          |
| `nvidia-smi --query-gpu=memory.used,memory.free --format=csv`| Display GPU memory usage                                         |
| `nvidia-smi --query-gpu=driver_version --format=csv`         | Display GPU driver version                                       |
| `nvidia-smi --query-gpu=temperature.gpu --format=csv`        | Display GPU temperature                                          |
| `nvidia-smi --query-gpu=power.draw,power.limit --format=csv` | Display GPU power usage and limit                                |
| `nvidia-smi --query-gpu=compute_mode --format=csv`           | Display GPU compute mode                                         |
| `nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv` | Display detailed GPU memory information |
| `nvidia-smi pmon -c 1`                                       | Monitor processes using GPUs                                      |
| `watch -n 1 nvidia-smi`                                     | Display real-time GPU usage (updates every second)               |
| `nvidia-smi -i 0 -pl 100`                                   | Set power limit to 100 watts for GPU 0                           |
| `nvidia-smi -pm 1`                                          | Enable persistence mode                                          |
| `nvidia-smi -pm 0`                                          | Disable persistence mode                                         |
| `nvidia-smi -ac 3505,875`                                   | Set application clocks to 3505 MHz memory and 875 MHz GPU clock |
| `nvidia-smi --gpu-reset -i 0`                               | Reset GPU 0                                                      |
| `nvidia-smi --query-gpu=health --format=csv`                 | Display GPU health status                                        |
| `nvidia-smi --query-gpu=all --format=csv`                    | Query all GPUs for various metrics                               |
| `nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory --format=csv` | Display current GPU clock speeds               |
| `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` | Show detailed GPU processes information              |
| `nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv --loop=1` | Monitor GPU usage over time (loop)                |
| `nvidia-smi --query-gpu=memory.free,memory.total,memory.used --format=csv,noheader` | Display memory usage without headers           |
| `nvidia-smi -q`                                              | Display detailed GPU information (query)                        |
| `nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv`  | Display GPU utilization for GPU 0                                |
| `nvidia-smi -i 1 --query-gpu=temperature.gpu --format=csv`   | Display GPU temperature for GPU 1                                |
| `nvidia-smi --list-gpus`                                    | List all available GPUs                                          |
| `nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=10` | Display GPU utilization every 10 seconds               |
| `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` | Show memory usage of GPU processes              |
| `nvidia-smi --query-gpu=power.draw --format=csv`             | Display GPU power draw                                           |
| `nvidia-smi --query-gpu=power.limit --format=csv`            | Display GPU power limit                                          |
| `nvidia-smi --query-gpu=memory.used --format=csv`            | Display GPU memory used                                          |
| `nvidia-smi --query-gpu=memory.free --format=csv`            | Display GPU memory free                                          |
| `nvidia-smi --query-gpu=driver_version --format=csv,noheader`| Display GPU driver version without headers                       |
| `nvidia-smi --query-gpu=name --format=csv`                   | Display GPU name                                                 |
| `nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader` | Display GPU temperature without headers          |
| `nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv`   | Display temperature of GPU 0                                     |
| `nvidia-smi -i 1 --query-gpu=power.draw --format=csv`        | Display power draw of GPU 1                                      |
| `nvidia-smi -i 0 --query-gpu=power.limit --format=csv`       | Display power limit of GPU 0                                     |
| `nvidia-smi --query-gpu=utilization.memory --format=csv`     | Display GPU memory utilization                                   |
| `nvidia-smi --query-compute-apps=pid,process_name --format=csv` | Show processes using GPUs with names            |
| `nvidia-smi --query-gpu=clocks.max.graphics,clocks.max.memory --format=csv` | Display maximum GPU clock speeds         |
| `nvidia-smi --query-gpu=uuid --format=csv`                   | Display GPU UUID                                                  |

----


| **Command**                                                        | **Description**                                           |
|--------------------------------------------------------------------|-----------------------------------------------------------|
| `nohup python <script> > output.log 2>&1 &`                       | Run a Python script in the background, redirect output.   |
| `ps aux | grep python`                                            | List processes and filter for those containing 'python'. |
| `tail -f output.log`                                              | Continuously monitor the `output.log` file.               |
| `ps -p <PID> -o`                                                  | Display process information for a specific PID.           |
| `nvidia-smi pmon -i 0`                                            | Monitor processes using GPU 0.                            |
| `nvidia-smi -q -d PIDS`                                           | Query GPU processes (PIDS) details.                       |
| `ps -p <PID> -o user,cmd`                                         | Display user and command for a specific PID.              |
| `top`                                                             | Display real-time system summary and process information. |
| `htop`                                                            | Interactive process viewer with a user-friendly interface.|
| `ps aux`                                                          | List all processes with detailed information.            |
| `ps -u username`                                                  | List processes for a specific user.                       |
| `ps -ef`                                                          | List all processes with full details.                     |
| `ps -p <PID>`                                                     | Display process details for a specific PID.               |
| `ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem`                      | List processes sorted by memory usage.                    |
| `ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu`                      | List processes sorted by CPU usage.                       |
| `ps -ef --forest`                                                 | Display process tree.                                    |
| `ps -p <PID> -o pid,ppid,cmd,%cpu,%mem`                           | Display detailed process information for a specific PID.  |
| `ps -C <process_name>`                                            | List processes by process name.                           |
| `ps -o ppid= -p <PID>`                                            | Display the parent process ID (PPID) for a specific PID.  |
| `ps -eo pid,ppid,cmd,%cpu,%mem --sort=pid`                        | List processes sorted by PID.                             |
| `ps --sort=-vsz`                                                  | Sort processes by virtual memory size (descending).       |
| `ps --sort=pcpu`                                                  | Sort processes by CPU usage (ascending).                  |
| `ps --sort=pmem`                                                  | Sort processes by memory usage (ascending).               |
| `ps --pid <PID1>,<PID2>`                                          | Display details for multiple PIDs.                        |
| `ps --user <username>`                                            | List processes for a specific user.                       |
| `ps --pid <PID> --format pid,uid,cmd`                             | Display process ID, user ID, and command.                 |
| `ps --sort=-pid`                                                  | Sort processes by PID in descending order.               |
| `ps --sort=comm`                                                  | Sort processes by command name.                           |
| `ps --sort=stime`                                                 | Sort processes by start time.                             |
| `ps -eLf`                                                         | List all threads of all processes.                        |
| `ps -e -o pid,uid,cmd`                                            | Display process ID, user ID, and command.                 |
| `ps -L -p <PID>`                                                  | List threads for a specific PID.                          |
| `ps -o etime=`                                                    | Display elapsed time since process start.                 |
| `ps -o pid,etime,cmd --sort=etime`                                | List processes sorted by elapsed time.                    |
| `ps -o pid,comm,etime --sort=etime`                               | List processes sorted by elapsed time, showing command name. |
| `ps --sort=-etime`                                                | Sort processes by elapsed time (descending).              |
| `ps -eo pid,tty,cmd | grep <tty>`                                 | List processes associated with a specific terminal.       |
| `ps -o pid,user,comm,etime --sort=-etime`                         | List processes sorted by elapsed time, showing user and command. |

