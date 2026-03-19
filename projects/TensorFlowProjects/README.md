# TensorFlow GPU Development Environment

This repository provides a standardized, containerized environment for deep learning projects based on **TensorFlow 2.15.0** and **NVIDIA CUDA**.

## 🚀 Environment Architecture
- **Engine**: Docker Desktop (WSL2 Backend)
- **Base Image**: `tensorflow/tensorflow:2.15.0-gpu-jupyter`
- **IDE**: VS Code with Dev Containers extension
- **Hardware Acceleration**: NVIDIA GeForce RTX 2080 Super (laptop) (Passthrough enabled)

## 🛠 Quick Start

### Option A: Immersive Development (Recommended)
1. Open this folder in VS Code.
2. Click **"Reopen in Container"** when prompted (or via Command Palette).
3. The environment will automatically handle Python dependencies and GPU drivers.

### Option B: Remote Kernel Mode (For Notebooks)
If you are working on notebooks outside this directory, use the container as a remote compute engine:
1. **Start the Server** (Inside Container Terminal):
   ```bash
   jupyter notebook --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --notebook-dir=/workspaces/TensorFlowProjects
2. Connect from Host:
    - Kernel Picker -> Existing Jupyter Server
    - URL: http://localhost:8888
3. ⚠️ 深度排查与已知现象豁免说明
    1. 右上角 Select Kernel 持续转圈 (内核发现与状态同步异常)
        - 现象：代码已执行完毕并成功输出结果，但 Jupyter Cell 状态 UI 始终持续转圈。
        - 成因参考：VS Code Jupyter 插件在容器/远程模式下特有的前端 UI 假死现象。VS Code 会在后台持续轮询 Jupyter Server 的 REST API（如 /api/kernels）以获取内核列表和状态。在 Docker 复杂的端口转发环境下，这个用于“状态查询”的 HTTP 响应可能发生延迟或丢失，导致右上角 UI 陷入无限等待。但与此同时，用于实际“代码执行”的底层通道（已经建立连接的 ZMQ socket）其实是独立且畅通的。
        - 处理：纯粹是 VS Code 插件前端 UI 的状态更新 Bug。只要代码单元格能正常运行并输出结果，完全不影响底层计算，直接无视右上角的转圈即可，或者可以尝试直接Develop: Reload Window。 （注：关闭 remote.autoForwardPorts 可以减少宿主机网络模块压力，缓解此现象，但无法彻底根除该插件 Bug）。
    2. TensorFlow 底层 C++ 注册报错 (已知冗余日志)
        - 现象：终端或单元格输出 Unable to register cuDNN/cuFFT/cuBLAS factory... when one has already been registered。
        - 成因参考：这是 TF 2.15+ 结合 XLA 编译时的一个已知底层逻辑瑕疵。CUDA 工厂类在初始化时被重复调用注册。首次注册已经成功并在内存中生效，二次注册被拒打印的 Error 日志不影响任何 GPU 加速功能，可安全忽略。
    3. GPU 显存动态分配 (性能优化预期行为)
        - 现象：日志提示 Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set.
        - 成因参考：默认情况下 TensorFlow 会在启动时直接映射并霸占 100% 的 GPU 显存。本环境通过 devcontainer.json 注入了 TF_FORCE_GPU_ALLOW_GROWTH=true 环境变量，强制启用显存按需动态分配。此举有效避免了在轻量测试或多任务时的 OOM (显存溢出) 崩溃，属于主动性能优化，为正常预期行为。