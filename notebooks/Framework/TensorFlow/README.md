# TensorFlow Learning Notebooks

## 🧠 Compute Backend Specification
The notebooks in this directory are designed to be executed via a **Remote Jupyter Kernel** provided by a dedicated Docker container.

### Architecture Decoupling:
- **Storage Path**: `notebooks/Framework/TensorFlow/` (Host Filesystem)
- **Compute Provider**: Docker Container `Starboy_dx` (located in `projects/TensorFlowProjects/`)
- **Connection Protocol**: HTTP/WebSocket via Port `8888`

### Why this setup?
1. **Environment Isolation**: Prevents "Dependency Hell" on the host Windows machine.
2. **Hardware Passthrough**: Seamlessly utilizes the **RTX 2080 Super GPU** through WSL2 passthrough without local CUDA installation.
3. **Consistency**: Ensures that experiments are reproducible across different machines by simply spinning up the predefined Docker image.

### Usage:
Set the VS Code Jupyter Kernel to `http://localhost:8888` to access the high-performance TensorFlow environment.