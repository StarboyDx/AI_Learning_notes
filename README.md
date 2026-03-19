# AI Learning Notes

This repository documents my learning journey and practical explorations in Data Analysis, Machine Learning, and AI Development.

The primary focus here is on **engineering implementations** rather than theoretical derivations. Algorithm concepts and mathematical principles are documented separately. The goal is to clearly bridge the gap between abstract principles and executable code, making it highly readable and highly practical.

## 📂 Repository Structure
- `notebooks/`: Jupyter notebooks covering data analysis and ML basics. This directory also contains documentation of key concepts and my personal problem-solving processes.
- `projects/`: Hands-on practice projects derived from structured courses and self-directed learning.

## 🛠 Environments

### 1. Local Host Environment (Base ML & PyTorch)
- **Anaconda**: 24.11.3
- **Python**: 3.10.19
- **Core Libraries**: `numpy` 2.2.5, `pandas` 2.3.3, `matplotlib` 3.10.8, `seaborn` 0.13.2
- **Deep Learning**: `pytorch` 2.10 (CUDA 12.6)

> **Compatibility Note**: 
> The packages above represent the primary local learning environment. Mixing `conda` and `pip` installations occasionally introduces compatibility hurdles. For instance, I deliberately retain this setup to practice debugging runtime issues, such as resolving PyTorch OMP conflicts using `os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'`. For more complex or production-level projects, I switch to strict, isolated container environments.

### 2. Containerized Environment (TensorFlow)
- **Engine**: Docker (WSL2 Backend) + VS Code Dev Containers
- **Image**: `tensorflow/tensorflow:2.15.0-gpu-jupyter` (Hardware Passthrough Enabled)

## 📌 File Status Tags
This repository is updated incrementally. Files are tagged with the following suffixes to track my mastery level:
- **_H**: Hurdles encountered. Difficult to complete at the time; requires future review and optimization.
- **_HS**: Hurdles solved. The core problems within have been basically resolved.
- **_HH**: Highly complex and challenging.

## 📝 Language Policy
To maximize information density and convey intuition simply, complex concepts and personal reflections are generally written in **Chinese**. However, to align with industry standards, **English** is heavily used for code annotations, variables, and technical nouns.

*Note: If a specific chapter in the `notebooks` folder is directly related to a project in the `projects` folder, cross-references will be detailed in the local `README.md` within that chapter's directory.*

## 🗺️ Learning Roadmap

**Prerequisites**: Data Structures & Algorithms, UE C++ Development, Basic Python Syntax.

1. **Data Science Basics**: Pandas, Numpy, Matplotlib
2. **Machine Learning**: Supervised/Unsupervised Learning Algorithms, Classic Neural Networks
3. **Deep Learning Frameworks**: PyTorch, TensorFlow & Keras
4. **Project-Oriented Applications**: 
   - CV (Computer Vision) & NLP (Natural Language Processing)
   - Popular AI Open-Source Projects
5. **LLM Development**: LangChain, Web Integration, Big Data Technologies