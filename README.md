# Neural SLAM: Real-Time 3D Reconstruction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-black.svg)](https://flask.palletsprojects.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![Three.js](https://img.shields.io/badge/Three.js-r128-black.svg)](https://threejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Real-time 3D reconstruction with Neural Radiance Fields and SLAM tracking.**

## 🌐 Live Demo

**🚀 [Try it now on Hugging Face Spaces](https://huggingface.co/spaces/aarrvvee9/neural-slam)**

## ✨ Features

- 🎯 **Real-time NeRF** - Instant neural radiance field reconstruction
- 📍 **SLAM Tracking** - Camera pose estimation and trajectory mapping  
- 🎨 **3D Visualization** - Interactive Three.js viewer
- 🌐 **Apple-Style UI** - Stunning landing page with smooth animations
- ⚡ **Zero Installation** - Runs entirely in browser

## 🎥 How It Works

1. **Upload video frames** - Process frame by frame
2. **3D reconstruction** - Extract point clouds from each frame
3. **Camera tracking** - Estimate camera pose in real-time
4. **Visualize** - See the 3D scene build up

## 🚀 Run Locally

```bash
git clone https://github.com/vraul92/neural-slam.git
cd neural-slam
pip install -r requirements.txt
python app.py
```

**Opens at:** http://localhost:7860

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Flask + Gradio |
| **3D Rendering** | Three.js |
| **Animations** | GSAP |
| **NeRF** | Simplified for demo |
| **SLAM** | Feature-based tracking |

## 📁 Project Structure

```
neural-slam/
├── app.py              # Flask + Gradio application
├── requirements.txt    # Dependencies
├── templates/
│   └── index.html      # Apple-style landing page
├── static/
│   ├── css/style.css   # Modern UI styles
│   └── js/main.js      # Three.js + animations
├── src/                # Core SLAM algorithms
└── README.md
```

## 🤝 Author

**Rahul Vuppalapati** - Senior Data Scientist
- Previously: Apple, Walmart, IBM
- GitHub: https://github.com/vraul92
- LinkedIn: https://linkedin.com/in/vrc7

## 📄 License

MIT License - Feel free to use for research and commercial projects.

---

Built with ❤️ using Flask, Three.js, and PyTorch
