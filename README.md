# Dog Breed Classifier API

Production-ready REST API for dog breed identification using deep learning.

[![Live Demo](https://img.shields.io/badge/demo-live-success)](https://dog-vision-api.onrender.com) [![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org) [![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)

## 🚀 Live Demo

- **Web Interface:** https://dog-vision-api.onrender.com
- **API Docs:** https://dog-vision-api.onrender.com/docs



## 🛠️ Tech Stack

**Backend:** FastAPI, TensorFlow 2.19, ResNet50  
**Frontend:** HTML/CSS/JavaScript  
**DevOps:** Docker, Render


## 🔧 Quick Start

### Local Setup
Clone repository
git clone https://github.com/Anushka-200617/dog-vision-API.git
cd dog-vision-API

Create virtual environment
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Run application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


