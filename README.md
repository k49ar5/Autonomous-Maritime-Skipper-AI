Autonomous Maritime Decision-Support System (Skipper AI)
Project Overview
This project implements an autonomous maritime decision-support system designed to assist in vessel navigation and obstacle avoidance. By integrating Computer Vision (YOLOv11) with Generative AI (Llama 3.2) and Vector Databases (Qdrant), the system transitions from simple object detection to intelligent, rule-based reasoning. The architecture is built using LangGraph to ensure a stateful, controllable, and modular agentic workflow.

Technical Architecture
The system operates as a directed acyclic graph (DAG) consisting of three primary nodes:

Perception Node: Leverages a fine-tuned YOLOv11 model to perform real-time object detection on maritime imagery. It identifies critical navigation marks such as lateral buoys, cardinal marks, and vessels.

Knowledge Retrieval Node (RAG): Utilizes a Qdrant vector store to perform similarity searches based on detected objects. It retrieves relevant maritime regulations (e.g., IALA standards) stored as high-dimensional embeddings.

Navigation Reasoning Node: A local LLM (Llama 3.2 via Ollama) synthesizes the visual data and retrieved regulatory text to generate a formalized navigation command and safety assessment.

Key Features
Edge-Optimized: Designed for local execution on CPU-only environments (e.g., Raspberry Pi 5 / NVIDIA Jetson) using quantized models and OpenVINO optimization.

Offline-First: Operates without external API dependencies, ensuring reliability in remote maritime or aerial environments.

Agentic Orchestration: Uses LangGraph to manage state and ensure data consistency between vision and reasoning layers.

Tech Stack
Artificial Intelligence: YOLOv11, LangChain, LangGraph, Ollama (Llama 3.2).

Vector Database: Qdrant (Dockerized).

Data Processing: OpenCV, HuggingFace Transformers (Embeddings).

Language: Python 3.10+.

Installation and Usage
Prerequisites
Docker (for Qdrant)

Ollama (for Llama 3.2 inference)

Python 3.10+

Setup
Clone the repository and install dependencies:

Bash
pip install -r requirements.txt
Initialize the Qdrant database with regulatory data:

Bash
python src/database_ingestion.py
Run the main agent:

Bash
python src/main_agent.py
Development and Training
The model training process is documented in notebooks/yolov11_training_maritime.ipynb. The dataset was curated via Roboflow and trained on NVIDIA T4 GPUs to achieve high mAP scores for maritime-specific classes.
