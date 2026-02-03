# Autonomous Maritime Decision-Support System (Skipper AI)

## Project Overview
This project implements an autonomous maritime decision-support system designed to assist in vessel navigation and obstacle avoidance. By integrating **YOLOv11** for computer vision, **Llama 3.2** for reasoning, and **Qdrant** as a vector database, the system transitions from basic object detection to intelligent, rule-based decision-making. The architecture is built using **LangGraph** to ensure a stateful, modular, and controllable agentic workflow.



## Technical Architecture
The system operates as a directed acyclic graph (DAG) consisting of three primary nodes:

1. **Perception Node**: Leverages a fine-tuned **YOLOv11** model to perform object detection. It identifies critical navigation marks such as lateral buoys, cardinal marks, and other vessels.
2. **Knowledge Node (RAG)**: Utilizes a **Qdrant** vector store to perform similarity searches based on detected objects. It retrieves relevant maritime regulations (IALA standards) stored as high-dimensional embeddings.
3. **Navigation Node**: A local LLM (**Llama 3.2 via Ollama**) synthesizes the visual data and retrieved regulatory text to generate formalized navigation commands and safety assessments.



## Key Features
* **Edge-Optimized**: Designed for local execution on CPU-only environments (e.g., Raspberry Pi 5 or NVIDIA Jetson) using quantized models.
* **Offline-First**: Operates without external API dependencies, ensuring reliability in remote maritime environments.
* **Agentic Orchestration**: Implements LangGraph to manage system state and ensure data consistency between vision and reasoning layers.
* **Local Embeddings**: Uses local HuggingFace embedding models to maintain a fully air-gapped RAG pipeline.

## Tech Stack
* **Vision**: YOLOv11 (Ultralytics)
* **Orchestration**: LangGraph, LangChain
* **Vector Database**: Qdrant (Dockerized)
* **Inference Engine**: Ollama (Llama 3.2)
* **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
* **Language**: Python 3.10+

## Project Structure
```text
AI_SKIPPER_PROJECT/
├── notebooks/                  # Training and experimentation
│   └── yolov11_training.ipynb  # Fine-tuning process
├── src/                        # Production source code
│   ├── database_ingestion.py   # RAG initialization
│   ├── main_agent.py           # LangGraph orchestration
│   └── logger_config.py        # Centralized logging
├── models/                     # Local model storage
│   ├── best.pt                 # YOLOv11 weights
│   └── embedding_model/        # Offline embedding weights
├── requirements.txt            # Dependency list
└── README.md                   # Documentation
