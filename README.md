# 🚀 Codexia

Codexia is a modular AI-powered code analysis and improvement platform designed to help developers understand, optimize, and enhance their codebases efficiently. It combines modern **GenAI**, **Machine Learning**, and **static analysis techniques** to provide intelligent suggestions, reports, and insights.

---

## 🧠 Overview

Codexia is built with a **React (Vite)** frontend and a **FastAPI** backend. The system is structured in a modular way to support scalability, maintainability, and future AI integrations.

It focuses on:

* Code analysis
* AI-driven suggestions
* Report generation
* Dataset-based learning

---

## 🏗️ Architecture

```
Codexia/
│
├── frontend/ (React + Vite)
├── backend/ (FastAPI)
│
├── analyzer/        # Code analysis logic
├── ml/              # Machine learning models & pipelines
├── suggestions/     # AI-based recommendations
├── reports/         # Report generation system
├── dataset/         # Training & evaluation datasets
│
└── codedog/         # Core execution module
```

---

## ⚙️ Tech Stack

### Frontend

* React (Vite)
* Tailwind CSS (optional)

### Backend

* FastAPI
* Python

### AI/ML

* GenAI (LLMs)
* Custom ML models

### Tools & Utilities

* GitHub API integration (optional)
* Environment-based configuration (.env)

---

## 🔑 Key Features

### 1. Code Analyzer

* Parses and analyzes source code
* Detects issues, inefficiencies, and patterns

### 2. AI Suggestions

* Provides intelligent improvements
* Suggests refactoring and optimizations

### 3. Report Generation

* Generates structured reports
* Helps in understanding code quality

### 4. Dataset Integration

* Uses datasets for training and evaluation
* Supports continuous improvement of models

### 5. Modular Design

* Easy to extend and scale
* Each module works independently

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* Node.js 18+

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Codexia
```

### 2. Setup Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Setup Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key
```

### 4. Run Backend

```bash
uvicorn main:app --reload
```

### 5. Setup Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## 🔄 Workflow

1. User uploads or inputs code
2. Analyzer processes the code
3. ML/GenAI modules generate insights
4. Suggestions module provides improvements
5. Reports module generates structured output

---

## 📊 Future Enhancements

* RAG (Retrieval-Augmented Generation) integration
* Multi-language code support
* Real-time code analysis
* GitHub repo auto-analysis
* CI/CD integration

---

## 🤝 Contribution

Contributions are welcome!

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

## 📜 License

This project is licensed under the MIT License.

---

## 💡 Author

**Tanvi Pandey**
Final Year Project | AI + Full Stack Development

---

 
