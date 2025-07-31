# Federated MNIST Training with LIFL -Based on LIFL: A LIGHTWEIGHT, EVENT-DRIVEN SERVERLESS PLATFORM FOR FEDERATED LEARNING

This project implements **federated learning** using the **MNIST dataset** and simulates training across **10 clients** with hierarchical aggregation. It’s inspired by the paper *Asynchronous Hierarchical Federated Learning (LIFL)*.

---

## 📦 Features

✅ Actual model training on MNIST  
✅ 10 simulated clients  
✅ Logs of per-round loss/accuracy  
✅ Results visualization  
✅ Easily extensible (middle aggregator, non-IID data)

---

## 🧠 Architecture

- **Client**: Trains on local MNIST data
- **Aggregator**: Averages models from all clients
- (Optional) **Hierarchical**: Add mid-level aggregators for real-world simulation

---

## 🚀 How to Run

```bash
# Clone repo
git clone https://github.com/djshanker/lifl-mnist-federated.git
cd lifl-mnist-federated

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training
python main.py
