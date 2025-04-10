# Analysis of Consumptions

This repository contains the **Analysis of Consumptions** module, a Python project for analyzing consumption data.

## Repository and Project Structure

- **Git Repository:** [tfm](git@github.com:marcosboj/tfm.git)
- **Project Directory:** `tfm/analisis_consumos`
- **Dependencies File:** `requirements-dev.txt`

## Installation

To install the project in a virtual environment, follow these steps:

### 1. Clone the Repository

```bash
git clone git@github.com:marcosboj/tfm.git
cd tfm/analisis_consumos
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements-dev.txt
```

### 4. Install the Module in Editable Mode

```bash
python -m pip install -e . --config-settings editable_mode=strict
```

### Running the Clustering Script

To run the clustering script, use the following command:

```bash
python scripts/energy_consumption_groups.py
```
