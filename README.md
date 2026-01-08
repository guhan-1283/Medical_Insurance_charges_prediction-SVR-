# Support Vector Regression — Medical Insurance Cost Prediction

Project to build and demonstrate a Support Vector Regression (SVR) model that predicts medical insurance costs.

**Repository contents**
- `app.py` — application entry (interactive app / demo).
- `Model_building.ipynb` — notebook used to explore data, train the SVR model, and evaluate results.
- `medical_insurance.csv` — dataset used for training and demonstration.
- `requirements.txt` — Python dependencies for the project.

## Project summary
This project trains and demonstrates an SVR model to predict medical insurance charges from patient data. The notebook contains data preprocessing, feature engineering, model training, evaluation, and optional model serialization.

## Setup (Windows)
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1  # PowerShell
# or .venv\Scripts\activate for cmd
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Running
- To run the Streamlit app (if `app.py` uses Streamlit):

```powershell
streamlit run app.py
```

- To run `app.py` as a normal script (if applicable):

```powershell
python app.py
```

- To reproduce experiments and retrain the model, open `Model_building.ipynb` in Jupyter or VS Code and run the cells.

## Dataset
`medical_insurance.csv` (included) contains the records used to train the model. It typically includes demographic and health-related features and a target variable for insurance charges.

## Notes
- The `requirements.txt` provides a minimal set of common packages used for data science and Streamlit apps. Adjust versions as needed for your environment.
- If you serialize models in the notebook (e.g., with `joblib.dump`), the README can be updated with filenames and loading instructions.

## Next steps
- (Optional) I can install dependencies, run the Streamlit app, or update the README with more project-specific details (examples, config, saved model file names).

---
If you'd like, tell me which action you'd like next (install deps, run app, or update README content).
