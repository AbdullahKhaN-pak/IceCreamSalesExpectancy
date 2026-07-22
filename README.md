# Polynomial Regression — Streamlit App

This small Streamlit app loads an ice-cream sales CSV, fits a polynomial regression model, shows a smooth fit, reports R², and lets you predict sales for a given temperature.

How to run:

1. (Optional) Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run streamlit_app.py
```

Notes:
- The app accepts a CSV upload. If no upload is provided it will try to load `3-1-icecream_sales.csv` or `icecream.csv` from the working directory. A sample `3-1-icecream_sales.csv` is included in the repository.
- The CSV should have temperature in the first column and sales in the second column.
- You can choose the plotting backend in the app: `Matplotlib` (static) or `Plotly` (interactive).

If you want me to also convert the existing `polynomial.ipynb` into a notebook cell that demonstrates the same Plotly visualization, say so and I will add it.
