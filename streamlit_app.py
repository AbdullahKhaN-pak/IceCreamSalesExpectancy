import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="Polynomial Regression — Ice Cream Sales", layout="centered")
st.title("Polynomial Regression — Ice Cream Sales")
st.write("Upload a CSV or the app will try to load a file from the workspace.")

# File upload or local fallback
uploaded = st.file_uploader("Upload CSV file", type=["csv"])
data = None
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
else:
    # Try common filenames from the notebook/workspace
    candidates = ["icecream.csv", "3-1-icecream_sales.csv", "3-1-icecream sales.csv"]
    for c in candidates:
        if os.path.exists(c):
            try:
                data = pd.read_csv(c)
                st.info(f"Loaded local file: {c}")
                break
            except Exception:
                continue

if data is None:
    st.info("No dataset available. Please upload a CSV with two columns: Temperature and Sales.")
    st.stop()

if data.shape[1] < 2:
    st.error("Dataset must have at least two columns (X and y).")
    st.stop()

# Use first two columns as X and y
X = data.iloc[:, 0].to_numpy().reshape(-1, 1)
y = data.iloc[:, 1].to_numpy()

st.subheader("Data preview")
st.dataframe(data.head())

plot_backend = st.radio("Plot backend", ("Matplotlib", "Plotly"), index=0)

degree = st.slider("Polynomial degree", min_value=1, max_value=8, value=4)

poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

r2 = model.score(X_poly, y)
st.metric("R² Score", f"{r2:.4f}")

temp = st.number_input("Temperature to predict (°C)", value=20.0, format="%.2f")
pred = model.predict(poly.transform(np.array([[temp]])))[0]
st.write(f"**Predicted ice-cream sales at {temp:.2f}°C:** {pred:.2f}")

# Plot
X_curve = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_curve = model.predict(poly.transform(X_curve))

if plot_backend == "Matplotlib":
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, color="blue", label="Actual sales")
    ax.plot(X_curve, y_curve, color="red", linewidth=2, label=f"Degree {degree} polynomial fit")
    ax.set_title("Temperature vs Ice Cream Sales (Polynomial Regression)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Ice Cream Sales")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
else:
    # Create interactive Plotly figure
    df_points = pd.DataFrame({data.columns[0]: X.ravel(), data.columns[1]: y})
    df_line = pd.DataFrame({data.columns[0]: X_curve.ravel(), "predicted_sales": y_curve.ravel()})
    fig = px.scatter(df_points, x=data.columns[0], y=data.columns[1], labels={data.columns[0]: "Temperature (°C)", data.columns[1]: "Ice Cream Sales"}, title=f"Temperature vs Ice Cream Sales (Degree {degree})")
    fig.add_scatter(x=df_line[data.columns[0]], y=df_line['predicted_sales'], mode='lines', name='Polynomial fit')
    st.plotly_chart(fig, use_container_width=True)

# Download fitted predictions
out_df = pd.DataFrame({data.columns[0]: X_curve.ravel(), "predicted_sales": y_curve.ravel()})
csv = out_df.to_csv(index=False)
st.download_button("Download smooth predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

st.write("---")
st.write("Tip: If results look odd, check the CSV formatting and that the first column is temperature and second column is sales.")
