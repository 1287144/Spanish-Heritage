import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Simulated historical population data (millions) for simplicity. Replace with more granular if needed.
years = np.arange(1955, 2025, 5)
BRAZIL = [62, 72, 95, 121, 149, 176, 195, 200, 208, 213, 218, 220, 223, 228]
MEXICO = [30, 35, 49, 67, 84, 100, 112, 115, 120, 126, 128, 129, 130, 131]
ARGENTINA = [20, 20.7, 23, 26, 29, 32, 35, 37, 39, 41, 43, 44, 45, 46]
us_latino = {
    "Brazilian": [0.1, 0.2, 0.3, 0.7, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.85, 1.9, 2, 2.1],
    "Mexican": [2, 3, 5, 8, 13, 17, 20, 24, 28, 32, 34, 36, 37, 38],
    "Argentinian": [0.05, 0.07, 0.1, 0.14, 0.19, 0.21, 0.23, 0.25, 0.27, 0.3, 0.31, 0.32, 0.33, 0.35]
}
country_names = ['Brazil', 'Mexico', 'Argentina']
data_dict = {
    'Brazil': BRAZIL,
    'Mexico': MEXICO,
    'Argentina': ARGENTINA,
}
category_dict = {
    'Population': (years, data_dict),
    # Add further categories with real data in the same structure as above.
}

def poly_fit_fn(x, *coeffs):
    return sum([coeffs[i] * x**i for i in range(len(coeffs))])

st.title("Latin America 70-Year Historical Data Regression Analysis")

category = st.selectbox("Select data category", list(category_dict.keys()))
years, country_data = category_dict[category]
countries = st.multiselect("Select countries to plot", country_names, default=country_names)
compare_us_groups = st.checkbox("Compare with Latin-origin groups in the US")
degree = st.slider("Select polynomial degree for regression", 3, 6, 3)
increment = st.slider("Graph in increments of how many years?", 1, 10, 5)

df = pd.DataFrame({'Year': years})
for c in countries:
    df[c] = country_data[c]

if compare_us_groups:
    us_groups = st.multiselect(
        "Select US Latino groups",
        list(us_latino.keys()),
        default=[]
    )
    for g in us_groups:
        df[f"{g} (US)"] = us_latino[g]

st.write("**Raw data table (editable):**")
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

st.write("**Inputs:** Year; **Outputs:** " + ", ".join(edited_df.columns[1:]))

extrapolate_years = st.slider("Years to extrapolate beyond last data point", 0, 30, 10)
extrapolate_option = st.radio("Show extrapolated curve?", ["Yes", "No"])
interp_extrap_choice = st.radio(
    "Want to interpolate or extrapolate a value?",
    ["No", "Interpolate/Extrapolate"]
)
if interp_extrap_choice == "Interpolate/Extrapolate":
    interp_year = st.number_input(
        "Input year (can be outside the current data range)", 
        min_value=int(df['Year'].min())-50, 
        max_value=int(df['Year'].max())+extrapolate_years
    )
    output_pred = True
else:
    output_pred = False

roq_choice = st.checkbox("Calculate average rate of change between years?")
if roq_choice:
    year1 = st.selectbox("First year", list(edited_df['Year']))
    year2 = st.selectbox("Second year", list(edited_df['Year']))

# Regression and plotting
plt.figure(figsize=(10, 6))
x = edited_df['Year'].to_numpy()
analysis_out = []
extrap_x = np.arange(x.min(), x.max() + extrapolate_years + 1, increment)

for col in edited_df.columns[1:]:
    y = edited_df[col].to_numpy()
    popt, _ = curve_fit(poly_fit_fn, x, y, p0=[1]*(degree+1))
    y_fit = poly_fit_fn(extrap_x, *popt)
    is_extrapolated = extrap_x > x.max()
    plt.scatter(x, y, label=f"{col} data")
    plt.plot(
        extrap_x[~is_extrapolated], y_fit[~is_extrapolated],
        label=f"{col} regression",
        linewidth=2
    )
    if extrapolate_option=="Yes" and is_extrapolated.any():
        plt.plot(
            extrap_x[is_extrapolated], y_fit[is_extrapolated], 
            linestyle='dashed',
            color='gray', 
            label=f"{col} extrapolation"
        )
    # Display model equation
    equation = " + ".join(
        [f"{coeff:.3e}x^{idx}" for idx, coeff in enumerate(popt)]
    )
    st.write(f"**Fitted equation for {col}:** y = {equation}")
    
    # Function analysis (local extrema, monotonicity, fastest growth/decay)
    from numpy.polynomial.polynomial import Polynomial
    p = Polynomial(popt)
    dp = p.deriv()
    ddp = p.deriv(2)
    crit_points = dp.roots()
    real_crit = crit_points[(crit_points >= x.min()) & (crit_points <= x.max())]
    crit_texts = []
    for rc in real_crit:
        val = p(rc)
        trend = ddp(rc)
        typ = 'local maximum' if trend < 0 else 'local minimum'
        crit_texts.append(f"The {category.lower()} of {col} reached a {typ} on {int(rc)}; value: {val:.2f}")

    # Growth rates
    dom_in = x.min()
    dom_ax = x.max()
    domain = f"{dom_in} to {dom_ax}"
    range_min, range_max = min(y_fit), max(y_fit)
    for c in crit_texts:
        st.write(c)

    # Monotonicity
    inc_intervals = []
    dec_intervals = []
    last = dom_in
    for i in range(len(extrap_x)-1):
        if dp(extrap_x[i]) > 0:
            inc_intervals.append(extrap_x[i])
        elif dp(extrap_x[i]) < 0:
            dec_intervals.append(extrap_x[i])
    st.write(f"Function is generally increasing over {inc_intervals[0] if inc_intervals else dom_in } to {inc_intervals[-1] if inc_intervals else dom_ax}")
    st.write(f"Function is generally decreasing over {dec_intervals[0] if dec_intervals else dom_in } to {dec_intervals[-1] if dec_intervals else dom_ax}")
    # Fastest growth/decline
    if len(inc_intervals):
        max_rate_idx = np.argmax([dp(xi) for xi in extrap_x])
        st.write(f"Fastest increase at {int(extrap_x[max_rate_idx])}; Rate: {dp(extrap_x[max_rate_idx]):.2f} per year")
    if len(dec_intervals):
        min_rate_idx = np.argmin([dp(xi) for xi in extrap_x])
        st.write(f"Fastest decrease at {int(extrap_x[min_rate_idx])}; Rate: {dp(extrap_x[min_rate_idx]):.2f} per year")
    st.write(f"Domain: years {domain}, range: {range_min:.2f}-{range_max:.2f}")
    # Conjecture on changes
    st.write("**Conjecture:** Significant changes in population growth are often due to policy, economic booms/crises, or health phenomena (e.g., Brazil's 1960s expansion, Argentina's economic crises, Mexico's demographic transition).")
    if output_pred:
        y_year = poly_fit_fn(interp_year, *popt)
        st.write(f"Prediction: {col} {category.lower()} in year {int(interp_year)}: {y_year:.2f} million.")
    if roq_choice:
        idx1 = np.where(extrap_x == int(year1))[0][0]
        idx2 = np.where(extrap_x == int(year2))[0][0]
        roc = (y_fit[idx2] - y_fit[idx1]) / (extrap_x[idx2] - extrap_x[idx1])
        st.write(f"Average rate of change from {year1} to {year2}: {roc:.2f} million per year.")

plt.xlabel("Year")
plt.ylabel(category)
plt.legend()
plt.grid()
st.pyplot(plt)

if st.button("Print results - printer friendly"):
    st.write(edited_df.to_csv(index=False))
    st.write("Copy and paste this summary above into a document to print.")
