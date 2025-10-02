# app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import beta

st.set_page_config(page_title="Ordenes Uniforme(0,1)", layout="wide")

# -------- Apariencia (gráficos más pequeños) --------
plt.rcParams.update({
    "figure.figsize": (4.6, 3.0),   # ancho x alto en pulgadas (más pequeño)
    "figure.dpi": 110,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.6
})

def small_fig():
    # helper por si quieres forzar tamaño en cada plot
    return plt.subplots(figsize=(4.6, 3.0), dpi=110)

# =========================
# Sidebar (parámetros)
# =========================
st.sidebar.title("Parámetros de simulación")
N = st.sidebar.number_input("Tamaño de muestra N (≥2)", min_value=2, max_value=2000, value=50, step=1)
m = st.sidebar.number_input("Número de réplicas m (1–1000)", min_value=1, max_value=1000, value=500, step=1)
seed = st.sidebar.number_input("Semilla (opcional)", min_value=0, max_value=10**9, value=1234, step=1)
k_focus = st.sidebar.slider("Orden k para gráficos detallados", min_value=1, max_value=int(N), value=min(5, int(N)))

st.sidebar.markdown("---")
st.sidebar.caption("Tip: aumenta m para ver convergencia en promedios y varianzas.")

# Mensaje informativo si m es pequeño
if m < 50:
    st.info("m es pequeño: la comparación empírica-teórica puede ser ruidosa. Sube m para ver mejor la convergencia.")

# =========================
# Utilidades teóricas
# =========================
def theoretical_mean_var(N):
    k = np.arange(1, N + 1)
    mean = k / (N + 1.0)
    var = (k * (N + 1 - k)) / ((N + 1.0)**2 * (N + 2.0))
    return k, mean, var

# =========================
# Simulación (cache)
# =========================
@st.cache_data(show_spinner=False)
def simulate_sorted_order_stats(N, m, seed):
    rng = np.random.default_rng(seed)
    X = rng.random((m, N))
    X_sorted = np.sort(X, axis=1)                 # cada fila es una muestra, ordenada ascendente
    emp_mean = X_sorted.mean(axis=0)              # promedio por orden k
    emp_var = X_sorted.var(axis=0, ddof=1)        # var empírica por orden k
    return X_sorted, emp_mean, emp_var

X_sorted, emp_mean, emp_var = simulate_sorted_order_stats(N, m, seed)
k_vec, th_mean, th_var = theoretical_mean_var(N)

# =========================
# Tabla de comparación
# =========================
df = pd.DataFrame({
    "k": k_vec,
    "Empirical mean": emp_mean,
    "Theoretical mean": th_mean,
    "Mean diff": emp_mean - th_mean,
    "Empirical var": emp_var,
    "Theoretical var": th_var,
    "Var diff": emp_var - th_var
})

st.title("Estadísticos de orden para Uniforme(0,1)")
st.write(
    f"Simulación con **m = {m}** réplicas y tamaño **N = {N}**. "
    f"El orden k-th teórico ~ Beta(k, N+1−k)."
)

# Resumen de errores
rmse_mean = np.sqrt(np.mean((emp_mean - th_mean)**2))
rmse_var = np.sqrt(np.mean((emp_var - th_var)**2))

col_a, col_b, col_c = st.columns(3)
col_a.metric("RMSE (means)", f"{rmse_mean:.4f}")
col_b.metric("RMSE (variances)", f"{rmse_var:.4f}")
col_c.metric("k seleccionado", k_focus)

st.markdown("### Comparación empírica vs teórica (redondeado a 2 decimales)")
st.dataframe(df.round(2), use_container_width=True)

# Descargar CSV
csv = df.to_csv(index=False)
st.download_button("Descargar tabla CSV", data=csv, file_name="order_stats_uniform.csv", mime="text/csv")

st.markdown("---")

# =========================
# Gráfico: medias por orden
# =========================
st.subheader("Promedios por orden k")
fig1, ax1 = small_fig()
ax1.plot(k_vec, th_mean, label="Teórico")
ax1.plot(k_vec, emp_mean, label="Empírico", linestyle="--")
ax1.set_xlabel("k")
ax1.set_ylabel("Media del estadístico de orden")
ax1.set_title("Medias: empírico vs teórico")
ax1.legend(frameon=False)
st.pyplot(fig1, clear_figure=True, use_container_width=False)

# =========================
# Gráfico: varianzas por orden
# =========================
st.subheader("Varianzas por orden k")
fig2, ax2 = small_fig()
ax2.plot(k_vec, th_var, label="Teórico")
ax2.plot(k_vec, emp_var, label="Empírico", linestyle="--")
ax2.set_xlabel("k")
ax2.set_ylabel("Varianza del estadístico de orden")
ax2.set_title("Varianzas: empírico vs teórico")
ax2.legend(frameon=False)
st.pyplot(fig2, clear_figure=True, use_container_width=False)

st.markdown("---")

# =========================
# Detalle para un k específico
# =========================
st.subheader(f"Detalle para k = {k_focus}")
# Muestra empírica del k-ésimo orden en las m réplicas
sample_k = X_sorted[:, k_focus - 1]

col1, col2 = st.columns(2)

# Parámetros para visuales adaptados a m (mantengo pequeño)
bins = max(5, int(np.sqrt(m)))  # 5 mínimo; ~sqrt(m) recomendado
pt_size = 10 if m <= 200 else 6 if m <= 800 else 3

# Histograma con PDF teórica
with col1:
    fig_h, ax_h = small_fig()
    ax_h.hist(sample_k, bins=bins, density=True, alpha=0.6)
    a, b = k_focus, N + 1 - k_focus
    xs = np.linspace(0, 1, 400)
    ax_h.plot(xs, beta.pdf(xs, a, b))
    ax_h.set_title("Histograma vs PDF Beta(k, N+1-k)")
    ax_h.set_xlabel("Valor")
    ax_h.set_ylabel("Densidad")
    st.pyplot(fig_h, clear_figure=True, use_container_width=False)

# QQ-plot contra Beta teórica
with col2:
    fig_q, ax_q = small_fig()
    emp_sorted = np.sort(sample_k)
    u = (np.arange(1, m + 1) - 0.5) / m
    th_quant = beta.ppf(u, a, b)
    ax_q.scatter(th_quant, emp_sorted, s=pt_size, alpha=0.65)
    lim0 = min(th_quant[0], emp_sorted[0])
    lim1 = max(th_quant[-1], emp_sorted[-1])
    ax_q.plot([lim0, lim1], [lim0, lim1])
    ax_q.set_title("QQ-plot: empírico vs Beta(k, N+1-k)")
    ax_q.set_xlabel("Cuantiles teóricos")
    ax_q.set_ylabel("Cuantiles empíricos")
    st.pyplot(fig_q, clear_figure=True, use_container_width=False)

# =========================
# Notas teóricas
# =========================
with st.expander("Detalles teóricos"):
    st.markdown(
        r"""
Para \(X_1,\dots,X_N \stackrel{iid}{\sim} U(0,1)\), el estadístico de orden \(X_{(k)}\) tiene
distribución **Beta**\((k,\;N{+}1{-}k)\). Por tanto:
\[
\mathbb{E}[X_{(k)}] = \frac{k}{N+1},\qquad
\operatorname{Var}(X_{(k)}) = \frac{k\,(N+1-k)}{(N+1)^2\,(N+2)}.
\]
Aquí comparamos promedios y varianzas empíricas (a partir de \(m\) réplicas) con estos valores teóricos.
        """
    )
