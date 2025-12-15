import os
import io
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, request, render_template_string

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

app = Flask(__name__)

# =========================
# UBAH INI kalau nama file CSV kamu beda
# =========================
DATA_PATH = "Mall_Customers.csv"


def fig_to_base64(fig) -> str:
    """Convert figure matplotlib -> base64 PNG string (biar HTML gak perlu file gambar terpisah)."""
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def load_and_prepare_data(include_gender: bool) -> pd.DataFrame:
    """
    PREPROCESSING (sesuai TA-09):
    - Load CSV lokal
    - Drop kolom identitas/noise (CustomerID)
    - Gender optional (encode Female=0, Male=1)
    - Missing value: isi mean
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"File dataset '{DATA_PATH}' tidak ditemukan.\n"
            "Taruh file CSV kamu di folder yang sama dengan app.py.\n"
            "Kalau nama filenya beda, ubah DATA_PATH di bagian atas."
        )

    df = pd.read_csv(DATA_PATH)

    # Drop kolom ID kalau ada
    for col in ["CustomerID", "CUST_ID", "ID", "Id"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            break

    # Gender (opsional)
    if "Gender" in df.columns:
        if include_gender:
            df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
        else:
            df = df.drop(columns=["Gender"])

    # Pastikan numerik
    df = df.apply(pd.to_numeric, errors="coerce")

    # Isi missing values dengan mean
    df = df.fillna(df.mean(numeric_only=True))

    return df


def build_simple_segment_labels(profile: pd.DataFrame) -> tuple[dict, dict]:
    """
    Bikin label segmen + rekomendasi aksi sederhana.
    Fokus ke 2 kolom yang umum di Mall Customers:
    - Annual Income (k$)
    - Spending Score (1-100)
    """
    labels = {}
    recommendations = {}

    income_col = "Annual Income (k$)"
    spend_col = "Spending Score (1-100)"

    if income_col not in profile.columns or spend_col not in profile.columns:
        # fallback kalau kolomnya beda
        for c in profile.index:
            labels[int(c)] = f"Cluster {int(c)}"
            recommendations[int(c)] = "Buat strategi berdasarkan fitur yang paling tinggi/rendah pada cluster ini."
        return labels, recommendations

    income_med = float(profile[income_col].median())
    spend_med = float(profile[spend_col].median())

    for c in profile.index:
        c_int = int(c)
        inc = float(profile.loc[c, income_col])
        sp = float(profile.loc[c, spend_col])

        if inc >= income_med and sp >= spend_med:
            labels[c_int] = "VIP (Income tinggi, belanja tinggi)"
            recommendations[c_int] = "Fokus retensi: membership premium, reward eksklusif, layanan prioritas."
        elif inc >= income_med and sp < spend_med:
            labels[c_int] = "Hemat Tapi Kaya (Income tinggi, belanja rendah)"
            recommendations[c_int] = "Upselling produk premium + bundling, promo minimal pembelian."
        elif inc < income_med and sp >= spend_med:
            labels[c_int] = "Impulsif (Income rendah, belanja tinggi)"
            recommendations[c_int] = "Promo flash sale, kupon kecil tapi sering, rekomendasi produk trending."
        else:
            labels[c_int] = "Low Activity (Income rendah, belanja rendah)"
            recommendations[c_int] = "Re-activation: diskon besar 1x, reminder, campaign ‘ayo belanja lagi’."

    return labels, recommendations


def run_kmeans_pipeline(k: int, include_gender: bool):
    """
    End-to-end TA-09:
    1) Preprocessing
    2) Scaling (StandardScaler)
    3) Elbow Method (k=1..10, WCSS/Inertia)
    4) KMeans final
    5) PCA 2D visualization
    6) Profiling (mean per cluster)
    7) Insight + rekomendasi
    """
    df = load_and_prepare_data(include_gender=include_gender)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    # Elbow (k=1..10)
    ks = list(range(1, 11))
    wcss = []
    for kk in ks:
        km = KMeans(n_clusters=kk, random_state=42, n_init="auto")
        km.fit(X_scaled)
        wcss.append(float(km.inertia_))

    # Silhouette (opsional)
    sil_scores = {}
    for kk in range(2, 11):
        km = KMeans(n_clusters=kk, random_state=42, n_init="auto")
        labels_tmp = km.fit_predict(X_scaled)
        sil_scores[kk] = float(silhouette_score(X_scaled, labels_tmp))

    # KMeans final
    k = int(k)
    if k < 2:
        k = 2
    if k > 10:
        k = 10

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)

    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Plot Elbow
    fig1 = plt.figure(figsize=(7, 4))
    plt.plot(ks, wcss, marker="o")
    plt.title("Elbow Method (WCSS / Inertia)")
    plt.xlabel("Jumlah cluster (k)")
    plt.ylabel("WCSS")
    elbow_b64 = fig_to_base64(fig1)

    # Plot PCA scatter
    fig2 = plt.figure(figsize=(7, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, s=18)
    plt.title("PCA 2D - Hasil KMeans")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    pca_b64 = fig_to_base64(fig2)

    # Profiling
    out_df = df.copy()
    out_df["Cluster"] = clusters
    profile = out_df.groupby("Cluster").mean(numeric_only=True)
    counts = out_df["Cluster"].value_counts().sort_index()

    labels, recommendations = build_simple_segment_labels(profile)

    profile_table = profile.round(2).to_html(classes="table", border=0)
    count_table = pd.DataFrame({"Jumlah Data": counts}).to_html(classes="table", border=0)

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "k": k,
        "elbow_img": elbow_b64,
        "pca_img": pca_b64,
        "profile_table": profile_table,
        "count_table": count_table,
        "sil_scores": sil_scores,
        "labels": labels,
        "recommendations": recommendations,
    }


PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>TA-09 K-Means Clustering - Segmentasi Data Real</title>
  <style>
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #f9f5ff 0, #eef7ff 40%, #f8fafc 100%);
      color: #0f172a;
      line-height: 1.45;
    }
    .app-shell {
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    .topbar {
      background: linear-gradient(90deg, #0f172a, #1d4ed8);
      color: #e5e7eb;
      padding: 16px 32px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .topbar-title {
      font-size: 22px;
      font-weight: 650;
      letter-spacing: 0.02em;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .topbar-badge {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      padding: 2px 10px;
      border-radius: 999px;
      border: 1px solid rgba(191, 219, 254, 0.6);
      background: rgba(15, 23, 42, 0.4);
    }
    .topbar-subtitle {
      font-size: 13px;
      opacity: 0.85;
      margin-top: 4px;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 260px) minmax(0, 1fr);
      gap: 20px;
      padding: 24px 32px 32px;
      max-width: 1280px;
      margin: 0 auto;
      width: 100%;
    }
    @media (max-width: 900px) {
      .layout {
        grid-template-columns: minmax(0, 1fr);
        padding: 18px 16px 24px;
      }
    }
    .card {
      background: rgba(255, 255, 255, 0.9);
      backdrop-filter: blur(10px);
      border-radius: 18px;
      padding: 18px 18px 16px;
      margin-bottom: 16px;
      border: 1px solid rgba(148, 163, 184, 0.25);
      box-shadow: 0 22px 45px rgba(15, 23, 42, 0.12);
    }
    .card h3 {
      margin-top: 0;
      margin-bottom: 10px;
      font-size: 16px;
      font-weight: 600;
      color: #0f172a;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .card h3 span.step {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 22px;
      height: 22px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      background: #1d4ed8;
      color: white;
      box-shadow: 0 0 0 1px rgba(191, 219, 254, 0.9);
    }
    .card + .card {
      margin-top: 16px;
    }
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .main {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .row {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
    }
    img {
      max-width: 100%;
      height: auto;
      border-radius: 16px;
      border: 1px solid rgba(226, 232, 240, 0.8);
      box-shadow: 0 20px 35px rgba(15, 23, 42, 0.18);
    }
    .table {
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }
    .table th,
    .table td {
      border: 1px solid #e5e7eb;
      padding: 7px 9px;
      text-align: left;
    }
    .table th {
      background: linear-gradient(90deg, #eff6ff, #e0f2fe);
      color: #0f172a;
      font-weight: 600;
    }
    code {
      background: #0f172a;
      color: #e5e7eb;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 11px;
    }
    .small {
      color: #6b7280;
      font-size: 0.9em;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      padding: 4px 11px;
      border-radius: 999px;
      margin: 3px 7px 3px 0;
      font-size: 11px;
      border: 1px solid rgba(148, 163, 184, 0.5);
      background: rgba(248, 250, 252, 0.85);
      color: #111827;
      gap: 6px;
    }
    .pill span.k {
      font-weight: 600;
      color: #1d4ed8;
    }
    .warn {
      background: #fef9c3;
      border: 1px solid #facc15;
      padding: 10px 12px;
      border-radius: 12px;
      font-size: 13px;
      display: flex;
      gap: 8px;
      align-items: flex-start;
    }
    .warn-icon {
      font-weight: 700;
      color: #92400e;
    }
    label {
      font-size: 13px;
      color: #111827;
      font-weight: 500;
    }
    input[type="number"] {
      margin-top: 5px;
      padding: 7px 10px;
      border-radius: 10px;
      border: 1px solid #cbd5f5;
      width: 90px;
      font-size: 13px;
      outline: none;
      transition: all 0.18s ease;
    }
    input[type="number"]:focus {
      border-color: #1d4ed8;
      box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.5);
    }
    input[type="checkbox"] {
      accent-color: #1d4ed8;
    }
    button[type="submit"] {
      padding: 8px 18px;
      border-radius: 999px;
      border: none;
      margin-top: 4px;
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.03em;
      cursor: pointer;
      background: linear-gradient(135deg, #1d4ed8, #3b82f6);
      color: white;
      box-shadow: 0 12px 30px rgba(37, 99, 235, 0.45);
      display: inline-flex;
      align-items: center;
      gap: 6px;
      text-transform: uppercase;
    }
    button[type="submit"]:hover {
      transform: translateY(-1px);
      box-shadow: 0 16px 35px rgba(37, 99, 235, 0.6);
    }
    button[type="submit"]:active {
      transform: translateY(0);
      box-shadow: 0 8px 20px rgba(37, 99, 235, 0.5);
    }
    .btn-dot {
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: #bfdbfe;
    }
    .section-tag {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #6b7280;
      margin-bottom: 6px;
    }
    .section-tag-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #22c55e;
      box-shadow: 0 0 0 3px rgba(187, 247, 208, 0.9);
    }
    .cluster-list {
      list-style: none;
      padding-left: 0;
      margin: 0;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .cluster-list li {
      padding: 8px 10px;
      border-radius: 12px;
      background: rgba(248, 250, 252, 0.95);
      border: 1px solid rgba(226, 232, 240, 0.8);
    }
    .cluster-name {
      font-size: 13px;
      font-weight: 600;
      color: #111827;
    }
    .cluster-action {
      font-size: 12px;
    }
    .muted {
      color: #94a3b8;
      font-size: 12px;
      margin-top: 4px;
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <div class="topbar">
      <div class="topbar-title">
        <span>TA-09 · K-Means Clustering</span>
        <span class="topbar-badge">Segmentasi Data Real</span>
      </div>
      <div class="topbar-subtitle">
        Eksplorasi perilaku pelanggan, pilih jumlah cluster terbaik, dan baca insight bisnis dari hasil segmentasi.
      </div>
    </div>

    <div class="layout">
      <div class="sidebar">
        <div class="card">
          <div class="section-tag">
            <span class="section-tag-dot"></span> Dataset
          </div>
          <h3><span class="step">0</span> Dataset Lokal</h3>
          <p class="small">
            Aplikasi ini membaca dataset dari file: <code>{{dataset_name}}</code><br/>
            Pastikan file CSV ada di folder yang sama dengan <code>app.py</code>.
          </p>
          <p class="muted">
            Tip: gunakan dataset <i>Mall_Customers</i> atau dataset serupa yang punya income &amp; spending score.
          </p>
        </div>

        <div class="card">
          <div class="section-tag">
            <span class="section-tag-dot" style="background:#3b82f6; box-shadow:0 0 0 3px rgba(191,219,254,.9);"></span> Konfigurasi
          </div>
          <h3><span class="step">1</span> Pilih Parameter</h3>
          <form method="POST">
            <label>Jumlah cluster (k) (2 - 10)</label><br/>
            <input type="number" name="k" min="2" max="10" value="{{k_default}}" />
            <br/><br/>
            <label>
              <input type="checkbox" name="include_gender" {% if include_gender %}checked{% endif %}/>
              Masukkan fitur <b>Gender</b> (di-encode 0/1)
            </label>
            <br/><br/>
            <button type="submit">
              <span class="btn-dot"></span>
              Jalankan K-Means
            </button>
          </form>
        </div>

        {% if error %}
        <div class="card">
          <div class="warn">
            <span class="warn-icon">!</span>
            <span><b>Error:</b> {{error}}</span>
          </div>
        </div>
        {% endif %}
      </div>

      <div class="main">
        {% if result %}
        <div class="card">
          <div class="section-tag">
            <span class="section-tag-dot" style="background:#22c55e; box-shadow:0 0 0 3px rgba(187,247,208,.9);"></span> Ringkasan
          </div>
          <h3><span class="step">2</span> Ringkasan Data</h3>
          <div class="small">
            Jumlah baris x kolom:
            <b>{{result.shape[0]}} x {{result.shape[1]}}</b>
          </div>
          <div class="small" style="margin-top:4px;">
            Kolom yang dipakai:
            <b>{{ result.columns }}</b>
          </div>
        </div>

        <div class="row">
          <div class="card" style="flex: 1 1 320px;">
            <div class="section-tag">
              <span class="section-tag-dot" style="background:#a855f7; box-shadow:0 0 0 3px rgba(233,213,255,.9);"></span> Evaluasi k
            </div>
            <h3><span class="step">3</span> Elbow Method (k = 1 – 10)</h3>
            <p class="small">
              Cari titik “siku” di mana penurunan WCSS mulai melandai untuk menentukan kandidat k yang bagus.
            </p>
            <img src="data:image/png;base64,{{result.elbow_img}}" alt="Elbow Plot" />
          </div>

          <div class="card" style="flex: 1 1 260px;">
            <div class="section-tag">
              <span class="section-tag-dot" style="background:#f97316; box-shadow:0 0 0 3px rgba(255,237,213,.9);"></span> Kualitas Cluster
            </div>
            <h3><span class="step">4</span> Silhouette Score</h3>
            <p class="small">
              Semakin tinggi nilainya (mendekati 1), semakin baik pemisahan antar cluster.
            </p>
            <div class="small" style="margin-top:6px;">
              {% for kk, sc in result.sil_scores.items() %}
                <span class="pill">
                  <span class="k">k={{kk}}</span>
                  <span>{{ "%.3f"|format(sc) }}</span>
                </span>
              {% endfor %}
            </div>
          </div>
        </div>

        <div class="card">
          <div class="section-tag">
            <span class="section-tag-dot" style="background:#06b6d4; box-shadow:0 0 0 3px rgba(34,211,238,.3);"></span> Visualisasi
          </div>
          <h3><span class="step">5</span> KMeans Final + PCA 2D</h3>
          <p class="small">
            Visualisasi 2D (PCA) untuk melihat sebaran cluster secara intuitif.
          </p>
          <img src="data:image/png;base64,{{result.pca_img}}" alt="PCA Scatter" />
          <p class="small" style="margin-top:8px;">
            <b>k yang dipakai:</b> {{result.k}}
          </p>
        </div>

        <div class="card">
          <div class="section-tag">
            <span class="section-tag-dot" style="background:#6366f1; box-shadow:0 0 0 3px rgba(199,210,254,.9);"></span> Profiling
          </div>
          <h3><span class="step">6</span> Cluster Profiling</h3>
          <p class="small"><b>Jumlah data per cluster</b></p>
          {{ result.count_table | safe }}
          <p class="small" style="margin-top:8px;"><b>Rata-rata fitur per cluster</b></p>
          {{ result.profile_table | safe }}
        </div>

        <div class="card">
          <div class="section-tag">
            <span class="section-tag-dot" style="background:#ef4444; box-shadow:0 0 0 3px rgba(254,202,202,.9);"></span> Insight Bisnis
          </div>
          <h3><span class="step">7</span> Label Segmen &amp; Rekomendasi Aksi</h3>
          <ul class="cluster-list">
            {% for c, name in result.labels.items() %}
              <li>
                <div class="cluster-name">Cluster {{c}} — {{name}}</div>
                <div class="cluster-action small">
                  Aksi: {{result.recommendations[c]}}
                </div>
              </li>
            {% endfor %}
          </ul>
        </div>
        {% else %}
        <div class="card">
          <div class="section-tag">
            <span class="section-tag-dot"></span> Mulai Analisis
          </div>
          <h3><span class="step">2</span> Siap menjalankan K-Means</h3>
          <p class="small">
            Pilih nilai k dan opsi fitur di panel kiri, lalu klik <b>Jalankan K-Means</b> untuk melihat visualisasi dan insight.
          </p>
        </div>
        {% endif %}
      </div>
    </div>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    k_default = 5
    include_gender = False
    result = None
    error = None

    if request.method == "POST":
        try:
            k_default = int(request.form.get("k", "5"))
            include_gender = bool(request.form.get("include_gender"))
            result = run_kmeans_pipeline(k=k_default, include_gender=include_gender)
        except Exception as e:
            error = str(e)

    return render_template_string(
        PAGE,
        dataset_name=DATA_PATH,
        k_default=k_default,
        include_gender=include_gender,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
