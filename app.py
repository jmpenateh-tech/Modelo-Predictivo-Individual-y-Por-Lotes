import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #c0392b;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #c0392b;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 0.8rem; color: #555; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #2c3e50; }
    .positive-pred {
        background: #fde8e8; border: 2px solid #c0392b; border-radius: 12px;
        padding: 1.5rem; text-align: center; margin: 1rem 0;
    }
    .negative-pred {
        background: #e8f5e9; border: 2px solid #27ae60; border-radius: 12px;
        padding: 1.5rem; text-align: center; margin: 1rem 0;
    }
    .pred-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }
    .pred-prob  { font-size: 1rem; color: #555; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    .dataset-badge {
        display: inline-block; background: #c0392b; color: white;
        border-radius: 20px; padding: 2px 12px; font-size: 0.8rem; margin: 2px;
    }
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #2c3e50;
        border-bottom: 2px solid #c0392b; padding-bottom: 0.4rem; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data loading & preprocessing ───────────────────────────────────────────
COLS = ['age','sex','cp','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal','target']

FEATURE_COLS = ['age','sex','cp','trestbps','chol','fbs','restecg',
                'thalach','exang','oldpeak','slope','ca','thal']

SOURCES = {
    'Cleveland':    'processed.cleveland.data',
    'Hungarian':    'processed.hungarian.data',
    'Switzerland':  'processed.switzerland.data',
    'Long Beach VA':'processed.va.data',
}

COL_INFO = {
    'age':      ('Edad (años)', 29, 77, 54, None),
    'sex':      ('Sexo', 0, 1, 1, {0:'Femenino', 1:'Masculino'}),
    'cp':       ('Tipo de dolor torácico', 1, 4, 3,
                 {1:'Angina típica', 2:'Angina atípica', 3:'Dolor no anginoso', 4:'Asintomático'}),
    'trestbps': ('Presión arterial en reposo (mmHg)', 80, 200, 130, None),
    'chol':     ('Colesterol sérico (mg/dl)', 100, 600, 246, None),
    'fbs':      ('Glucosa en ayunas > 120 mg/dl', 0, 1, 0, {0:'No', 1:'Sí'}),
    'restecg':  ('Resultados ECG en reposo', 0, 2, 0,
                 {0:'Normal', 1:'Anormalidad ST-T', 2:'HVI por criterios Estes'}),
    'thalach':  ('Frecuencia cardíaca máxima', 60, 210, 150, None),
    'exang':    ('Angina inducida por ejercicio', 0, 1, 0, {0:'No', 1:'Sí'}),
    'oldpeak':  ('Depresión ST por ejercicio', 0.0, 6.2, 1.0, None),
    'slope':    ('Pendiente segmento ST pico', 1, 3, 2,
                 {1:'Ascendente', 2:'Plano', 3:'Descendente'}),
    'ca':       ('Nº vasos coloreados (fluoroscopía)', 0, 3, 0, None),
    'thal':     ('Talasemia', 3, 7, 3,
                 {3:'Normal', 6:'Defecto fijo', 7:'Defecto reversible'}),
}

@st.cache_resource(show_spinner=False)
def load_and_train():
    """Load all four datasets, combine, binarize target, train two models."""
    dfs = []
    for name, fname in SOURCES.items():
        path = f"heart+disease/{fname}"
        df = pd.read_csv(path, header=None, names=COLS, na_values='?')
        df['source'] = name
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)

    # Binarize: 0 = no disease, 1 = disease (values 1-4)
    raw['target'] = (raw['target'] > 0).astype(int)

    # Impute missing values with median (per feature)
    for col in FEATURE_COLS:
        if raw[col].isnull().any():
            raw[col] = raw[col].fillna(raw[col].median())

    X = raw[FEATURE_COLS].values
    y = raw['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(X_train_s, y_train)

    # Neural Network (MLP)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        alpha=0.01,
        max_iter=2000,
        random_state=42,
    )
    mlp.fit(X_train_s, y_train)

    return {
        'raw': raw,
        'scaler': scaler,
        'lr': lr,
        'mlp': mlp,
        'X_test_s': X_test_s,
        'y_test': y_test,
        'feature_names': FEATURE_COLS,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_total': len(raw),
    }

# ─── Metric helpers ──────────────────────────────────────────────────────────
def compute_metrics(model, X, y_true):
    y_pred = model.predict(X)
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'cm':        confusion_matrix(y_true, y_pred),
        'y_pred':    y_pred,
    }

def render_metric_card(label, value, fmt='.2%'):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value:{fmt}}</div>
    </div>""", unsafe_allow_html=True)

def plot_cm(cm, title, cmap):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Sin enfermedad','Con enfermedad'],
                yticklabels=['Sin enfermedad','Con enfermedad'],
                linewidths=0.5, linecolor='#ccc')
    ax.set_xlabel('Predicción', fontsize=10)
    ax.set_ylabel('Real', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">🫀 Heart Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predicción de enfermedad cardíaca con Regresión Logística y Red Neuronal</div>', unsafe_allow_html=True)

with st.spinner("⚙️ Entrenando modelos con los 4 datasets combinados…"):
    state = load_and_train()

# ── Sidebar summary ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Dataset")
    raw = state['raw']
    for src, badge in [('Cleveland',''), ('Hungarian',''), ('Switzerland',''), ('Long Beach VA','')]:
        n = (raw['source'] == src).sum()
        st.markdown(f"<span class='dataset-badge'>{src}</span> {n} registros", unsafe_allow_html=True)
    st.markdown(f"**Total:** {state['n_total']} registros  \n**Entrenamiento:** {state['n_train']} | **Prueba:** {state['n_test']}")

    st.markdown("---")
    st.markdown("### 🎯 Variable objetivo")
    vc = raw['target'].value_counts()
    fig_s, ax_s = plt.subplots(figsize=(3, 2.2))
    ax_s.bar(['Sin enfermedad','Con enfermedad'], [vc.get(0,0), vc.get(1,0)],
             color=['#27ae60','#c0392b'])
    ax_s.set_ylabel("Pacientes", fontsize=8)
    ax_s.tick_params(labelsize=7)
    plt.tight_layout()
    st.pyplot(fig_s, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.markdown("### 🤖 Modelos")
    st.markdown("- **Regresión Logística**\n- **Red Neuronal (MLP)**")
    st.markdown("*Los modelos se re-entrenan en cada carga de página.*")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Predicción Individual", "📂 Predicción por Lotes"])

# ════════════════════════════════════════════════
# TAB 1 – PREDICCIÓN INDIVIDUAL
# ════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Ingrese los datos del paciente</div>', unsafe_allow_html=True)

    model_choice = st.radio(
        "Seleccionar modelo",
        ["Regresión Logística", "Red Neuronal (MLP)"],
        horizontal=True,
    )

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**🧍 Datos demográficos**")
            age      = st.slider("Edad (años)", 29, 77, 54)
            sex_opt  = st.selectbox("Sexo", [0, 1], format_func=lambda x: COL_INFO['sex'][4][x])
            cp_opt   = st.selectbox("Tipo de dolor torácico", [1,2,3,4],
                                    format_func=lambda x: COL_INFO['cp'][4][x])

            st.markdown("**💉 Parámetros clínicos**")
            trestbps = st.slider("Presión arterial reposo (mmHg)", 80, 200, 130)
            chol     = st.slider("Colesterol sérico (mg/dl)", 100, 600, 246)
            fbs_opt  = st.selectbox("Glucosa ayunas > 120 mg/dl", [0, 1],
                                    format_func=lambda x: COL_INFO['fbs'][4][x])

        with c2:
            st.markdown("**📈 Electrocardiograma**")
            restecg_opt = st.selectbox("ECG en reposo", [0,1,2],
                                       format_func=lambda x: COL_INFO['restecg'][4][x])
            thalach  = st.slider("FC máxima alcanzada", 60, 210, 150)
            exang_opt= st.selectbox("Angina por ejercicio", [0, 1],
                                    format_func=lambda x: COL_INFO['exang'][4][x])

            st.markdown("**📉 Segmento ST**")
            oldpeak  = st.slider("Depresión ST (mm)", 0.0, 6.2, 1.0, 0.1)
            slope_opt= st.selectbox("Pendiente ST pico", [1,2,3],
                                    format_func=lambda x: COL_INFO['slope'][4][x])

        with c3:
            st.markdown("**🩻 Imagen & laboratorio**")
            ca       = st.slider("Vasos coloreados (0-3)", 0, 3, 0)
            thal_opt = st.selectbox("Talasemia", [3,6,7],
                                    format_func=lambda x: COL_INFO['thal'][4][x])

            st.markdown(" ")
            st.markdown(" ")
            submitted = st.form_submit_button("🔮 Predecir", use_container_width=True, type="primary")

    if submitted:
        features = np.array([[age, sex_opt, cp_opt, trestbps, chol,
                               fbs_opt, restecg_opt, thalach, exang_opt,
                               oldpeak, slope_opt, ca, thal_opt]])

        feats_s = state['scaler'].transform(features)
        model = state['lr'] if model_choice == "Regresión Logística" else state['mlp']

        pred = model.predict(feats_s)[0]
        prob = model.predict_proba(feats_s)[0]

        st.markdown("---")
        r1, r2 = st.columns([1, 1])
        with r1:
            if pred == 1:
                st.markdown(f"""
                <div class="positive-pred">
                    <div class="pred-title">❤️‍🩹 Enfermedad cardíaca detectada</div>
                    <div class="pred-prob">Probabilidad: <b>{prob[1]:.1%}</b></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="negative-pred">
                    <div class="pred-title">💚 Sin enfermedad cardíaca</div>
                    <div class="pred-prob">Probabilidad: <b>{prob[0]:.1%}</b></div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"**Modelo usado:** {model_choice}")

        with r2:
            fig_p, ax_p = plt.subplots(figsize=(4, 2.5))
            colors = ['#27ae60', '#c0392b']
            bars = ax_p.barh(['Sin enfermedad', 'Con enfermedad'], [prob[0], prob[1]],
                             color=colors, height=0.5)
            for bar, val in zip(bars, [prob[0], prob[1]]):
                ax_p.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                          f'{val:.1%}', va='center', fontweight='bold', fontsize=11)
            ax_p.set_xlim(0, 1.15)
            ax_p.set_xlabel("Probabilidad", fontsize=9)
            ax_p.set_title("Probabilidades de predicción", fontsize=10, fontweight='bold')
            ax_p.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig_p, use_container_width=True)
            plt.close()

        # Feature importance (LR only)
        if model_choice == "Regresión Logística":
            st.markdown("---")
            st.markdown("**Importancia de variables (coeficientes del modelo)**")
            coefs = pd.Series(
                np.abs(model.coef_[0]),
                index=FEATURE_COLS
            ).sort_values(ascending=True)
            fig_c, ax_c = plt.subplots(figsize=(6, 3.5))
            ax_c.barh(coefs.index, coefs.values, color='#c0392b', alpha=0.8)
            ax_c.set_xlabel("|Coeficiente|", fontsize=9)
            ax_c.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_c, use_container_width=True)
            plt.close()

# ════════════════════════════════════════════════
# TAB 2 – PREDICCIÓN POR LOTES
# ════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Cargue un archivo CSV para predicción masiva</div>',
                unsafe_allow_html=True)

    st.info(
        "El CSV debe tener las columnas: **age, sex, cp, trestbps, chol, fbs, restecg, "
        "thalach, exang, oldpeak, slope, ca, thal**  \n"
        "Opcionalmente puede incluir **target** (0/1) para calcular métricas y matriz de confusión."
    )

    # Download template
    template_df = pd.DataFrame(
        [[54, 1, 3, 130, 246, 0, 0, 150, 0, 1.0, 2, 0, 3]],
        columns=FEATURE_COLS
    )
    st.download_button(
        "⬇️ Descargar plantilla CSV",
        template_df.to_csv(index=False),
        file_name="plantilla_prediccion.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Cargar CSV de pacientes", type="csv")

    if uploaded:
        try:
            user_df = pd.read_csv(uploaded)
            st.success(f"✅ {len(user_df)} registros cargados")

            has_target = 'target' in user_df.columns
            if has_target:
                user_df['target'] = (user_df['target'] > 0).astype(int)

            # Check columns
            missing_cols = [c for c in FEATURE_COLS if c not in user_df.columns]
            if missing_cols:
                st.error(f"Columnas faltantes: {missing_cols}")
            else:
                X_user = user_df[FEATURE_COLS].copy()
                for col in FEATURE_COLS:
                    if X_user[col].isnull().any():
                        X_user[col] = X_user[col].fillna(X_user[col].median())

                X_user_s = state['scaler'].transform(X_user.values)

                model_b = st.selectbox(
                    "Seleccionar modelo para predicción por lotes",
                    ["Regresión Logística", "Red Neuronal (MLP)"],
                    key="batch_model"
                )
                model_obj = state['lr'] if model_b == "Regresión Logística" else state['mlp']

                preds = model_obj.predict(X_user_s)
                probas = model_obj.predict_proba(X_user_s)

                user_df['prediccion'] = preds
                user_df['prob_sin_enfermedad'] = probas[:, 0].round(3)
                user_df['prob_con_enfermedad'] = probas[:, 1].round(3)
                user_df['resultado'] = user_df['prediccion'].map(
                    {0: '✅ Sin enfermedad', 1: '❌ Con enfermedad'})

                # Summary
                n_pos = int(preds.sum())
                n_neg = len(preds) - n_pos
                k1, k2, k3 = st.columns(3)
                k1.metric("Total pacientes", len(preds))
                k2.metric("Con enfermedad",  n_pos, delta=f"{n_pos/len(preds):.1%}")
                k3.metric("Sin enfermedad",  n_neg, delta=f"{n_neg/len(preds):.1%}")

                # Results table
                st.markdown("**Resultados de predicción**")
                display_cols = FEATURE_COLS + (['target'] if has_target else []) + \
                               ['prob_con_enfermedad', 'resultado']
                st.dataframe(user_df[display_cols], use_container_width=True, height=300)

                # Download
                st.download_button(
                    "⬇️ Descargar resultados CSV",
                    user_df.to_csv(index=False),
                    file_name="predicciones_cardiacas.csv",
                    mime="text/csv",
                )

                # Confusion matrix section (only if target present)
                if has_target:
                    st.markdown("---")
                    st.markdown('<div class="section-title">Métricas del conjunto cargado</div>',
                                unsafe_allow_html=True)

                    y_true_b = user_df['target'].values
                    y_pred_b = preds
                    cm_b = confusion_matrix(y_true_b, y_pred_b)
                    acc_b = accuracy_score(y_true_b, y_pred_b)
                    prec_b = precision_score(y_true_b, y_pred_b, zero_division=0)
                    rec_b  = recall_score(y_true_b, y_pred_b, zero_division=0)
                    f1_b   = f1_score(y_true_b, y_pred_b, zero_division=0)

                    m1, m2, m3, m4 = st.columns(4)
                    with m1: render_metric_card("Accuracy",  acc_b)
                    with m2: render_metric_card("Precisión", prec_b)
                    with m3: render_metric_card("Recall",    rec_b)
                    with m4: render_metric_card("F1-Score",  f1_b)

                    col_cm, col_dist = st.columns([1, 1])
                    with col_cm:
                        cmap = 'Reds' if model_b == "Regresión Logística" else 'Blues'
                        fig_cm = plot_cm(cm_b, f"Matriz de Confusión\n{model_b}", cmap)
                        st.pyplot(fig_cm, use_container_width=True)
                        plt.close()

                    with col_dist:
                        fig_d, ax_d = plt.subplots(figsize=(4.5, 3.8))
                        ax_d.hist(probas[y_true_b==0, 1], bins=20, alpha=0.7,
                                  color='#27ae60', label='Real: Sin enfermedad')
                        ax_d.hist(probas[y_true_b==1, 1], bins=20, alpha=0.7,
                                  color='#c0392b', label='Real: Con enfermedad')
                        ax_d.set_xlabel("Probabilidad predicha (con enfermedad)", fontsize=9)
                        ax_d.set_ylabel("Pacientes", fontsize=9)
                        ax_d.set_title("Distribución de probabilidades", fontsize=10, fontweight='bold')
                        ax_d.legend(fontsize=8)
                        plt.tight_layout()
                        st.pyplot(fig_d, use_container_width=True)
                        plt.close()
                else:
                    st.info("💡 Incluya una columna **target** (0/1) en su CSV para ver métricas y matriz de confusión.")

        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")

    # ── Test set results (always shown) ──────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Resultados en datos de prueba internos</div>',
                unsafe_allow_html=True)

    tab_lr, tab_nn = st.tabs(["📈 Regresión Logística", "🧠 Red Neuronal (MLP)"])

    for tab_m, model_key, model_label, cmap in [
        (tab_lr, 'lr', "Regresión Logística", "Reds"),
        (tab_nn, 'mlp', "Red Neuronal (MLP)", "Blues"),
    ]:
        with tab_m:
            m = compute_metrics(state[model_key], state['X_test_s'], state['y_test'])
            ma1, ma2, ma3, ma4 = st.columns(4)
            with ma1: render_metric_card("Accuracy",  m['accuracy'])
            with ma2: render_metric_card("Precisión", m['precision'])
            with ma3: render_metric_card("Recall",    m['recall'])
            with ma4: render_metric_card("F1-Score",  m['f1'])

            col_a, col_b = st.columns([1, 1])
            with col_a:
                fig_m = plot_cm(m['cm'], f"Matriz de Confusión\n{model_label}", cmap)
                st.pyplot(fig_m, use_container_width=True)
                plt.close()
            with col_b:
                proba_m = state[model_key].predict_proba(state['X_test_s'])
                fig_pr, ax_pr = plt.subplots(figsize=(4.5, 3.8))
                ax_pr.hist(proba_m[state['y_test']==0, 1], bins=20, alpha=0.7,
                           color='#27ae60', label='Sin enfermedad')
                ax_pr.hist(proba_m[state['y_test']==1, 1], bins=20, alpha=0.7,
                           color='#c0392b', label='Con enfermedad')
                ax_pr.set_xlabel("Probabilidad (con enfermedad)", fontsize=9)
                ax_pr.set_ylabel("Pacientes", fontsize=9)
                ax_pr.set_title("Distribución de probabilidades", fontsize=10, fontweight='bold')
                ax_pr.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_pr, use_container_width=True)
                plt.close()

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:0.8rem;'>"
    "🫀 Heart Disease Predictor · UCI Heart Disease Dataset (Cleveland, Hungarian, Switzerland, Long Beach VA) · "
    "Modelos re-entrenados en cada sesión"
    "</div>",
    unsafe_allow_html=True
)
