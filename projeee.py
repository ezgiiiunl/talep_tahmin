import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="Talep ve Maliyet Tahmin Sistemi",
    layout="wide"
)

st.markdown("""
<style>
/* Uygulama genelindeki boşlukları azaltır */
.block-container {
    padding-top: 4.5rem !important;
    padding-bottom: 0rem !important;
}

.stApp {
    background: #f6f8fc;
    font-family: 'Segoe UI', sans-serif;
}

/* BAŞLIK */
.title {
    text-align:center;
    font-size:32px;
    font-weight:800;
    color:#1f2937;
    margin-bottom: 5px;
}

.subtitle {
    text-align:center;
    color:#6b7280;
    margin-bottom:20px;
    font-size:14px;
}

/* CARD - Boşluklar minimize edildi */
.card {
    background:white;
    padding:12px 18px;
    border-radius:14px;
    box-shadow:0 2px 8px rgba(0,0,0,0.05);
    margin-bottom:10px;
    border:1px solid #eef2f7;
}

/* KPI */
.kpi {
    background: linear-gradient(135deg,#4f46e5,#6366f1);
    color:white;
    padding:15px;
    border-radius:14px;
    text-align:center;
    margin-bottom: 10px;
}

.kpi-title {
    font-size:12px;
    opacity:0.85;
}

.kpi-value {
    font-size:20px;
    font-weight:700;
    margin-top:3px;
}

/* Tabloyu daralt */
div[data-testid="stDataFrame"] {
    margin-top: 10px;
}

.success {
    background:#ecfdf5;
    border-left:4px solid #10b981;
    padding:12px;
    border-radius:10px;
    margin-top: 5px;
}

.warning {
    background:#fff7ed;
    border-left:4px solid #f59e0b;
    padding:12px;
    border-radius:10px;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)


data = pd.read_csv("stoktahminveri(2).csv")

st.markdown('<div class="title">Talep & Maliyet Tahmin Sistemi</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Linear Regresyon ile Karar Destek Paneli</div>', unsafe_allow_html=True)


X = data[["stok_miktari", "yil_icinde_kullanilan", "gecen_yil_kullanilan", "yil_ici_alinan"]]
Y = data["siparis_miktari"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_test)

mse = mean_squared_error(Y_test, pred)
r2 = r2_score(Y_test, pred)


c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f'<div class="kpi"><div class="kpi-title">Model</div><div class="kpi-value">Linear Regression</div></div>',
        unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi"><div class="kpi-title">Hata (MSE)</div><div class="kpi-value">{mse:.2f}</div></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown(
        f'<div class="kpi"><div class="kpi-title">Doğruluk (R²)</div><div class="kpi-value">{r2:.2f}</div></div>',
        unsafe_allow_html=True)


st.markdown('<div class="card">', unsafe_allow_html=True)
secim = st.multiselect("Ürün Seçimi", data["urun_adi"].unique())
st.markdown('</div>', unsafe_allow_html=True)

if secim:
    secilen = data[data["urun_adi"].isin(secim)].copy()
    tahminler = {}

    st.markdown('<div class="card">', unsafe_allow_html=True)
    # Number inputlar arası boşluğu daraltmak için sütunlar kullanılabilir veya CSS halleder
    for i, row in secilen.iterrows():
        tahminler[i] = st.number_input(
            f"{row['urun_adi']} - Kullanıcı Tahmini",
            min_value=0,
            step=1,
            key=i
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Hesaplamalar
    secilen["model_tahmin"] = model.predict(secilen[X.columns])
    secilen["kullanici_tahmin"] = secilen.index.map(tahminler)
    secilen["maliyet"] = secilen["model_tahmin"] * secilen["birim_maliyet"]


    def yorum(m, k):
        if k == 0: return "Girilmedi"
        if k > m: return "Fazla"
        if k < m: return "Az"
        return "Uygun"


    secilen["durum"] = secilen.apply(lambda x: yorum(x["model_tahmin"], x["kullanici_tahmin"]), axis=1)


    st.markdown('<div class="card">', unsafe_allow_html=True)
    b_col1, b_col2 = st.columns([2, 1])
    with b_col1:
        butce = st.number_input("Bütçe (TL)", value=800000)

    toplam = secilen["maliyet"].sum()

    with b_col2:
        st.markdown(f"""
        <div class="kpi" style="margin-bottom:0; padding:10px;">
        <div class="kpi-title">Toplam Maliyet</div>
        <div class="kpi-value" style="font-size:18px;">{toplam:,.0f} TL</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(
        secilen[["urun_adi", "stok_miktari", "model_tahmin", "kullanici_tahmin", "maliyet", "durum"]],
        use_container_width=True
    )

    if toplam <= butce:
        st.markdown('<div class="success">Bütçeyi aşmıyor.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning">Bütçe aşılıyor, optimizasyon uygulanıyor.</div>', unsafe_allow_html=True)
        # Optimizasyon mantığı
        secilen["oncelik"] = secilen["yil_icinde_kullanilan"] / (secilen["stok_miktari"] + 1)
        secilen = secilen.sort_values("oncelik", ascending=False)
        liste = []
        t = 0
        for _, r in secilen.iterrows():
            if t + r["maliyet"] <= butce:
                liste.append(r)
                t += r["maliyet"]
        if liste:
            opt = pd.DataFrame(liste)
            st.dataframe(opt[["urun_adi", "maliyet", "oncelik"]], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="warning">Lütfen ürün seçiniz</div>', unsafe_allow_html=True)
