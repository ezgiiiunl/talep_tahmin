import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Veri setini oku
data = pd.read_csv("stoktahminveri(2).csv")

# Başlık ve arka plan stili
st.markdown("""
    <h1 style='text-align: center; color: #6B74B0;'>Yedek Parça ve Malzemeler İçin Maliyet ve Talep Tahmin Sistemi</h1>
    <hr style='border-top: 1px solid #bbb;'/>
    <style>
    .stApp {
        background-color: #D4D1DE;
    }
    </style>
""", unsafe_allow_html=True)

# Model eğitimi
X = data[["stok_miktari", "yil_icinde_kullanilan", "gecen_yil_kullanilan", "yil_ici_alinan"]]
Y = data["siparis_miktari"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
Y_pred_lr = lr_model.predict(X_test)

mse_lr = mean_squared_error(Y_test, Y_pred_lr)
r2_lr = r2_score(Y_test, Y_pred_lr)

st.subheader("Model Hata Oranları")
st.write(f"Doğrusal Regresyon - Ortalama Kare Hatası (MSE): {mse_lr:.2f}, R2 Skoru: {r2_lr:.2f}")
st.success("Kullanılan model: LinearRegression")

# Ürün seçimi
secim = st.multiselect("Ürün Seçiniz", data["urun_adi"].unique())

if secim:
    secilen_veri = data[data["urun_adi"].isin(secim)].copy()

    # Model tahmini
    secilen_veri["tahmini_siparis"] = lr_model.predict(
        secilen_veri[["stok_miktari", "yil_icinde_kullanilan", "gecen_yil_kullanilan", "yil_ici_alinan"]]
    )
    secilen_veri["tahmini_maliyet"] = secilen_veri["tahmini_siparis"] * secilen_veri["birim_maliyet"]

    # Bütçe kısıtı
    Butce_limiti = st.number_input("Bütçenizi giriniz (TL)", min_value=0, value=800000, step=1000)
    toplam_maliyet = secilen_veri["tahmini_maliyet"].sum()

    st.write(f"Seçilen ürünlerin toplam tahmini maliyeti: {toplam_maliyet:.2f} TL")

    st.markdown("### Seçilen Ürünler ve Tahminler")
    st.dataframe(secilen_veri[[
        "urun_adi",
        "stok_miktari",
        "yil_icinde_kullanilan",
        "gecen_yil_kullanilan",
        "yil_ici_alinan",
        "birim_maliyet",
        "tahmini_siparis",
        "tahmini_maliyet"
    ]])

    # Bütçe kontrolü
    if toplam_maliyet <= Butce_limiti:
        st.success("Seçilen tüm ürünler bütçeye uygun.")
    else:
        st.warning("Bütçenizi aştınız, sistem en öncelikli ürünleri önerecek.")

        # Öncelik hesaplama ve sıralama
        secilen_veri["oncelik"] = secilen_veri["yil_icinde_kullanilan"] / (secilen_veri["stok_miktari"] + 1)
        secilen_veri = secilen_veri.sort_values(by="oncelik", ascending=False)

        secilenler = []
        toplam = 0
        for _, row in secilen_veri.iterrows():
            if toplam + row["tahmini_maliyet"] <= Butce_limiti:
                secilenler.append(row)
                toplam += row["tahmini_maliyet"]

        secilen_df = pd.DataFrame(secilenler)

        st.markdown(f"""
            <div style='background-color:#BABEE0; padding:15px; border-radius:10px; border:1px solid purple;'>
                <strong style='color:#6D5FB3;'>Bütçe aşıldığı için yalnızca {len(secilen_df)} ürün önerildi.</strong>
            </div>
        """, unsafe_allow_html=True)

        st.write(f"Önerilen ürünlerin toplam maliyeti: {toplam:.2f} TL")

        st.dataframe(secilen_df[[
            "urun_adi",
            "stok_miktari",
            "yil_icinde_kullanilan",
            "gecen_yil_kullanilan",
            "yil_ici_alinan",
            "birim_maliyet",
            "tahmini_siparis",
            "tahmini_maliyet"
        ]])
else:
    st.info("Lütfen ürün veya ürünler seçiniz.")






















