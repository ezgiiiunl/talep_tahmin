import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("stoktahminveri(2).csv")

st.markdown("""
    <h1 style='text-align: center; color: #6B74B0;'>Yedek Parça Ve Malzemeler İçin Maliyet ve Talep Tahmin Sistemi</h1>
    <hr style='border-top: 1px solid #bbb;'/>
    <style>
    .stApp {
        background-color: #D4D1DE;
    }
    </style>
""", unsafe_allow_html=True)

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
model = lr_model
st.success("Kullanılan model: LinearRegression")

secim = st.multiselect("Ürün Seçiniz", data["urun_adi"].unique())

if secim:
    secilen_veri = data[data["urun_adi"].isin(secim)].copy()
    kullanici_tahminleri = {}

    st.markdown("Kendi Tahminlerinizi Giriniz")
    for idx, row in secilen_veri.iterrows():
        tahmin_input = st.number_input(
            f"{row['urun_adi']} için kendi tahmin ettiğiniz sipariş miktarı",
            min_value=0,
            step=1,
            key=f"tahmin_{idx}"
        )
        kullanici_tahminleri[idx] = tahmin_input

    secilen_veri["tahmini_siparis"] = model.predict(
        secilen_veri[["stok_miktari", "yil_icinde_kullanilan", "gecen_yil_kullanilan", "yil_ici_alinan"]]
    )
    secilen_veri["kullanici_tahmin"] = secilen_veri.index.map(kullanici_tahminleri)

    def yorum(model_tahmin, kullanici):
        if kullanici == 0:
            return "Girilmedi!"
        elif kullanici > model_tahmin:
            return "Fazla tahmin ettiniz,lütfen tahmininizi güncelleyin!"
        elif kullanici < model_tahmin:
            return "Az tahmin ettiniz,lütfen tahmininizi güncelleyin!"
        else:
            return "Tahmininiz doğru!"

    secilen_veri["yorum"] = secilen_veri.apply(
        lambda row: yorum(row["tahmini_siparis"], row["kullanici_tahmin"]), axis=1
    )
    secilen_veri["tahmini_maliyet"] = secilen_veri["tahmini_siparis"] * secilen_veri["birim_maliyet"]

    Butce_limiti = st.number_input("Bütçenizi giriniz (TL)", min_value=0, value=800000, step=1000)
    toplam_maliyet = secilen_veri["tahmini_maliyet"].sum()
    st.write(f"Seçilen ürünlerin toplam tahmini maliyeti: {toplam_maliyet:.2f} TL")

    st.markdown("Seçilen Ürünler ve Tahminler")
    st.dataframe(secilen_veri[[
        "urun_adi",
        "stok_miktari",
        "yil_icinde_kullanilan",
        "gecen_yil_kullanilan",
        "yil_ici_alinan",
        "birim_maliyet",
        "tahmini_siparis",
        "kullanici_tahmin",
        "yorum",
        "tahmini_maliyet"
    ]])

    if toplam_maliyet <= Butce_limiti:
        st.success("Seçilen tüm ürünler bütçeye uygun.")
    else:
        st.markdown("""
        <div style='background-color:#BABEE0; padding:15px; border-radius:10px; border:1px solid purple;'>
            <strong style='color:#6D5FB3;'>Bütçenizi aştınız, ilk olarak belirtilen malzemeleri alın.</strong>
        </div>
        """, unsafe_allow_html=True)

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
                   <strong style='color:#6D5FB3;'>Bütçe aşıldığı için sadece {len(secilen_df)} ürün önerildi.</strong>
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
            "kullanici_tahmin",
            "yorum",
            "tahmini_maliyet"
        ]])
else:
    st.info("Lütfen ürün veya ürünler seçiniz.")