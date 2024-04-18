import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Membaca file CSV
df = pd.read_csv('Laptop_price.csv', encoding='utf-8')

# Judul dashboard
st.title('PREDIKSI HARGA LAPTOP')

# Gambar header
st.image("https://siplahtelkom.com/public/products/49696/4055520/55171.38327383-0208-4ab1-9396-42a8c6af5db0.laptop-128.png", width=500)

# Deskripsi
st.markdown("""
            Kumpulan data ini mengemulasikan harga laptop, menangkap berbagai fitur yang umumnya dikaitkan dengan laptop dan simulasi harga terkaitnya. 
            Kumpulan data tersebut mencakup atribut utama seperti merek, kecepatan prosesor, ukuran RAM, kapasitas penyimpanan, ukuran layar, dan berat.
            """)

# Sidebar
section = st.sidebar.selectbox("Choose Section", ["EDA", "4 Pilar Visualisasi", "Modelling"])

# Menampilkan EDA
if section == "EDA":
    # Menampilkan DataFrame
    st.header('Dataframe')
    st.write(df)
    st.write("Dataframe yang ditampilkan adalah dataframe yang berisi informasi Merek, Kecepatan Prosesor, Ukuran RAM, Kapasitas Penyimpanan, Ukuran Layar, Berat dan Harga dari laptop.")

    # Menampilkan heatmap korelasi jika terdapat kolom numerik
    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 0:
        st.header('EDA')
        corr_fig = px.imshow(df[numeric_columns].corr(), x=numeric_columns, y=numeric_columns, color_continuous_scale='RdBu', width=800, height=600)
        corr_fig.update_layout(title='Korelasi', title_x=0.5, title_font_size=20)
        st.plotly_chart(corr_fig)
    else:
        st.warning('Tidak ada kolom numerik yang dapat ditampilkan.')
    
    st.markdown("""
                **EDA (Exploratory Data Analysis)**
                ###### Interpretasi:
                Heatmap korelasi menunjukkan hubungan antara berbagai fitur numerik dalam dataset. Korelasi dapat membantu dalam memahami hubungan antarvariabel dalam dataset.
                ###### Insight:
                Heatmap korelasi menunjukkan bahwa ada korelasi positif antara beberapa fitur numerik, seperti RAM dan harga, serta layar dan harga. Hal ini mengindikasikan bahwa ada hubungan yang signifikan antara atribut-atribut ini dalam menentukan harga laptop.
                ###### Actionable Insight:
                Identifikasi fitur-fitur yang memiliki korelasi tinggi dengan harga, seperti RAM dan layar, dan fokus pada fitur-fitur ini dalam analisis lebih lanjut.
                """)

# Menampilkan 4 Pilar Visualisasi
elif section == "4 Pilar Visualisasi":
    # Visualisasi 4 pilar
    st.header('4 Pilar Visualisasi')

    ## Pilar 1: Scatter plot - Harga vs Ukuran Layar
    st.subheader('Pilar 1: Scatter Plot - Harga vs Ukuran Layar')
    scatter_fig = px.scatter(df, x='Screen_Size', y='Price', hover_name='Screen_Size', title='Harga vs Ukuran Layar')
    st.plotly_chart(scatter_fig)
    st.markdown("""
                **Pilar 1: Scatter Plot - Harga vs Ukuran Layar**
                ###### Interpretasi:
                Visualisasi menunjukkan hubungan antara harga laptop dan ukuran layar. Hal ini dapat membantu dalam memahami pola distribusi harga berdasarkan ukuran layar laptop.
                ###### Insight:
                Scatter plot menunjukkan bahwa harga laptop cenderung meningkat dengan ukuran layar yang lebih besar.
                ###### Actionable Insight:
                Berdasarkan scatter plot, pertimbangkan strategi harga yang berbeda untuk laptop dengan ukuran layar yang berbeda.
                """)

    ## Pilar 2: Histogram - Distribusi Harga
    st.subheader('Pilar 2: Histogram - Distribusi Harga')
    hist_fig = px.histogram(df, x='Price', title='Distribusi Harga')
    st.plotly_chart(hist_fig)
    st.markdown("""
                **Pilar 2: Histogram - Distribusi Harga**
                ###### Interpretasi:
                Histogram menampilkan distribusi harga laptop dalam dataset. Informasi ini dapat memberikan wawasan tentang sebaran harga dan kemungkinan range harga yang paling umum.
                ###### Insight:
                Histogram menunjukkan bahwa sebagian besar harga laptop berada pada rentang tertentu, dengan sebagian kecil laptop memiliki harga yang jauh lebih tinggi.
                ###### Actionable Insight:
                Berdasarkan Histogram distribusi harga, Menawarkan promosi atau diskon untuk laptop dengan harga di rentang yang paling umum, untuk menarik konsumen yang lebih memilih produk dengan harga sedang.
                """)

    ## Pilar 3: Bar plot - Distribusi Merek
    st.subheader('Pilar 3: Bar Plot - Distribusi Merek')
    # Membuat plot bar
    bar_fig = px.bar(df, x='Price', y='Brand', title='Distribusi Merek Berdasarkan Harga Laptop', 
                     labels={'Price': 'Harga', 'Brand': 'Merek'})
    st.plotly_chart(bar_fig)
    st.markdown("""
                **Pilar 3: Bar Plot - Distribusi Merek**
                ###### Interpretasi:
                Grafik batang menunjukkan distribusi merek laptop berdasarkan harga. Ini dapat memberikan pemahaman tentang merek mana yang paling banyak terjual pada rentang harga tertentu.
                ###### Insight:
                Grafik batang menunjukkan bahwa beberapa merek memiliki penjualan yang signifikan pada berbagai rentang harga, sementara merek lain mungkin lebih terfokus pada rentang harga tertentu.
                ###### Actionable Insight:
                Dari grafik batang, pertimbangkan untuk mempromosikan merek-merek yang paling banyak terjual pada rentang harga tertentu, atau melihat apakah ada peluang untuk memperluas penjualan merek-merek yang kurang populer di rentang harga yang sama.
                """)

    ## Pilar 4: Pie chart - Distribusi RAM_Size
    st.subheader('Pilar 4: Pie Chart - Distribusi RAM_Size')
    pie_fig = px.pie(df['RAM_Size'].value_counts().reset_index(), 
                     values='RAM_Size', names='RAM_Size', 
                     title='Distribusi Harga Berdasarkan RAM_Size')
    st.plotly_chart(pie_fig)
    st.markdown("""
                **Pilar 4: Pie Chart - Distribusi RAM_Size**
                ###### Interpretasi:
                Pie chart memperlihatkan proporsi harga laptop berdasarkan ukuran RAM. Ini dapat memberikan wawasan tentang preferensi konsumen terhadap ukuran RAM dalam kaitannya dengan harga laptop.
                ###### Insight:
                Pie chart menunjukkan preferensi ukuran RAM tertentu dalam kaitannya dengan harga laptop, yang dapat memberikan wawasan tentang preferensi konsumen dan segmen pasar potensial.
                ###### Actionable Insight:
                Dari pie chart, pertimbangkan untuk menyesuaikan penawaran produk atau strategi pemasaran berdasarkan preferensi ukuran RAM konsumen.
                """)

# Menampilkan Modelling
elif section == "Modelling":
    st.header('Modelling')

   # Load combined data
    combined_data = pd.read_csv('sample_combined_data.csv')

    # Sidebar untuk memilih model
    st.sidebar.title('Pilih Model')
    selected_model = st.sidebar.selectbox('Model yang Dipilih', ['Gaussian Naive Bayes', 'K-Nearest Neighbors', 'Decision Tree'])

   # Menampilkan data dalam bentuk tabel
    st.subheader('Data Hasil Prediksi')
    st.write(combined_data)

    # Memisahkan fitur dan target
    X = combined_data.drop('PriceCategory', axis=1)  # Ubah 'target_column' dengan nama kolom target
    y = combined_data['PriceCategory']

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalisasi data
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # Inisialisasi dan latih model
    def build_train_predict_model(selected_model, X_train, X_test, y_train):
        if selected_model == 'Gaussian Naive Bayes':
            model = GaussianNB()
        elif selected_model == 'K-Nearest Neighbor':
            model = KNeighborsClassifier()
        else:
            model = DecisionTreeClassifier()

        model.fit(X_train, y_train)

        # Lakukan prediksi pada data uji
        y_pred = model.predict(X_test)
        return y_pred
    
    # Mengeksekusi model yang dipilih
    if st.sidebar.button('Mulai Prediksi'):
        st.subheader('Model yang Dipilih: {}'.format(selected_model))
        y_pred = build_train_predict_model(selected_model, X_train_norm, X_test_norm, y_train)
        st.write('Hasil Prediksi:', y_pred)

    # Evaluasi kinerja model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Tampilkan hasil evaluasi kinerja model
    st.subheader('Hasil Evaluasi Kinerja Model')
    st.write(f'**Akurasi:** {accuracy}')
    st.write(f'**Presisi:** {precision}')
    st.write(f'**Recall:** {recall}')
    st.write(f'**F1-score:** {f1}')
