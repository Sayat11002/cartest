import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import os
from datetime import datetime
import subprocess

# === Page Config ===
st.set_page_config(
    page_title="🚘 Car Assistant",
    page_icon="🧠",
    layout="wide"
)

# === THEME SWITCHER ===
theme = st.radio("🌗 Choose Theme", ["Light mode", "Dark mode"], horizontal=True)

if theme == "Light mode":
    background_color = "#f5f0e6"
    text_color = "#1f1f1f"
    button_color = "#e0dbd1"
    hover_color = "#d1cfc7"
else:
    background_color = "#1e1e1e"
    text_color = "#f0f0f0"
    button_color = "#333333"
    hover_color = "#444444"

# === Dynamic CSS based on theme ===
st.markdown(f"""
    <style>
        html, body, [class*="css"], .main, .block-container {{
            background-color: {background_color} !important;
            color: {text_color} !important;
            font-size: 18px !important;
            font-weight: 500 !important;
        }}
        h1, h2, h3, h4, h5, h6, p, span, label, div {{
            color: {text_color} !important;
            font-weight: 600 !important;
        }}
        .stButton > button {{
            background-color: {button_color};
            color: {text_color};
            border: 1px solid #888;
            padding: 0.5rem 1rem;
            border-radius: 12px;
            transition: all 0.2s ease-in-out;
            font-size: 16px;
            font-weight: 600;
        }}
        .stButton > button:hover {{
            background-color: {hover_color};
            transform: scale(1.05);
        }}
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {hover_color};
            border-radius: 12px;
        }}
        .stTabs [data-baseweb="tab"] {{
            font-weight: bold;
            padding: 0.5rem 1.2rem;
            color: {text_color} !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: {button_color};
            border-radius: 10px;
        }}
        input, textarea, select {{
            background-color: {button_color} !important;
            color: {text_color} !important;
            border: 1px solid #aaa !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            font-weight: 500 !important;
        }}
        ::placeholder {{
            color: {text_color}99 !important;
            font-size: 15px !important;
        }}
    </style>
""", unsafe_allow_html=True)

# === Car Description Matching Functions ===
DATA_PATH = "car_data_full.json"
FEEDBACK_PATH = "user_feedback.json"
ADMIN_PASSWORD = "admin123"

def git_commit_and_push(message="Auto update"):
    try:
        subprocess.run(["git", "add", "."], cwd="car-config-sync", check=True)
        subprocess.run(["git", "commit", "-m", message], cwd="car-config-sync", check=True)
        subprocess.run(["git", "push"], cwd="car-config-sync", check=True)
    except Exception as e:
        print("Git push failed:", e)

def load_data():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {
            "Кузов": {
                'пикап': "Пикап — это мощный автомобиль с открытым грузовым отсеком. Идеален для перевозки строительных материалов, сельскохозяйственной продукции и крупногабаритных грузов. Часто используется фермерами, строителями и любителями активного отдыха. Обладает высоким клиренсом, возможностью буксировки тяжелых прицепов и обычно оснащен полным приводом. Лучший выбор для бездорожья, тяжелых условий эксплуатации и работы в сельской местности.",
                'минивэн': "Минивэн — это просторный автомобиль, предназначенный для перевозки больших семей или групп людей. Оснащен тремя рядами сидений, комфортным салоном и большим багажным отделением. Идеален для дальних поездок, семейных путешествий и регулярных поездок в школу или на мероприятия. Главные плюсы: вместительность, высокий уровень комфорта, множество систем безопасности.",
                'кроссовер': "Кроссовер — это универсальный автомобиль, сочетающий черты внедорожника и легкового авто. Отличается высоким клиренсом, удобной посадкой, вместительным салоном и современными технологиями. Подходит как для городской езды, так и для поездок по неасфальтированным дорогам. Идеален для людей, ведущих активный образ жизни, семей с детьми, любителей путешествий.",
                'седан': "Седан — классический тип автомобиля с отдельным багажником и комфортным салоном. Лучший выбор для городской эксплуатации и дальних поездок по хорошим дорогам. Отличается высоким уровнем шумоизоляции, удобством посадки и устойчивостью на трассе. Плюсы: комфорт, управляемость, презентабельный внешний вид.",
                'хэтчбек': "Хэтчбек — компактный автомобиль с укороченной задней частью и поднимающейся вверх дверью багажника. Идеален для городской жизни: удобен в парковке, экономичен в расходе топлива, маневренен в пробках. Часто выбирается молодыми людьми, студентами, городскими жителями, которым важны мобильность и экономия.",
                'универсал': "Универсал — автомобиль с удлиненным кузовом и увеличенным багажным отделением. Оптимален для семейных поездок, перевозки большого количества багажа, домашних животных или спортивного инвентаря. Сочетает удобство седана с вместимостью кроссовера. Выбор для тех, кто ценит практичность и комфорт в путешествиях.",
                'внедорожник': "Внедорожник — крупный автомобиль с высокой проходимостью, усиленной подвеской и полным приводом. Способен преодолевать тяжелые дорожные условия: грязь, снег, камни, броды. Идеальный выбор для охотников, рыболовов, любителей экстремального отдыха, а также для проживания в сельской местности. Плюсы: мощность, безопасность, проходимость."
            },
            "Трансмиссия": {
                'автомат': "Автоматическая коробка передач обеспечивает плавную езду без необходимости вручную переключать передачи. Идеально подходит для городского движения, пробок и начинающих водителей. Повышает комфорт в повседневной эксплуатации, особенно в мегаполисах.",
                'механика': "Механическая коробка передач дает полный контроль над автомобилем. Любима опытными водителями и автолюбителями за возможность динамичного разгона и экономию топлива. Требует большего внимания и навыков вождения, особенно в пробках."
            },
            "Топливо": {
                'бензин': "Бензиновые двигатели обеспечивают хорошую динамику разгона, универсальность и тишину работы. Идеальны для городской езды и умеренных пробегов. Шире доступны и дешевле в ремонте по сравнению с дизельными и электрическими.",
                'дизель': "Дизельные двигатели отличаются высокой экономичностью на дальних расстояниях и повышенным крутящим моментом. Идеальны для тех, кто много ездит по трассам, перевозит грузы или живет в сельской местности. Дизельные автомобили имеют больший ресурс двигателя, но требуют регулярного обслуживания.",
                'электро': "Электромобили полностью работают на электричестве, не выбрасывая вредных веществ. Отличаются очень низкими эксплуатационными расходами и тишиной хода. Подходят для жителей городов с развитой инфраструктурой зарядных станций.",
                'гибрид': "Гибридные автомобили совмещают бензиновый двигатель и электромотор. Обеспечивают экономичность в городе, низкий уровень выбросов и комфорт. Подходят для тех, кто хочет снизить расход топлива без перехода на чистую электроэнергию."
            }
        }

def save_data(data):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    git_commit_and_push("Updated car config")

def save_feedback(entry):
    feedback = []
    feedback_dir = os.path.dirname(FEEDBACK_PATH)
    os.makedirs(feedback_dir, exist_ok=True)  # Создаём папку, если её нет

    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            feedback = json.load(f)

    feedback.append(entry)

    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)

    git_commit_and_push("New feedback entry")

def delete_feedback(index):
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            fb_data = json.load(f)
        if 0 <= index < len(fb_data):
            del fb_data[index]
            with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
                json.dump(fb_data, f, ensure_ascii=False, indent=2)
            git_commit_and_push("Feedback deleted")

category_blocks = load_data()

def prepare_tfidf_data(keywords_dict):
    labels = list(keywords_dict.keys())
    texts = [keywords_dict[k] for k in labels]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return labels, matrix, vectorizer

def find_best_match(query, labels, matrix, vectorizer, top_k=1):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    return [(labels[i], round(similarities[i], 3)) for i in top_indices]

def semantic_search_grouped(query, top_k=1):
    grouped_results = {}
    for cat_name, data in category_blocks.items():
        labels, matrix, vectorizer = prepare_tfidf_data(data)
        matches = find_best_match(query, labels, matrix, vectorizer, top_k)
        grouped_results[cat_name] = matches
    return grouped_results

def add_to_category(category, class_label, query):
    if class_label in category_blocks[category]:
        category_blocks[category][class_label] += f" {query}."
    else:
        category_blocks[category][class_label] = query
    save_data(category_blocks)

# === Price Estimation Functions ===
@st.cache_data
def load_price_data():
    car_df = pd.read_csv("22613data.csv")
    car_df = car_df.drop(['City', 'Volume'], axis=1)
    return car_df

raw_data = load_price_data()

categorical_cols = ['Company', 'Mark', 'Fuel Type', 'Transmission', 'Car_type']

def remove_outliers(data, column):
    Q1, Q3 = data[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

@st.cache_data
def preprocess_data(data):
    df = data.drop_duplicates()
    df.fillna({'Mark': 'Unknown', 'Fuel Type': 'Unknown', 'Transmission': 'Unknown'}, inplace=True)
    df['Year'] = df['Year'].fillna(df['Year'].median()).astype(int)
    df['Mileage'] = df['Mileage'].fillna(df['Mileage'].median())

    df = remove_outliers(df, 'Price')
    df = remove_outliers(df, 'Mileage')

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).str.upper())
        encoders[col] = le

    return df, encoders

df, encoders = preprocess_data(raw_data)

@st.cache_resource
def train_model(df):
    X = df.drop('Price', axis=1)
    y = df['Price']
    model = RandomForestRegressor(n_estimators=350, random_state=4)
    model.fit(X, y)
    return model

model = train_model(df)

# === Interface ===
st.title("🚘 Car Assistant")

tabs = st.tabs(["🔍 Match by Description", "💰 Estimate Price", "📆 Credit Calc"])

# === Tab 1: Match by Description ===
with tabs[0]:
    st.markdown("### 🧾 Опишите автомобиль своей мечты и позвольте нам порекомендовать вам тип топлива, трансмиссию и тип кузова:")
    
    mode = st.radio("Выберите режим:", ["Пользователь", "Админ"], horizontal=True, key="mode_radio")
    if mode == "Админ":
        password = st.text_input("Введите пароль:", type="password", key="admin_pass")
        is_admin = password == ADMIN_PASSWORD
    else:
        is_admin = False

    if not is_admin:
        query = st.text_area("💬 Ваш запрос:", key="user_query")
        if st.button("✨ Find Best Match", key="desc_button"):
            if query.strip() == "":
                st.warning("🚨 Please enter a description.")
            else:
                results = semantic_search_grouped(query, top_k=1)
                feedback = {"query": query, "timestamp": str(datetime.now()), "results": {}, "rating": None, "comment": ""}

                st.markdown("### ✅ Suggested Specs:")
                for cat, matches in results.items():
                    best_label = matches[0][0] if matches else "Не найдено"
                    st.markdown(f"**{cat}:** {best_label}")
                    feedback["results"][cat] = matches  # сохраняем с score для админа

                st.subheader("📝 Оцените результат")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Всё подошло", key="like_btn"):
                        feedback["rating"] = "like"
                        save_feedback(feedback)
                        st.success("Спасибо за положительную оценку!")
                with col2:
                    if st.button("👎 Не подошло", key="dislike_btn"):
                        feedback["rating"] = "dislike"
                        feedback["comment"] = st.text_input("Комментарий (по желанию)", key="comment_input")
                        save_feedback(feedback)
                        st.warning("Ваш отзыв сохранён. Спасибо!")
    else:
        st.subheader("📋 Отчёты от пользователей")
        if os.path.exists(FEEDBACK_PATH):
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                fb_data = json.load(f)

            sort_option = st.selectbox("Сортировка отзывов:", ["Все", "Только лайки", "Только дизлайки"], key="sort_feedback")

            filtered_fb = fb_data
            if sort_option == "Только лайки":
                filtered_fb = [x for x in fb_data if x.get("rating") == "like"]
            elif sort_option == "Только дизлайки":
                filtered_fb = [x for x in fb_data if x.get("rating") == "dislike"]

            for i, entry in enumerate(reversed(filtered_fb)):
                feedback_index = len(fb_data) - 1 - i
                short = entry['query'][:40] + '...' if len(entry['query']) > 40 else entry['query']
                with st.expander(f"#{feedback_index + 1} — {short}"):
                    st.markdown(f"**Полный запрос:** {entry['query']}")
                    st.markdown(f"**Оценка:** {entry['rating']}")
                    st.markdown(f"**Дата:** {entry['timestamp']}")
                    for cat, matches in entry["results"].items():
                        st.markdown(f"**{cat}:**")
                        for label, score in matches:
                            st.write(f"- {label}: {score}")
                    if entry.get("comment"):
                        st.markdown(f"**Комментарий:** {entry['comment']}")

                    st.markdown("---")
                    st.markdown("**➡️ Добавить этот запрос в категорию:**")
                    fb_cat = st.selectbox("Категория:", list(category_blocks.keys()), key=f"fbcat_{i}")
                    fb_cls = st.selectbox("Класс:", list(category_blocks[fb_cat].keys()), key=f"fbcls_{i}")

                    if st.button("📥 Добавить запрос", key=f"addfb_{i}"):
                        add_to_category(fb_cat, fb_cls, entry['query'])

                        if "corrections" not in entry:
                            entry["corrections"] = []
                        entry["corrections"].append(f"Добавлено в {fb_cat} → {fb_cls}")
                        save_feedback(entry)

                        st.success("Добавлено в описание!")

                    if entry.get("corrections"):
                        st.markdown("**🛠 Корректировки:**")
                        for c in entry["corrections"]:
                            st.write("-", c)

                    if st.button("🗑 Удалить этот отзыв", key=f"delfb_{i}"):
                        delete_feedback(feedback_index)
                        st.warning("Отзыв удалён. Обновите страницу.")

        else:
            st.info("Пока нет отзывов от пользователей.")

        st.markdown("---")
        st.subheader("✍️ Ручная корректировка описаний")
        category = st.selectbox("Выберите категорию:", list(category_blocks.keys()), key="edit_category")
        label = st.selectbox("Выберите класс:", list(category_blocks[category].keys()), key="edit_label")
        new_text = st.text_area("Изменить описание:", category_blocks[category][label], height=200, key="edit_text")

        if st.button("💾 Сохранить изменения", key="save_edit"):
            category_blocks[category][label] = new_text
            save_data(category_blocks)
            st.success("Описание обновлено и сохранено!")

# === Tab 2: Estimate Price ===
with tabs[1]:
    st.markdown("### 📊 Enter your car's features to get a price estimate:")

    company = st.selectbox("🏢 Manufacturer", sorted(raw_data['Company'].dropna().unique()), key="company_select")
    filtered_data = raw_data[raw_data['Company'] == company]
    mark = st.selectbox("🚘 Model", sorted(filtered_data['Mark'].dropna().unique()), key="model_select")
    year = st.number_input("📅 Year", 1990, 2025, 2015, key="year_input")
    fuel = st.selectbox("⛽ Fuel Type", sorted(raw_data['Fuel Type'].dropna().unique()), key="fuel_select")
    trans = st.selectbox("⚙️ Transmission", sorted(raw_data['Transmission'].dropna().unique()), key="trans_select")
    mileage = st.number_input("🛣️ Mileage (km)", 0, 1_000_000, 100_000, key="mileage_input")
    car_type = st.selectbox("🚗 Body Type", sorted(raw_data['Car_type'].dropna().unique()), key="type_select")

    if st.button("📈 Estimate Price", key="price_button"):
        new_car = pd.DataFrame({
            'Company': [company],
            'Mark': [mark],
            'Year': [year],
            'Fuel Type': [fuel],
            'Transmission': [trans],
            'Mileage': [mileage],
            'Car_type': [car_type]
        })

        try:
            for col in categorical_cols:
                new_car[col] = new_car[col].astype(str).str.upper()
                if any(v not in encoders[col].classes_ for v in new_car[col]):
                    raise ValueError(f"❌ Unknown value in column '{col}'")
                new_car[col] = encoders[col].transform(new_car[col])

            pred = model.predict(new_car)[0]
            st.success(f"💵 Estimated Price: **{int(pred):,} ₸**")
        except ValueError as e:
            st.error(str(e))

# === Tab 3: Credit Calculator ===
with tabs[2]:
    st.markdown("### 💳 Credit Calculator")

    car_price = st.number_input("Car Price (₸)", min_value=100000, value=1000000, step=10000, key="price_input")
    down_payment = st.number_input("Down Payment (₸)", min_value=0, max_value=car_price, value=int(car_price * 0.2), step=10000, key="down_payment")
    term = st.slider("Term (months)", 6, 84, 36, step=6, key="term_slider")
    rate = st.slider("Interest (%/yr)", 0.0, 100.0, 10.0, step=0.1, key="rate_slider")

    if car_price > down_payment:
        loan = car_price - down_payment
        monthly_rate = (rate / 100) / 12

        if rate > 0:
            m = monthly_rate
            monthly = loan * (m * (1 + m)**term) / ((1 + m)**term - 1)
        else:
            monthly = loan / term

        st.success(f"Monthly: **{int(monthly):,} ₸**")
    else:
        st.warning("Down payment >= price")
