import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set Streamlit config at the top
st.set_page_config(page_title="Toyota Corolla Price Predictor", layout="wide")

# ----------------------------
# Load and Prepare Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ToyotaCorolla (2).csv", encoding='latin1')
    cols = ['Price', 'Age_08_04', 'KM', 'HP', 'Fuel_Type', 'Automatic', 'Doors', 'Quarterly_Tax', 'Weight']
    df = df[cols].dropna()
    df = pd.get_dummies(df, columns=['Fuel_Type'], drop_first=True)
    return df

df = load_data()

# ----------------------------
# UI Header
# ----------------------------
st.title("🚗 Toyota Corolla Price Prediction App")
st.markdown("""
This app uses machine learning to estimate the **price of a used Toyota Corolla** in the Netherlands.  
Choose a model, explore predictions, and input your own car's details to see its estimated value.
""")

# ----------------------------
# Feature & Target Split
# ----------------------------
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Sidebar: Model Choice
# ----------------------------
st.sidebar.header("🔧 Choose Model")
model_type = st.sidebar.radio("Select a regression model:", ['Linear Regression', 'Random Forest'])

# Train selected model
if model_type == 'Linear Regression':
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------------------------
# Evaluation
# ----------------------------
st.subheader(f"📊 Model Evaluation — {model_type}")
col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
col2.metric("RMSE", f"{mean_squared_error(y_test, y_pred, squared=False):,.2f} €")

# ----------------------------
# Actual vs Predicted Plot
# ----------------------------
st.subheader("🔍 Actual vs Predicted Prices")
fig1, ax1 = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax1, alpha=0.7)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
ax1.set_xlabel("Actual Price (€)")
ax1.set_ylabel("Predicted Price (€)")
ax1.legend()
st.pyplot(fig1)

# ----------------------------
# Feature Importance
# ----------------------------
st.subheader("📌 Feature Importance")
if model_type == 'Random Forest':
    importance = pd.Series(model.feature_importances_, index=X.columns)
else:
    importance = pd.Series(model.coef_, index=X.columns)

fig2, ax2 = plt.subplots()
importance.sort_values().plot(kind='barh', ax=ax2)
ax2.set_title("Model Feature Impact")
st.pyplot(fig2)

# ----------------------------
# Sidebar: User Input for Prediction
# ----------------------------
st.sidebar.header("💸 Predict Your Corolla's Price")

user_input = {}
user_input['Age_08_04'] = st.sidebar.slider('Age (in months)', int(df['Age_08_04'].min()), int(df['Age_08_04'].max()), int(df['Age_08_04'].mean()))
user_input['KM'] = st.sidebar.number_input('Kilometers Driven', int(df['KM'].min()), int(df['KM'].max()), int(df['KM'].mean()))
user_input['HP'] = st.sidebar.slider('Horsepower (HP)', int(df['HP'].min()), int(df['HP'].max()), int(df['HP'].mean()))
user_input['Doors'] = st.sidebar.selectbox('Number of Doors', sorted(df['Doors'].unique()))
user_input['Quarterly_Tax'] = st.sidebar.slider('Quarterly Tax (€)', int(df['Quarterly_Tax'].min()), int(df['Quarterly_Tax'].max()), int(df['Quarterly_Tax'].mean()))
user_input['Weight'] = st.sidebar.slider('Car Weight (kg)', int(df['Weight'].min()), int(df['Weight'].max()), int(df['Weight'].mean()))

fuel = st.sidebar.radio('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
auto = st.sidebar.radio('Automatic Transmission?', ['Yes', 'No'])

# Encode categorical
user_input['Automatic'] = 1 if auto == 'Yes' else 0
user_input['Fuel_Type_Diesel'] = 1 if fuel == 'Diesel' else 0
user_input['Fuel_Type_Petrol'] = 1 if fuel == 'Petrol' else 0

# Create input DataFrame and align with training columns
input_df = pd.DataFrame([user_input])
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Predict and display
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.sidebar.success(f"Estimated Price: €{int(prediction):,}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("Made with ❤️ by Group 5  \nDataset: ToyotaCorolla.csv  \n© 2025")
