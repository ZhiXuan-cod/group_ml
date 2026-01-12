import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# --- SETUP & CONFIG ---
st.set_page_config(page_title="Poverty Analytics Malaysia", layout="wide")

# --- DATA AUGMENTATION (From Notebook) ---
def regression_augmentation(df, n_copies=5, noise_level=0.02):
    augmented_df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for _ in range(n_copies):
        noise = df[numeric_cols] * np.random.normal(0, noise_level, size=(len(df), len(numeric_cols)))
        temp_df = df.copy()
        temp_df[numeric_cols] = temp_df[numeric_cols] + noise
        augmented_df = pd.concat([augmented_df, temp_df], axis=0)
    return augmented_df.reset_index(drop=True)

@st.cache_data
def load_and_prep_data():
    # Load your dataset
    df_raw = pd.read_csv('MLGroupAssignment.csv')
    features = ['state', 'year', 'crime_assault', 'crime_property', 
                'unemployment_rate', 'completion_primary_both', 
                'completion_secondary_lower_both', 'completion_secondary_upper_both']
    target = 'poverty_absolute'
    df = df_raw[features + [target]].dropna()
    df_aug = regression_augmentation(df)
    return df, df_aug, features, target

df_orig, df_aug, features, target = load_and_prep_data()

# --- TUNED MODEL TRAINING ---
@st.cache_resource
def train_tuned_models(data, features, target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = [f for f in features if f != 'state']
    categorical_features = ['state']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Tuned Hyperparameters from your project notebook
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree (Tuned)": DecisionTreeRegressor(max_depth=10, min_samples_leaf=4, min_samples_split=10, random_state=42),
        "Random Forest (Tuned)": RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2, min_samples_split=5, random_state=42),
        "Gradient Boosting (Tuned)": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }

    results = {}
    trained_pipes = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            "R2": r2_score(y_test, preds),
            "MAE": mean_absolute_error(y_test, preds),
            "MSE": mean_squared_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "Actual": y_test.values,
            "Predicted": preds
        }
        trained_pipes[name] = pipe
    return trained_pipes, results

models_dict, results_dict = train_tuned_models(df_aug, features, target)

# --- NAVIGATION ---
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Jump to:", ["🏠 Project Overview", "📊 Comparative EDA", "🔮 Poverty Predictor", "🧪 Evaluation Dashboard"])

# --- PAGE 1: OVERVIEW ---
if page == "🏠 Project Overview":
    st.title("📊 Socio-Economic Poverty Analytics: Malaysia")
    
    st.markdown("""
    ### Project Background
    This dashboard provides a comprehensive analysis of **Absolute Poverty** levels across Malaysian states. 
    By leveraging Machine Learning, we identify how socio-economic factors like crime rates, education levels, and 
    unemployment influence poverty outcomes.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total States", len(df_orig['state'].unique()))
    col2.metric("Raw Dataset Size", len(df_orig))
    col3.metric("Augmented Dataset", len(df_aug))
    col4.metric("Input Features", len(features))

    st.subheader("Dataset Preview")
    st.dataframe(df_orig.style.background_gradient(cmap='Blues', subset=[target]))
    
    st.info("💡 **Tip:** Use the 'Comparative EDA' page to see how data augmentation improved the dataset distribution.")

# --- PAGE 2: COMPARATIVE EDA ---
elif page == "📊 Comparative EDA":
    st.title("🔍 Comparative Exploratory Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Correlation Analysis", "State-wise Comparisons", "Variable Distributions"])

    with tab1:
        st.subheader("Annotated Correlation Heatmaps")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Original Heatmap**")
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(df_orig.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        with c2:
            st.write("**Augmented Heatmap**")
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(df_aug.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    with tab2:
        st.subheader("State-wise Poverty & Box-Cox Transformation")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Average Poverty Level by State**")
            state_comparison = df_orig.groupby('state')[target].mean().sort_values(ascending=False).reset_index()
            fig, ax = plt.subplots()
            sns.barplot(data=state_comparison, x=target, y='state', palette='viridis', ax=ax)
            ax.set_title("Original Mean Poverty by State")
            st.pyplot(fig)
        with c2:
            st.write("**Target Distribution (Box-Cox)**")
            # Apply box-cox to handle skewness
            data_transformed, _ = stats.boxcox(df_orig[target] + 1)
            fig, ax = plt.subplots()
            sns.histplot(data_transformed, kde=True, color='purple', ax=ax)
            ax.set_title("Poverty Absolute (Transformed)")
            st.pyplot(fig)

    with tab3:
        st.subheader("Numerical Variable Comparison")
        selected_var = st.selectbox("Select variable to compare distributions:", [f for f in features if f != 'state'])
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Original Distribution: {selected_var}**")
            fig, ax = plt.subplots()
            sns.histplot(df_orig[selected_var], kde=True, color='skyblue', ax=ax)
            st.pyplot(fig)
        with c2:
            st.write(f"**Augmented Distribution: {selected_var}**")
            fig, ax = plt.subplots()
            sns.histplot(df_aug[selected_var], kde=True, color='salmon', ax=ax)
            st.pyplot(fig)

# --- PAGE 3: PREDICTOR ---
elif page == "🔮 Poverty Predictor":
    st.title("🔮 Absolute Poverty Prediction Engine")
    
    st.success("🤖 **System Recommendation:** Use the **Random Forest (Tuned)** model for the most accurate results (R² ≈ 0.99).")
    
    selected_model = st.selectbox("Choose a Tuned Model Architecture:", list(models_dict.keys()), index=3)
    
    with st.form("prediction_form"):
        st.write("### Socio-Economic Inputs")
        c1, c2 = st.columns(2)
        with c1:
            st_state = st.selectbox("Malaysian State", df_orig['state'].unique())
            st_year = st.slider("Year", 2024, 2035, 2025)
            st_assault = st.number_input("Crime: Assault Cases", value=1500)
            st_prop = st.number_input("Crime: Property Cases", value=6000)
        with c2:
            st_unemp = st.number_input("Unemployment Rate (%)", value=3.2)
            st_pri = st.number_input("Primary Edu. Completion (%)", value=98.5)
            st_slo = st.number_input("Lower Sec. Completion (%)", value=97.5)
            st_sup = st.number_input("Upper Sec. Completion (%)", value=96.5)
        
        submit = st.form_submit_button("Predict Poverty Outcome")
        
    if submit:
        input_data = pd.DataFrame([[st_state, st_year, st_assault, st_prop, st_unemp, st_pri, st_slo, st_sup]], columns=features)
        prediction = models_dict[selected_model].predict(input_data)[0]
        st.info(f"Using the **{selected_model}**, the predicted Absolute Poverty is:")
        st.title(f"{prediction:.4f}%")

# --- PAGE 4: EVALUATION ---
elif page == "🧪 Evaluation Dashboard":
    st.title("🧪 Model Comparison & Evaluation Dashboard")
    
    # Summary Table
    metrics_df = pd.DataFrame({
        name: [res["R2"], res["RMSE"], res["MAE"], res["MSE"]] 
        for name, res in results_dict.items()
    }, index=["R² Score", "RMSE", "MAE", "MSE"]).T
    
    st.subheader("Key Performance Indicators (KPIs)")
    st.table(metrics_df.style.background_gradient(cmap='Greens', subset=['R² Score']).background_gradient(cmap='Reds', subset=['RMSE', 'MAE', 'MSE']))

    st.subheader("Error Analysis (Actual vs. Predicted)")
    cols = st.columns(2)
    for i, (name, res) in enumerate(results_dict.items()):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(res["Actual"], res["Predicted"], alpha=0.5, color='teal')
            ax.plot([res["Actual"].min(), res["Actual"].max()], [res["Actual"].min(), res["Actual"].max()], 'r--', lw=2)
            ax.set_title(f"Model: {name}")
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Model Predictions")
            st.pyplot(fig)
            
    st.info("Visual analysis shows that tuned tree-based models (Random Forest/Gradient Boosting) follow the ideal (red dashed line) almost perfectly.")