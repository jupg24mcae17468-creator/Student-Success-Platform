import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Student Success Platform", layout="wide", page_icon="🎓")

# --- 2. ROBUST DATA LOADING FUNCTION ---
# This function is cached, so it runs only once when the app starts.
@st.cache_data(ttl=60)
def load_data():
    file_path = 'usage-report.csv'
    try:
        # Load CSV, skipping bad lines if any exist
        df = pd.read_csv(file_path, on_bad_lines='skip')
        
        # --- DATA CLEANING & ERROR PROOFING ---
        
        # 1. Clean Column Names (remove extra spaces)
        df.columns = df.columns.str.strip()

        # 2. Ensure Essential Columns Exist (Prevent 'KeyError')
        required_cols = [
            'Job Type', 'Business Unit', 'Overall Progress', 
            'Course Grade', 'Estimated Learning Hours', 
            'Completed', 'Enrollment Time', 'Course', 'Name', 'Email'
        ]
        for col in required_cols:
            if col not in df.columns:
                # Create missing column with default values
                df[col] = 0 if col in ['Overall Progress', 'Course Grade'] else 'Unknown'

        # 3. Clean Categorical Data
        df['Job Type'] = df['Job Type'].astype(str).replace({'nan': 'Unknown', 'Msc': 'M.Sc', 'ARTIFICIAL INTELLIGENCE AND CYBER SECURITY': 'AI & Cyber Sec'})
        
        # 4. Clean Numerical Data (Handle blanks/text in number fields)
        cols_to_numeric = ['Overall Progress', 'Course Grade', 'Estimated Learning Hours']
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 5. Clean Date Data (standardize to tz-naive)
        df['Enrollment Time'] = pd.to_datetime(df['Enrollment Time'], errors='coerce', utc=True).dt.tz_localize(None)

        # 6. Create Target for AI (Convert 'Yes'/'No' to 1/0)
        df['Completed_Binary'] = df['Completed'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

        return df

    except FileNotFoundError:
        st.error("❌ CRITICAL ERROR: 'usage-report.csv' not found.")
        st.warning("👉 Please make sure the CSV file is in the EXACT same folder as this script.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
        return pd.DataFrame()

# Load Data
df = load_data()

# Stop execution if data failed to load
if df.empty:
    st.stop()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("🎓 Student Success Platform")
st.sidebar.info("Mini Project v1.0")
page = st.sidebar.radio("Navigate to Module:", ["📊 Dashboard", "🧠 AI Predictor", "📢 Alert System"])

# ==============================================================================
# MODULE 1: DASHBOARD
# ==============================================================================
if page == "📊 Dashboard":
    st.title("📊 Student Performance Dashboard")
    
    # Sidebar Filters
    st.sidebar.markdown("---")
    st.sidebar.header("Filter Options")
    
    # Dynamic Filters based on available data
    all_degrees = sorted(list(df['Job Type'].unique()))
    selected_degree = st.sidebar.multiselect("Select Degree", all_degrees, default=all_degrees)
    
    # Filter Logic
    if selected_degree:
        filtered_df = df[df['Job Type'].isin(selected_degree)]
    else:
        filtered_df = df

    if filtered_df.empty:
        st.warning("No data matches your filters.")
    else:
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", int(len(filtered_df)))
        col2.metric("Active Courses", filtered_df['Course'].nunique())
        col3.metric("Avg Progress", f"{filtered_df['Overall Progress'].mean():.1f}%")
        
        completion_rate = (filtered_df['Completed_Binary'].mean() * 100)
        col4.metric("Completion Rate", f"{completion_rate:.1f}%")

        st.markdown("---")

        # Visualization Area
        c1, c2 = st.columns((2,1))
        
        with c1:
            st.subheader("Top Courses by Enrollment")
            top_courses = filtered_df['Course'].value_counts().head(10).reset_index()
            top_courses.columns = ['Course', 'Enrollments']
            fig_bar = px.bar(top_courses, x='Enrollments', y='Course', orientation='h', color='Enrollments', color_continuous_scale='Blues')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, width='stretch')

        with c2:
            st.subheader("Status Distribution")
            status_counts = filtered_df['Completed'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig_pie = px.pie(status_counts, values='Count', names='Status', hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
            st.plotly_chart(fig_pie, width='stretch')

# ==============================================================================
# MODULE 2: AI PREDICTOR
# ==============================================================================
elif page == "🧠 AI Predictor":
    st.title("🧠 AI Completion Predictor")
    st.write("This module trains a Machine Learning model on your CSV data to predict student success.")

    # Data Check: We need both "Yes" and "No" in the data to train a model
    unique_outcomes = df['Completed_Binary'].unique()
    
    if len(unique_outcomes) < 2:
        st.warning("⚠️ Not enough data variety to train AI. (Need both 'Yes' and 'No' completions in the CSV).")
    else:
        # Train Model Section
        with st.spinner("Training Random Forest Model..."):
            try:
                feature_cols = ['Overall Progress', 'Course Grade', 'Estimated Learning Hours']
                X = df[feature_cols]
                y = df['Completed_Binary']

                # --- FIX: Filter out completely inactive records for better ML training ---
                # A record is considered *active enough* if progress > 0 OR hours > 0
                activity_mask = (X['Overall Progress'] > 0) | (X['Estimated Learning Hours'] > 0)
                
                X_valid = X[activity_mask]
                y_valid = y[activity_mask]
                # ------------------------------------------------------------------------

                if len(X_valid) < 50: # Minimum data required for a decent split
                    st.warning("⚠️ Insufficient data points for meaningful training after filtering. Displaying training parameters only.")
                    
                elif len(np.unique(y_valid)) < 2:
                    st.warning("⚠️ Only one type of outcome (Completed/Not Completed) in filtered data. Cannot train classification model.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42)
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    accuracy = model.score(X_test, y_test)
                    st.success(f"✅ AI Model Trained Successfully! Accuracy: **{accuracy*100:.1f}%**")

                    
                    
                    # User Input Section
                    st.markdown("---")
                    st.subheader("🔮 Predict Success for a New Student")
                    
                    col1, col2, col3 = st.columns(3)
                    input_progress = col1.slider("Current Progress (%)", 0, 100, 25)
                    input_grade = col2.slider("Current Grade (Avg)", 0, 100, 60)
                    input_hours = col3.number_input("Hours Spent Learning", min_value=0, max_value=500, value=5)

                    if st.button("Run Prediction"):
                        input_data = pd.DataFrame([[input_progress, input_grade, input_hours]], columns=feature_cols)
                        prediction = model.predict(input_data)[0]
                        probs = model.predict_proba(input_data)[0]
                        
                        # Display Result
                        if prediction == 1:
                            st.balloons()
                            st.success(f"🎉 Result: LIKELY TO COMPLETE (Confidence: {probs[1]*100:.1f}%)")
                        else:
                            st.error(f"⚠️ Result: AT RISK OF DROPPING OUT (Risk: {probs[0]*100:.1f}%)")

            except Exception as e:
                st.error(f"An error occurred during model training/testing: {e}")
                st.stop()


# ==============================================================================
# MODULE 3: ALERT SYSTEM
# ==============================================================================
elif page == "📢 Alert System":
    st.title("📢 Student Alert System")
    st.markdown("Identify students who have been enrolled for a long time but have low progress, for proactive outreach.")

    # Input Parameters
    col1, col2 = st.columns(2)
    days_threshold = col1.number_input("Days Since Enrollment >", value=30)
    progress_threshold = col2.number_input("Overall Progress < (%)", value=10)

    # Logic
    now = pd.Timestamp.now()

    # Filter out rows where date is missing to prevent errors
    valid_df = df.dropna(subset=['Enrollment Time']).copy()

    # Calculate days elapsed
    valid_df['Days_Since_Enrollment'] = (now - valid_df['Enrollment Time']).dt.days

    # Apply Logic
    at_risk_students = valid_df[
        (valid_df['Days_Since_Enrollment'] > days_threshold) &
        (valid_df['Overall Progress'] < progress_threshold) &
        (valid_df['Completed_Binary'] == 0)
    ]

    st.info(f"Found **{len(at_risk_students)}** students matching these criteria.")

    if not at_risk_students.empty:
        st.subheader("At-Risk Student List")
        st.dataframe(at_risk_students[['Name', 'Email', 'Course', 'Days_Since_Enrollment', 'Overall Progress']])

        # Email Generator: show first 5, with option to see all
        if st.button("✉️ Generate Email Drafts"):
            st.markdown("### 📝 Email Drafts")
            # First 5 drafts (immediate)
            first_n = at_risk_students.head(5)
            for n, (_, row) in enumerate(first_n.iterrows(), start=1):
                email_content = f"""
                **To:** {row['Email']}
                **Subject:** Checking in on your {row['Course']} course
                
                Hi {row['Name']},
                
                We noticed you enrolled in "{row['Course']}" {int(row['Days_Since_Enrollment'])} days ago, 
                but your progress is currently at {row['Overall Progress']}%.
                
                Is there anything we can do to help you move forward?
                
                Best regards,
                Course Admin
                """
                st.text_area(f"Draft #{n}", email_content, height=200)

            # Remaining drafts inside an expander (view all)
            remaining = at_risk_students.iloc[5:]
            if not remaining.empty:
                with st.expander(f"Show more (view remaining {len(remaining)} drafts)"):
                    for m, (_, row) in enumerate(remaining.iterrows(), start=6):
                        email_content = f"""
                        **To:** {row['Email']}
                        **Subject:** Checking in on your {row['Course']} course
                        
                        Hi {row['Name']},
                        
                        We noticed you enrolled in "{row['Course']}" {int(row['Days_Since_Enrollment'])} days ago, 
                        but your progress is currently at {row['Overall Progress']}%.
                        
                        Is there anything we can do to help you move forward?
                        
                        Best regards,
                        Course Admin
                        """
                        st.text_area(f"Draft #{m}", email_content, height=200)