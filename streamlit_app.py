import streamlit as st
from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳", layout="wide")

st.title("💳 Credit Risk Prediction")
st.caption("Snowflake ML Credit Risk Prediction")

DATABASE = "CREDIT_RISK_MLOPS"
MODEL = f"{DATABASE}.ML.CREDIT_RISK_BASELINE"

@st.cache_resource
def get_session():
    return get_active_session()

try:
    session = get_session()
except Exception as e:
    st.error("This app must run as a Streamlit in Snowflake app.")
    st.exception(e)
    st.stop()

st.subheader("Loan Applicant Details")

c1, c2, c3 = st.columns(3)

with c1:
    person_age = st.number_input("Person Age", 18, 100, 30)
    person_income = st.number_input("Person Income", min_value=0.0, value=50000.0, step=1000.0)
    person_home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "OWN", "RENT", "OTHER"])
    person_emp_length = st.number_input("Employment Length", min_value=0.0, max_value=50.0, value=5.0)

with c2:
    loan_intent = st.selectbox(
        "Loan Intent",
        ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"],
    )
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0, step=500.0)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

with c3:
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=10.0, value=0.20, step=0.01)
    cb_person_default_on_file = st.selectbox("Previous Default on File", ["N", "Y"])
    cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0.0, max_value=100.0, value=10.0)

if st.button("Predict Credit Risk", type="primary"):
    if person_income <= 0:
        st.error("Person Income must be greater than 0.")
        st.stop()
    if loan_amnt <= 0:
        st.error("Loan Amount must be greater than 0.")
        st.stop()

    loan_to_income = loan_amnt / person_income
    income_per_loan = person_income / loan_amnt
    credit_history_to_age = cb_person_cred_hist_length / person_age if person_age else None
    employment_to_age = person_emp_length / person_age if person_age else None
    has_previous_default = 1 if cb_person_default_on_file == "Y" else 0

    home_mortgage = 1 if person_home_ownership == "MORTGAGE" else 0
    home_own = 1 if person_home_ownership == "OWN" else 0
    home_rent = 1 if person_home_ownership == "RENT" else 0
    home_other = 1 if person_home_ownership == "OTHER" else 0

    intent_debtconsolidation = 1 if loan_intent == "DEBTCONSOLIDATION" else 0
    intent_education = 1 if loan_intent == "EDUCATION" else 0
    intent_homeimprovement = 1 if loan_intent == "HOMEIMPROVEMENT" else 0
    intent_medical = 1 if loan_intent == "MEDICAL" else 0
    intent_personal = 1 if loan_intent == "PERSONAL" else 0
    intent_venture = 1 if loan_intent == "VENTURE" else 0

    grade_a = 1 if loan_grade == "A" else 0
    grade_b = 1 if loan_grade == "B" else 0
    grade_c = 1 if loan_grade == "C" else 0
    grade_d = 1 if loan_grade == "D" else 0
    grade_e = 1 if loan_grade == "E" else 0
    grade_f = 1 if loan_grade == "F" else 0
    grade_g = 1 if loan_grade == "G" else 0

    def sql_num(value):
        return "NULL" if value is None else str(float(value))

    prediction_sql = f"""
    WITH scored AS (
        SELECT {MODEL}!PREDICT(
            INPUT_DATA => OBJECT_CONSTRUCT(
                'PERSON_AGE', {person_age},
                'PERSON_INCOME', {person_income},
                'PERSON_EMP_LENGTH', {person_emp_length},
                'LOAN_AMNT', {loan_amnt},
                'LOAN_INT_RATE', {loan_int_rate},
                'LOAN_PERCENT_INCOME', {loan_percent_income},
                'CB_PERSON_CRED_HIST_LENGTH', {cb_person_cred_hist_length},
                'LOAN_TO_INCOME', {loan_to_income},
                'INCOME_PER_LOAN', {income_per_loan},
                'CREDIT_HISTORY_TO_AGE', {sql_num(credit_history_to_age)},
                'EMPLOYMENT_TO_AGE', {sql_num(employment_to_age)},
                'HAS_PREVIOUS_DEFAULT', {has_previous_default},
                'HOME_MORTGAGE', {home_mortgage},
                'HOME_OWN', {home_own},
                'HOME_RENT', {home_rent},
                'HOME_OTHER', {home_other},
                'INTENT_DEBTCONSOLIDATION', {intent_debtconsolidation},
                'INTENT_EDUCATION', {intent_education},
                'INTENT_HOMEIMPROVEMENT', {intent_homeimprovement},
                'INTENT_MEDICAL', {intent_medical},
                'INTENT_PERSONAL', {intent_personal},
                'INTENT_VENTURE', {intent_venture},
                'GRADE_A', {grade_a},
                'GRADE_B', {grade_b},
                'GRADE_C', {grade_c},
                'GRADE_D', {grade_d},
                'GRADE_E', {grade_e},
                'GRADE_F', {grade_f},
                'GRADE_G', {grade_g}
            )
        ) AS PREDICTION
    )
    SELECT
        PREDICTION:class::STRING AS PREDICTED_CLASS,
        TRY_TO_DOUBLE(PREDICTION:probability:"1"::STRING) AS PD
    FROM scored
    """

    try:
        result = session.sql(prediction_sql).collect()
        if not result:
            st.error("No prediction returned.")
            st.stop()

        predicted_class = str(result[0]["PREDICTED_CLASS"])
        pd_value = float(result[0]["PD"])

        if pd_value < 0.10:
            risk_band = "LOW"
        elif pd_value < 0.30:
            risk_band = "MEDIUM"
        elif pd_value < 0.50:
            risk_band = "HIGH"
        else:
            risk_band = "VERY_HIGH"

        st.divider()
        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns(3)
        r1.metric("Probability of Default", f"{pd_value:.2%}")
        r2.metric("Prediction", "DEFAULT" if predicted_class == "1" else "NO DEFAULT")
        r3.metric("Risk Band", risk_band)

        st.success("Prediction generated successfully.")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

with st.expander("Project Information"):
    st.write(
        """
        **ML Problem:** Binary Classification

        **Dataset:** credit_risk_dataset.csv

        **Model:** CREDIT_RISK_BASELINE

        **Model Type:** SNOWFLAKE.ML.CLASSIFICATION
        """
    )
