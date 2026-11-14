WITH source AS (
    SELECT * FROM {{ source('raw_data', 'raw_telco_churn') }}
)

SELECT
    CAST(customerID AS TEXT) AS customer_id,
    CAST(gender AS TEXT) AS gender,
    CAST(SeniorCitizen AS INTEGER) AS is_senior_citizen,
    CAST(Partner AS TEXT) AS has_partner,
    CAST(Dependents AS TEXT) AS has_dependents,
    CAST(tenure AS INTEGER) AS tenure_months,
    CAST(PhoneService AS TEXT) AS phone_service,
    CAST(MultipleLines AS TEXT) AS multiple_lines,
    CAST(InternetService AS TEXT) AS internet_service,
    CAST(OnlineSecurity AS TEXT) AS online_security,
    CAST(OnlineBackup AS TEXT) AS online_backup,
    CAST(DeviceProtection AS TEXT) AS device_protection,
    CAST(TechSupport AS TEXT) AS tech_support,
    CAST(StreamingTV AS TEXT) AS streaming_tv,
    CAST(StreamingMovies AS TEXT) AS streaming_movies,
    CAST(Contract AS TEXT) AS contract_type,
    CAST(PaperlessBilling AS TEXT) AS paperless_billing,
    CAST(PaymentMethod AS TEXT) AS payment_method,
    CAST(MonthlyCharges AS REAL) AS monthly_charges,
    -- Fix Telco dataset quirk: blank spaces in TotalCharges remains NULL
    CASE 
        WHEN TRIM(TotalCharges) = '' OR TotalCharges IS NULL THEN NULL
        ELSE CAST(TotalCharges AS REAL)
    END AS total_charges,
    CASE 
        WHEN LOWER(TRIM(Churn)) = 'yes' THEN 1
        WHEN LOWER(TRIM(Churn)) = 'no' THEN 0
        ELSE NULL
    END AS churn_flag
FROM source