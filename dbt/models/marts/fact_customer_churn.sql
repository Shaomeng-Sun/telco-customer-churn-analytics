WITH stg AS (
    SELECT * FROM {{ ref('stg_telco_customers') }}
),
contracts AS (
    SELECT * FROM {{ ref('dim_contract_billing') }}
),
services AS (
    SELECT * FROM {{ ref('dim_services') }}
)

SELECT
    stg.customer_id,
    contracts.contract_id,
    services.service_id,
    stg.tenure_months,
    stg.monthly_charges,
    stg.total_charges,
    stg.churn_flag
FROM stg
LEFT JOIN contracts
    ON stg.contract_type = contracts.contract_type
    AND stg.payment_method = contracts.payment_method
    AND stg.paperless_billing = contracts.paperless_billing
LEFT JOIN services
    ON stg.phone_service = services.phone_service
    AND stg.multiple_lines = services.multiple_lines
    AND stg.internet_service = services.internet_service
    AND stg.online_security = services.online_security
    AND stg.online_backup = services.online_backup
    AND stg.device_protection = services.device_protection
    AND stg.tech_support = services.tech_support
    AND stg.streaming_tv = services.streaming_tv
    AND stg.streaming_movies = services.streaming_movies