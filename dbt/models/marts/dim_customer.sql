SELECT
    customer_id,
    gender,
    is_senior_citizen,
    has_partner,
    has_dependents
FROM {{ ref('stg_telco_customers') }}