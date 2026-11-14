-- One row per customer from the dbt Star Schema.
-- This dataset is for retrospective classification of the recorded Churn label.
-- It is NOT a dated point-in-time dataset for a future 30-day churn target.

SELECT
    f.customer_id,
    f.tenure_months,
    f.monthly_charges,
    f.total_charges,

    c.gender,
    c.is_senior_citizen,
    c.has_partner,
    c.has_dependents,

    cb.contract_type,
    cb.payment_method,
    cb.paperless_billing,

    s.phone_service,
    s.multiple_lines,
    s.internet_service,
    s.online_security,
    s.online_backup,
    s.device_protection,
    s.tech_support,
    s.streaming_tv,
    s.streaming_movies,

    f.churn_flag
FROM fact_customer_churn AS f
LEFT JOIN dim_customer AS c
    ON f.customer_id = c.customer_id
LEFT JOIN dim_contract_billing AS cb
    ON f.contract_id = cb.contract_id
LEFT JOIN dim_services AS s
    ON f.service_id = s.service_id
ORDER BY f.customer_id;
