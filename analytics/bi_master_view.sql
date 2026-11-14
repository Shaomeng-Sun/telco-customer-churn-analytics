/*
Purpose:
    Create a denormalized customer-level dataset for Tableau or Power BI.

Grain:
    One row per customer in fact_customer_churn.

Implementation:
    LEFT JOINs preserve fact rows if a dimension lookup is missing. The Python
    export script validates that the result has the same row count as the fact
    table and contains one row per customer.

Privacy:
    This is a customer-level export. Keep it local and do not commit the
    generated CSV to a public GitHub repository.
*/

SELECT
    f.customer_id,
    f.tenure_months,
    f.monthly_charges,
    f.total_charges,
    f.churn_flag,

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
    s.streaming_movies

FROM fact_customer_churn AS f
LEFT JOIN dim_customer AS c
    ON f.customer_id = c.customer_id
LEFT JOIN dim_contract_billing AS cb
    ON f.contract_id = cb.contract_id
LEFT JOIN dim_services AS s
    ON f.service_id = s.service_id;
