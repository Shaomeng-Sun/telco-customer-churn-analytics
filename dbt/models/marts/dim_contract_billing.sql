WITH distinct_contracts AS (
    SELECT DISTINCT
        contract_type,
        payment_method,
        paperless_billing
    FROM {{ ref('stg_telco_customers') }}
)

SELECT
    ROW_NUMBER() OVER (ORDER BY contract_type, payment_method, paperless_billing) AS contract_id,
    contract_type,
    payment_method,
    paperless_billing
FROM distinct_contracts