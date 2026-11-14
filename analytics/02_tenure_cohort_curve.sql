-- ============================================================================
-- QUERY 2: Customer Tenure Band Analysis
-- Purpose: Compare observed churn rates across customer
-- tenure bands in the supplied customer snapshot.
-- ============================================================================
WITH tenure_cohorts AS (
    SELECT 
        customer_id,
        tenure_months,
        monthly_charges,
        churn_flag,
        CASE 
            WHEN tenure_months <= 6 THEN '1. 0-6 Months (Onboarding)'
            WHEN tenure_months <= 12 THEN '2. 7-12 Months (First Year)'
            WHEN tenure_months <= 24 THEN '3. 13-24 Months (Year Two)'
            WHEN tenure_months <= 48 THEN '4. 25-48 Months (Established)'
            ELSE '5. 49+ Months (Loyalists)'
        END AS tenure_cohort
    FROM 
        fact_customer_churn
)
SELECT 
    tenure_cohort,
    COUNT(customer_id) AS cohort_size,
    SUM(churn_flag) AS churned_users,
    ROUND(CAST(SUM(churn_flag) AS REAL) / COUNT(customer_id) * 100, 2) AS cohort_churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charge,
    ROUND(SUM(CASE WHEN churn_flag = 1 THEN monthly_charges ELSE 0 END), 2) AS cohort_churn_associated_mrr
FROM 
    tenure_cohorts
GROUP BY 
    tenure_cohort
ORDER BY 
    tenure_cohort ASC;