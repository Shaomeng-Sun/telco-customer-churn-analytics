-- ============================================================================
-- QUERY 1: Contract & Payment Method Segmentation (Revenue Exposure Analysis)
-- Purpose: Identify billing segments with higher observed churn
-- and quantify their associated snapshot monthly charges.
-- ============================================================================
SELECT 
    cb.contract_type,
    cb.payment_method,
    COUNT(f.customer_id) AS total_customers,
    SUM(f.churn_flag) AS churned_customers,
    ROUND(CAST(SUM(f.churn_flag) AS REAL) / COUNT(f.customer_id) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(f.monthly_charges), 2) AS avg_monthly_arpu,
    ROUND(SUM(CASE WHEN f.churn_flag = 1 THEN f.monthly_charges ELSE 0 END), 2) AS churn_associated_mrr,
    ROUND(SUM(CASE WHEN f.churn_flag = 1 THEN f.monthly_charges ELSE 0 END) * 12, 2) AS annualized_churn_associated_mrr
FROM 
    fact_customer_churn f
JOIN 
    dim_contract_billing cb ON f.contract_id = cb.contract_id
GROUP BY 
    cb.contract_type, 
    cb.payment_method
ORDER BY 
    churn_associated_mrr DESC;