-- ============================================================================
-- QUERY 3: Product Ecosystem & Support Attachment Analysis
-- Purpose: Compare observed churn rates across DSL and Fiber Optic customers,
-- segmented by online security and technical support subscriptions.
-- ============================================================================
SELECT 
    s.internet_service,
    s.online_security,
    s.tech_support,
    COUNT(f.customer_id) AS customer_count,
    SUM(f.churn_flag) AS churned_count,
    ROUND(CAST(SUM(f.churn_flag) AS REAL) / COUNT(f.customer_id) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(f.monthly_charges), 2) AS avg_arpu
FROM 
    fact_customer_churn f
JOIN 
    dim_services s ON f.service_id = s.service_id
WHERE 
    s.internet_service != 'No'
GROUP BY 
    s.internet_service,
    s.online_security,
    s.tech_support
ORDER BY 
    s.internet_service, 
    churn_rate_pct DESC;