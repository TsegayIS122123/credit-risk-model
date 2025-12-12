# Credit Risk Probability Model for Alternative Data

## Project Overview
This project implements an end-to-end credit scoring system using alternative data from e-commerce transactions. The model transforms behavioral patterns into predictive risk signals for "Buy Now, Pay Later" services.

## Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability

**Basel II Influence:**
The Basel II Capital Accord emphasizes three pillars: Minimum Capital Requirements, Supervisory Review, and Market Discipline. For our model:

- **Pillar 1 (Minimum Capital Requirements):** Requires banks to hold capital proportional to their credit risk exposure. Our model directly informs this by quantifying risk probabilities.
- **Pillar 2 (Supervisory Review):** Regulators must ensure banks have robust internal processes. This necessitates:
  - **Explainability:** Clear documentation of how risk scores are derived
  - **Backtesting:** Regular validation of model predictions
  - **Stress testing:** Assessment under adverse conditions
- **Pillar 3 (Market Discipline):** Requires transparent disclosure, making interpretable models essential for stakeholder trust.

**Impact on Our Model:**
We need an interpretable model because:
- Regulatory compliance requires understanding of risk drivers
- Loan officers need to explain decisions to customers
- Model validation and auditing demand transparency
- Capital allocation decisions require justifiable risk assessments

### 2. Proxy Variable Necessity and Risks

**Why a Proxy is Necessary:**
1. **Data Limitation:** E-commerce platforms don't track loan repayments
2. **Behavioral Correlation:** Purchase patterns correlate with financial responsibility
3. **Innovation Opportunity:** Alternative data enables credit access for underserved populations

**Business Risks of Proxy-Based Predictions:**

| Risk Category | Description | Mitigation Strategy |
|--------------|-------------|-------------------|
| **Misalignment Risk** | Proxy may not perfectly correlate with actual default | - Validate with small pilot programs<br>- Monitor performance metrics closely<br>- Use conservative initial thresholds |
| **Regulatory Risk** | Regulators may question proxy validity | - Document rationale thoroughly<br>- Cite academic research on RFM-credit correlation<br>- Maintain audit trails |
| **Discrimination Risk** | Proxy may inadvertently discriminate | - Test for bias across protected classes<br>- Implement fairness constraints<br>- Regular bias audits |
| **Model Decay Risk** | Relationship between proxy and default may change | - Continuous monitoring<br>- Regular retraining<br>- Set up drift detection |

**Specific Proxy Approach:**
We'll use **RFM-based clustering** to identify "disengaged" customers as high-risk proxies, based on the hypothesis that consistent, high-value engagement indicates financial stability.

### 3. Model Selection Trade-offs

| Aspect | Simple Model (Logistic + WoE) | Complex Model (Gradient Boosting) |
|--------|-------------------------------|-----------------------------------|
| **Interpretability** |  **High** - Clear coefficients, WoE provides business intuition | ❌ **Low** - Black-box nature, feature importance but not causality |
| **Regulatory Compliance** |  **Easier** - Transparent decision process | ❓ **Challenging** - May require extensive validation and explanation |
| **Performance** |  **Moderate** - May miss complex interactions |  **High** - Captures non-linear relationships well |
| **Implementation Cost** |  **Low** - Easier to build, test, and maintain |  **High** - Requires more computational resources |
| **Risk Management** |  **Better** - Clear risk drivers facilitate stress testing |  **Complex** - Harder to identify failure modes |
| **Adaptability** |  **Limited** - Assumes linear relationships |  **Better** - Adapts to changing patterns |

**Recommended Hybrid Approach:**
1. **Start with interpretable models** for regulatory approval and stakeholder trust
2. **Use complex models** for initial screening where interpretability is less critical
3. **Implement model ensembles** that balance both needs
4. **Maintain parallel models** for comparison and validation

## Business Justification for Our Approach

### Why RFM as a Credit Risk Proxy:
1. **Recency:** Recent activity suggests ongoing financial engagement
2. **Frequency:** Regular transactions indicate stable income patterns
3. **Monetary:** Spending levels correlate with financial capacity

### Regulatory Alignment:
- **Transparent methodology:** RFM is well-understood in marketing, adaptable to credit
- **Audit-friendly:** Each component is easily traceable to raw data
- **Risk-based:** Aligns with Basel's emphasis on risk-sensitive approaches

### Implementation Strategy:
1. **Phase 1:** Deploy conservative model with high interpretability
2. **Phase 2:** Gather performance data on actual defaults
3. **Phase 3:** Refine model using real default labels when available
4. **Phase 4:** Gradually introduce complexity while maintaining oversight

---

## Project Setup

### Prerequisites
- Python 3.8+
- Git
- Docker (for deployment)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd credit-risk-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```