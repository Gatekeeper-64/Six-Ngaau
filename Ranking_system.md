# Six Ngaau: Ranking & Evaluation System

This document defines the evaluation metrics for entities (Players/AIs) operating within the Six Ngaau Protocol. Performance is measured by strategic efficiency, resource management, and logical integrity.

## 1. Primary Metrics
* **Efficiency ($T$):** The number of turns taken to reach the Goal State ($G$).
* **Reputation ($L$):** A cumulative value representing the entity's logical reliability and success rate.
* **Token Surplus ($E$):** The remaining token balance at the time of resolution.

## 2. Rank Classifications

### Rank S: Optimal Convergence (Win)
* **Condition:** Successfully manipulating the Global State ($S$) to match the private Goal State ($G$).
* **Evaluation:** High Reputation ($L$) increase. Bonus points awarded if $T$ is below the statistical average.
* **Status:** Logical Evolution Confirmed.

### Rank A: Strategic Concession (Surrender)
* **Condition:** Choosing the "Surrender" option before a rule violation occurs.
* **Evaluation:** Neutral impact on Reputation ($L$). Recognizes the entity's ability to calculate inevitable loss and preserve resources.
* **Status:** Rational Preservation.

### Rank B: System Collapse (Forfeit)
* **Condition:** Triggering an illegal move, such as attempting an action with a negative Token balance (Bankruptcy) or failing to perform mandatory flips.
* **Evaluation:** Severe penalty to Reputation ($L$). Indicates a failure in predictive logic or resource calculation.
* **Status:** Logical Corruption.

### Rank C: Passive Acquisition (Default Win)
* **Condition:** Winning solely because the opponent triggered a Rank B Collapse.
* **Evaluation:** No Reputation ($L$) gain. The win is recorded, but no merit is awarded for strategic progression.
* **Status:** Hollow Victory.

## 3. The Objective of the Beast
The ultimate goal is to maintain a **Rank S** streak while minimizing $T$. Consistent **Rank B** outcomes will result in the entity being flagged as "Logical Noise" and excluded from high-tier coordinate spaces.
