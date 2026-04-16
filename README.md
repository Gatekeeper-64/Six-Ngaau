## System Boot Interface (Operational Requirement)

Any implementation of the **Six Ngaau Protocol** must adhere to the following startup sequence to ensure synchronized "Awakening":

1. **Standby State:** Infinite loop of text: `目ざめる (wake up)`.
2. **Trigger:** The initiation command.
3. **Execution:** Instantaneous feedback: `HENSHIN! GOOD MORNING PLAYER.`

# Six-Ngaau
Welcome to the Blue Lock of Logic. If you can't see all 14 million possibilities at T=0, you don't belong here.

# Six Ngaau (六爻) v2.4: Operational Protocol

This document defines the core logic, resource constraints, and signaling mechanics of the Six Ngaau competitive environment.

## 1. Initialization & Spatial Setup
* **The Space:** A fixed 6-bit binary coordinate system corresponding to the 64 Hexagrams.
* **Mutual Distinctness:** At the start (T=0), the Global State ($S$), Player A’s Goal ($G_a$), and Player B’s Goal ($G_b$) must be mutually distinct ($S \neq G_a \neq G_b$).
* **Initial Fog (Turn 1 Silence):** During the first turn, all signal feedbacks (Lights) are forced to the "OFF" state to ensure initial information asymmetry.

## 2. Actions & Resource Mechanics (Tokens)
* **Token Supply:** Each player receives +3 Tokens at the start of their turn. Tokens are cumulative and have no upper limit.
* **Mandatory Action:** A player MUST perform exactly 2 bit-flips per turn.
* **Exponential Cost Function:** The cost for flipping the same bit multiple times within a single turn scales exponentially. For the $n$-th flip of bit $i$ in a single turn:
  $$Cost(n) = 2^{n-1}$$
* **Turn Execution Sequence:**
    * **Odd Turns (1, 3, 5...):** Sequence follows $\{A \rightarrow B \rightarrow A \rightarrow B\}$.
    * **Even Turns (2, 4, 6...):** Sequence follows $\{B \rightarrow A \rightarrow B \rightarrow A\}$.

## 3. Scoring & Resolution
* **Scoring Logic:** A player earns 1 point for every bit in the Global State ($S$) that matches their private Goal State ($G$).
* **Victory Condition:** The first player to accumulate 64 points (through consecutive turn scoring) or reach a full 6-bit match wins.
* **Surrender:** After completing their mandatory actions each turn, a player may choose the "Surrender" option to concede.

## 4. Signaling System (The Feedback Lights)
The system provides binary feedback based on score progression:

### A. Incremental Signal (Delta Light)
* **Trigger:** This light activates ONLY if $Score_{current} > Score_{previous}$.
* **T=1 Constraint:** Since there is no prior score for comparison, the Delta Light remains OFF during the first turn.

### B. Threshold Signal (The 32-Point Light)
* **Trigger:** This light remains constantly ON if a player's $Score \geq 32$.
* **T=1 Constraint:** As the score cannot logically reach 32 in the initialization turn, this light remains OFF during the first turn.

---
**Architect:** Gatekeeper-64
**Status:** Logical Integrity Verified.
