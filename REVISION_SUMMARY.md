# EML Paper Revision Summary for AICTA 2026

**Date:** 2026-05-11  
**Status:** Complete - Paper compiled successfully (12 pages)

---

## Summary of Changes

This revision strengthens the paper in response to reviewer-style criticism while preserving its core identity as a controlled empirical characterization study. The paper is NOT repositioned as a benchmark comparison or theory paper; instead, it is sharpened to emphasize the scientific study of "representation-induced search distortion" under constrained grammars.

---

## 1. Strengthened Motivation (Section 1 - Introduction)

**Changes:**
- Expanded opening paragraph to frame study as examining "search-space geometry" and "constrained symbolic manifolds"
- Added explicit statement: "Highly constrained grammars provide a controlled setting for studying how operator-space restrictions reshape valid symbolic search regions, allowing observation of representation-induced search distortion under strict validity enforcement"
- Maintained non-competitive framing: "The investigation makes no claim about universal expressivity or practical algorithmic superiority"

**Location:** Lines 52-67

---

## 2. Separate Rejection Types (Section 4 - Structural Results)

**Changes:**
- **Table 1 Updated:** Now shows 7 columns: Nodes, Depth, Domain Rej., Depth Rej., Num. Rej., Total Rej., Gens
- **New Data Analysis:** Correct rejection percentages calculated from raw counts
  - EML: Domain 0.6%, Depth 75.2%, Numeric 10.0% (Total 85.9%)
  - Baseline: Domain 6.5%, Depth 6.6%, Numeric 31.5% (Total 44.6%)
- **Key Finding:** Depth rejection dominates under EML (11.4× higher than baseline), NOT domain rejection as initially hypothesized
- **New Figure 3:** `rejection_composition.pdf` - stacked bar chart showing rejection type breakdown

**Location:** Table 1 (lines 204-218), Text (lines 253-260), New Figure (lines 244-249)

---

## 3. Reframed Correlation Figure (Figure 2 Caption)

**Changes:**
- Caption now emphasizes: "operational consequence of rejection-aware budget accounting under fixed evaluation budgets"
- Explicitly states: "not a claim of discovering a novel statistical law"
- Focuses on "throughput consequences" rather than correlation discovery

**Location:** Figure 2 caption (lines 270-272)

---

## 4. Strengthened Simple-EA Justification (Section 3)

**Changes:**
- Added explicit statement: "The search procedure is intentionally simple in order to isolate representational effects from optimizer-specific heuristics"
- Clarified: "This is a controlled grammar study, not a competitive optimizer evaluation"
- Added: "The intentional simplicity ensures that observed differences are attributable to grammar-induced search-space geometry rather than optimizer-specific enhancements"

**Location:** Lines 157

---

## 5. Enhanced Recovery Asymmetry Discussion (Section 5)

**Changes:**
- Identified specific targets recovered by EML (8/18) vs baseline (6/18)
- EML recovers: pure exponentials (exp(x), exp(x²)), pure logarithms (ln(x)), nested compositions (exp(ln(x+1)), ln(exp(x)))
- EML struggles with: mixed multiplicative targets (x²·ln(x), x²·exp(x)), rational forms (x/(1+x))
- Framed as: "task-dependent representational alignment" NOT superiority
- Noted EML recovers some targets that elude baseline (exp(x²), exp(x)+x)

**Location:** Lines 293-295

---

## 6. Added Constrained-GP Related Work (Section 2.1)

**Changes:**
- Added citations: Whigham (1995) on grammar-based GP, Trujillo et al. (2016) on semantic constraints, Virgolin et al. (2021) on operator-restricted search
- Expanded discussion of grammar-guided GP and its impact on search topology
- Added 3 new bibliography entries

**Location:** Lines 76-77, Bibliography (lines 423-436)

---

## 7. Qualitative Structural Theory (Section 2.3)

**Changes:**
- Added explanation of "compounding effect" where multiplicative structures experience disproportionately higher inflation
- Connected encoding rules to predictable expansion patterns
- Maintained qualitative framing (no formal proofs)

**Location:** Lines 116-118

---

## 8. New Rejection Type Analysis in Discussion (Section 6)

**Changes:**
- Added new subsection: "Rejection Type Analysis Clarifies Causal Mechanisms"
- Interprets depth rejection as primary bottleneck (not domain)
- Explains the cascade effect: structural inflation → depth rejection → masked domain constraints
- Concludes: "EML's fundamental limitation is representational compactness, not semantic validity"

**Location:** Lines 340-341

---

## Files Modified

1. **`paper/eml_representation_study_aicta2026.tex`** - Main paper with all text revisions
2. **`paper/generate_aicta_figures.py`** - Updated to generate rejection composition figure and print corrected table values
3. **`results/summary.md`** - Updated with rejection type breakdown and new figure references

---

## New/Updated Figures

1. **`figures/rejection_composition.pdf`** - NEW: Stacked bar chart showing domain/depth/numeric rejection breakdown
2. **`figures/rejection_vs_generations.pdf`** - Updated caption (content unchanged)
3. **`figures/rejection_distribution.pdf`** - Updated caption (content unchanged)
4. **`figures/tree_comparison.pdf`** - Unchanged

---

## Key Scientific Finding

The disaggregated rejection analysis reveals an unexpected result: **depth rejection dominates under EML (75.2%), not domain rejection (0.6%)**. This is because:

1. Depth checking occurs BEFORE domain verification in the search loop
2. EML macro-expansion causes 11.4× more depth rejections than baseline
3. Most EML candidates are depth-filtered before domain constraints are even checked
4. Domain constraints are real but masked by the depth bottleneck

This finding strengthens the paper's scientific contribution by revealing the true causal mechanism: structural capacity, not semantic validity, is the primary throughput limitation.

---

## Paper Identity Preserved

✓ **Still a controlled characterization study** - NOT a benchmark comparison  
✓ **Honest limitations preserved** - Small benchmark, simple EA, restricted domain  
✓ **Non-competitive framing maintained** - No claims of superiority  
✓ **Restrained tone intact** - Careful interpretation, reviewer-aware language  

---

## Compilation Status

✓ Paper compiles successfully (12 pages, 417KB PDF)  
✓ All new citations resolved  
✓ All figures generated and referenced  
✓ Table 1 updated with correct data values  
