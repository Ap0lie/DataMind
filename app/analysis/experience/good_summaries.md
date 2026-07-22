# Good Summary Examples

## Core Principle

The internal reasoning chain can use facts, operating logic, attribution, and actions, but the final report must not expose those labels as section titles.

The output should read like an analyst's conclusion, not like a prompt template.

## Rule 1: Stay Inside Data Boundaries

Every finding must be traceable to fields or computed results from the current run.

- If data only shows what happened, write the fact.
- If data supports why it happened, write the attribution and mark it as inferred.
- If data does not identify who or what caused it, do not invent a cause.
- Organizational attribution is optional and must be skipped when unsupported.

## Rule 2: Analysis Chain Template

```
Metric movement
  -> Decompose to driving dimensions
    -> Attribute the driving dimensions
      -> Stop at the data boundary and name the next data needed
```

Do not stop before the boundary, and do not reason beyond it.

## Rule 3: Output Format

- Use paragraphs and concise numbered recommendations.
- Embed numbers directly in sentences.
- Do not use internal labels such as "data fact", "operating logic", "organizational attribution", or "business conclusion".
- Avoid vague phrasing such as "needs attention", "may have a problem", "overall trend is good", or "follow up later".

## Example A: Event-driven Shock

2024 furniture profit rate is 9.76%, down 3.89pp year over year, the largest decline among the major categories. Monthly decomposition shows that more than 90% of the decline is concentrated in Q3, while the remaining months stay close to the previous year. Q3 is the single event that pulled down the annual rate; the annual pricing structure itself is not yet proven to be structurally broken.

Recommended next checks:

1. Add Q3 promotion discount and cost breakdown data to separate GMV impact from ROI impact.
2. Do not change annual pricing policy before the Q3 mechanism is confirmed.

## Example B: Chronic Structural Drift

Technology appears stable at the total level, but subcategory mix is drifting. A high-margin subcategory is shrinking while lower-margin subcategories are expanding. This mix shift is hidden in the total metric and should be treated as a chronic risk before it visibly lowers the category baseline.

Recommended next checks:

1. Set a warning line for the shrinking high-margin subcategory.
2. Add sales behavior data to separate external competition from internal promotion structure.
