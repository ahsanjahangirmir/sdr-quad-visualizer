# sdr-quad-visualizer

URL Query Params

You can deep-link the exact chart configuration using URL query parameters.  
Open the app, change settings, then use **Copy chart URL** to share a link that reproduces the same view.

---

## Supported Parameters

- `person` — column name used as the identifier (hover/table)
- `coe_vars` — comma list of COE inputs:
  - `dials`, `call`, `emails`, `sms`
- `w_dials`, `w_call`, `w_emails`, `w_sms` — weights for each COE input (floats)
- `coe_denom` — denominator for Normalized COE (float)
- `effort` — which inputs to include in Effort:
  - `both`, `coe`, `active`
- `w_coe` — weight for Normalized COE when `effort=both` (0..1)
- `perf` — which inputs to include in Performance:
  - `both`, `cr`, `ta`
- `cr_tf` — Conversion transform: `no` or `norm`
- `cr_norm` — Normalization factor for Conversion (float)
- `ta_tf` — Attainment transform: `no`, `clip`, or `norm`
- `w_cr` — weight for Conversion when `perf=both` (0..1)
- `final_tf` — final transform on Performance %: `no`, `clip`, or `norm`
- `e_thr` — Effort threshold % (0..100)
- `p_thr` — Performance threshold % (0..100)

---

## Examples

**Full example (defaults):**

https://sdr-quad-visualizer-3xjc29ejzxihpcshuqkyvu.streamlit.app/?person=Person&coe_vars=dials,call,emails,sms&w_dials=0.8&w_call=0.1&w_emails=0.05&w_sms=0.05&coe_denom=1000&effort=both&w_coe=0.6&perf=both&cr_tf=norm&cr_norm=0.025&ta_tf=no&w_cr=0.5&final_tf=clip&e_thr=60&p_thr=60


**Conversion-only with custom norm and thresholds:**

?perf=cr&cr_tf=norm&cr_norm=0.02&final_tf=no&e_thr=55&p_thr=70


---

## How it works

- On page load, the app reads `st.query_params` and pre-fills control defaults.
- Every time you change controls, the app flattens the current settings into a stable set of parameters and **updates the URL** (without reloading the page).
- Use **Copy chart URL** to copy an **absolute** link (origin + path + querystring) ready to share.

---

## Notes

- Unknown or malformed values are ignored and the app falls back to sensible defaults.
- When `effort != both`, the app still persists `w_coe` in the URL so switching back to `both` restores your previous weight.
- `coe_vars` order does not matter; values are validated against known inputs.


