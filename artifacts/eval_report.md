# Extraction Evaluation

- Documents scored: **120**
- **Field-level extraction accuracy: 82.7%**
- Classification accuracy: 94.2%
- **Manual-verification reduction (auto-accept rate): 65%**
- Accuracy within auto-accepted: 90.8% (vs 56.5% in the review queue)
- Auto-accepted docs error-free: 47.4% (37/78)
- Confidence calibration ECE: 0.078
- Mean pipeline latency: 10.1s/doc

## Field accuracy by document type

| type | accuracy |
|---|---|
| invoice | 64.2% |
| purchase_order | 99.1% |
| medical_report | 89.6% |
| contract | 53.8% |

## Field counts

| verdict | n |
|---|---|
| correct | 730 |
| incorrect | 39 |
| missing | 105 |
| spurious | 9 |

## Calibration buckets

| conf bucket | n | mean conf | accuracy |
|---|---|---|---|
| 0.2-0.3 | 1 | 0.225 | 0.0 |
| 0.3-0.4 | 8 | 0.364 | 0.375 |
| 0.4-0.5 | 10 | 0.46 | 0.7 |
| 0.5-0.6 | 36 | 0.548 | 0.806 |
| 0.6-0.7 | 122 | 0.613 | 0.254 |
| 0.7-0.8 | 22 | 0.751 | 0.682 |
| 0.8-0.9 | 26 | 0.853 | 0.769 |
| 0.9-1.0 | 637 | 0.994 | 0.981 |

## Per-field accuracy

| type.field | accuracy |
|---|---|
| contract.agreement_date | 55.2% |
| contract.document_name | 80.0% |
| contract.effective_date | 70.8% |
| contract.expiration_or_term | 20.0% |
| contract.governing_law | 14.8% |
| contract.parties | 90.0% |
| contract.renewal_term | 0.0% |
| invoice.address | 56.7% |
| invoice.company | 63.3% |
| invoice.date | 70.0% |
| invoice.total | 66.7% |
| medical_report.body_site | 93.3% |
| medical_report.diagnoses | 55.0% |
| medical_report.findings | 63.3% |
| medical_report.impression | 100.0% |
| medical_report.modality | 100.0% |
| medical_report.ordering_physician | 100.0% |
| medical_report.patient_id | 100.0% |
| medical_report.patient_name | 83.3% |
| medical_report.report_date | 100.0% |
| purchase_order.buyer | 96.7% |
| purchase_order.currency | 100.0% |
| purchase_order.delivery_date | 100.0% |
| purchase_order.line_items | 100.0% |
| purchase_order.order_date | 100.0% |
| purchase_order.po_number | 100.0% |
| purchase_order.ship_to | 96.7% |
| purchase_order.subtotal | 96.7% |
| purchase_order.tax | 100.0% |
| purchase_order.total | 100.0% |
| purchase_order.vendor | 100.0% |