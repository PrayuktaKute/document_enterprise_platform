"""Phase 1b: generate synthetic purchase orders and medical reports as PDFs.

Ground truth is exact because we generate it. Output:
  data/raw/purchase_orders/po_NNN.pdf     + ground_truth.json
  data/raw/medical_reports/med_NNN.pdf    + ground_truth.json

    python scripts/gen_synthetic.py --purchase-orders 30 --medical 30 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"

PRODUCTS = [
    "Steel bolts M8", "Copper wire 2.5mm", "HDPE pipe 32mm", "Safety gloves (pair)",
    "LED panel 40W", "Ethernet cable Cat6", "Hydraulic seal kit", "Aluminium sheet 2mm",
    "Industrial lubricant 5L", "Circuit breaker 16A", "Packing tape roll", "Pallet wrap film",
]
CURRENCIES = ["USD", "EUR", "GBP", "INR"]

MODALITIES = {
    "Chest X-ray": {
        "site": "Chest",
        "normal": "The lungs are clear without focal consolidation, effusion, or pneumothorax. "
                  "Cardiomediastinal silhouette is within normal limits. Osseous structures are intact.",
        "abnormal": [
            ("Patchy airspace opacity in the right lower lobe consistent with pneumonia.",
             "Right lower lobe pneumonia."),
            ("Blunting of the left costophrenic angle indicating a small pleural effusion.",
             "Small left pleural effusion."),
            ("Increased interstitial markings with cephalitisation of pulmonary vasculature.",
             "Findings suggestive of mild pulmonary oedema."),
        ],
    },
    "CT Abdomen": {
        "site": "Abdomen and pelvis",
        "normal": "Liver, spleen, pancreas, adrenal glands and kidneys are unremarkable. "
                  "No free fluid or lymphadenopathy. Bowel loops are non-dilated.",
        "abnormal": [
            ("A 1.8 cm hypodense lesion in hepatic segment VI, likely a simple cyst.",
             "Probable benign hepatic cyst, segment VI."),
            ("Fat stranding around the appendix with a dilated appendix measuring 11 mm.",
             "Acute appendicitis."),
            ("Non-obstructing 4 mm calculus in the left renal pelvis.",
             "Small non-obstructing left renal calculus."),
        ],
    },
    "MRI Brain": {
        "site": "Brain",
        "normal": "No acute infarct, haemorrhage, or mass effect. Ventricles and sulci are "
                  "age-appropriate. No abnormal enhancement after contrast.",
        "abnormal": [
            ("A few scattered T2/FLAIR hyperintense foci in the subcortical white matter.",
             "Nonspecific white matter changes, likely microvascular."),
            ("Restricted diffusion in the left middle cerebral artery territory.",
             "Acute left MCA territory infarct."),
        ],
    },
    "Knee X-ray": {
        "site": "Left knee",
        "normal": "No fracture or dislocation. Joint spaces are preserved. No significant effusion.",
        "abnormal": [
            ("Medial joint space narrowing with marginal osteophytes.",
             "Moderate medial compartment osteoarthritis."),
            ("Lucent line through the proximal tibial metaphysis.",
             "Nondisplaced proximal tibial fracture."),
        ],
    },
    "Lab CBC panel": {
        "site": "Whole blood",
        "normal": "Haemoglobin, white cell count, and platelet count are within reference ranges. "
                  "Red cell indices are normal.",
        "abnormal": [
            ("Haemoglobin 9.1 g/dL with low MCV, consistent with microcytic anaemia.",
             "Microcytic anaemia, likely iron deficiency."),
            ("White cell count 14.8 x10^9/L with neutrophilia.",
             "Leukocytosis with neutrophil predominance."),
        ],
    },
}


def _rand_date(fake: Faker, start_days_ago: int = 400, span: int = 380) -> date:
    base = date.today() - timedelta(days=random.randint(0, start_days_ago))
    return base - timedelta(days=random.randint(0, span))


# --------------------------------------------------------------------------- #
def gen_purchase_orders(n: int, fake: Faker) -> None:
    out = RAW / "purchase_orders"
    out.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    gt: dict[str, dict] = {}

    for i in range(n):
        doc_id = f"po_{i:03d}"
        vendor = fake.company()
        buyer = fake.company()
        currency = random.choice(CURRENCIES)
        order_dt = _rand_date(fake)
        delivery_dt = order_dt + timedelta(days=random.randint(7, 45))
        po_number = f"PO-{random.randint(10000, 99999)}"
        ship_to = fake.address().replace("\n", ", ")

        items = []
        for _ in range(random.randint(2, 5)):
            desc = random.choice(PRODUCTS)
            qty = random.randint(1, 40)
            unit = round(random.uniform(2.5, 240.0), 2)
            items.append({"description": desc, "quantity": qty,
                          "unit_price": unit, "amount": round(qty * unit, 2)})
        subtotal = round(sum(x["amount"] for x in items), 2)
        tax = round(subtotal * random.choice([0.0, 0.05, 0.08, 0.18]), 2)
        total = round(subtotal + tax, 2)

        story = [
            Paragraph("<b>PURCHASE ORDER</b>", styles["Title"]),
            Spacer(1, 6),
            Paragraph(f"PO Number: <b>{po_number}</b>", styles["Normal"]),
            Paragraph(f"Order Date: {order_dt.isoformat()}", styles["Normal"]),
            Paragraph(f"Requested Delivery: {delivery_dt.isoformat()}", styles["Normal"]),
            Spacer(1, 10),
            Paragraph(f"<b>Buyer:</b> {buyer}", styles["Normal"]),
            Paragraph(f"<b>Vendor:</b> {vendor}", styles["Normal"]),
            Paragraph(f"<b>Ship To:</b> {ship_to}", styles["Normal"]),
            Spacer(1, 12),
        ]
        table_data = [["Description", "Qty", f"Unit ({currency})", f"Amount ({currency})"]]
        for it in items:
            table_data.append([it["description"], str(it["quantity"]),
                               f"{it['unit_price']:.2f}", f"{it['amount']:.2f}"])
        table_data += [
            ["", "", "Subtotal", f"{subtotal:.2f}"],
            ["", "", "Tax", f"{tax:.2f}"],
            ["", "", "Total", f"{total:.2f}"],
        ]
        tbl = Table(table_data, colWidths=[3.2 * inch, 0.7 * inch, 1.3 * inch, 1.4 * inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f4b7c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -4), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("FONTNAME", (2, -1), (3, -1), "Helvetica-Bold"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 16))
        story.append(Paragraph(f"Authorised by {fake.name()}, Procurement", styles["Italic"]))

        SimpleDocTemplate(str(out / f"{doc_id}.pdf"), pagesize=LETTER).build(story)
        gt[doc_id] = {
            "po_number": po_number,
            "order_date": order_dt.isoformat(),
            "vendor": vendor,
            "buyer": buyer,
            "ship_to": ship_to,
            "line_items": items,
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
            "delivery_date": delivery_dt.isoformat(),
            "currency": currency,
        }

    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"purchase_orders: wrote {n} PDFs + ground_truth.json -> {out}")


# --------------------------------------------------------------------------- #
def gen_medical(n: int, fake: Faker) -> None:
    out = RAW / "medical_reports"
    out.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    gt: dict[str, dict] = {}

    for i in range(n):
        doc_id = f"med_{i:03d}"
        modality = random.choice(list(MODALITIES))
        spec = MODALITIES[modality]
        report_dt = _rand_date(fake, 300, 300)
        patient_name = fake.name()
        patient_id = f"MRN-{random.randint(100000, 999999)}"
        physician = f"Dr. {fake.last_name()}"

        if random.random() < 0.55:
            extra_find, dx = random.choice(spec["abnormal"])
            findings = f"{spec['normal']} {extra_find}"
            impression = dx
            diagnoses = [dx.rstrip(".")]
        else:
            findings = spec["normal"]
            impression = f"No acute abnormality of the {spec['site'].lower()}."
            diagnoses = []

        story = [
            Paragraph("<b>RADIOLOGY / LABORATORY REPORT</b>", styles["Title"]),
            Spacer(1, 8),
            Paragraph(f"Patient Name: {patient_name}", styles["Normal"]),
            Paragraph(f"Patient ID: {patient_id}", styles["Normal"]),
            Paragraph(f"Report Date: {report_dt.isoformat()}", styles["Normal"]),
            Paragraph(f"Ordering Physician: {physician}", styles["Normal"]),
            Paragraph(f"Study / Modality: {modality}", styles["Normal"]),
            Paragraph(f"Body Site: {spec['site']}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("<b>FINDINGS</b>", styles["Heading2"]),
            Paragraph(findings, styles["Normal"]),
            Spacer(1, 10),
            Paragraph("<b>IMPRESSION</b>", styles["Heading2"]),
            Paragraph(impression, styles["Normal"]),
        ]
        SimpleDocTemplate(str(out / f"{doc_id}.pdf"), pagesize=LETTER).build(story)
        gt[doc_id] = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "report_date": report_dt.isoformat(),
            "ordering_physician": physician,
            "modality": modality,
            "body_site": spec["site"],
            "findings": findings,
            "impression": impression,
            "diagnoses": diagnoses,
        }

    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"medical_reports: wrote {n} PDFs + ground_truth.json -> {out}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--purchase-orders", type=int, default=30)
    ap.add_argument("--medical", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    fake = Faker()
    Faker.seed(args.seed)

    gen_purchase_orders(args.purchase_orders, fake)
    gen_medical(args.medical, fake)


if __name__ == "__main__":
    main()
