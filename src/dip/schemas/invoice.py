from __future__ import annotations

from pydantic import Field

from dip.schemas.base import ExtractionBase


class InvoiceLineItem(ExtractionBase):
    description: str = Field(description="Line item description")
    quantity: float | None = Field(default=None, description="Units billed")
    unit_price: float | None = Field(default=None, description="Price per unit")
    amount: float | None = Field(default=None, description="Line total (quantity x unit_price)")


class Invoice(ExtractionBase):
    """Vendor invoice. Eval-critical fields (SROIE): company, date, address, total."""

    company: str = Field(description="Vendor / supplier company name issuing the invoice")
    invoice_number: str | None = Field(default=None, description="Invoice identifier / number")
    date: str | None = Field(default=None, description="Invoice date in ISO 8601 (YYYY-MM-DD)")
    address: str | None = Field(default=None, description="Vendor address exactly as printed")
    bill_to: str | None = Field(default=None, description="Customer / bill-to name")
    line_items: list[InvoiceLineItem] = Field(default_factory=list)
    subtotal: float | None = Field(default=None, description="Sum of line items before tax")
    tax: float | None = Field(default=None, description="Tax amount")
    total: float = Field(description="Grand total amount due")
    currency: str | None = Field(default=None, description="ISO currency code if shown")
