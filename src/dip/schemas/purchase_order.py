from __future__ import annotations

from pydantic import Field

from dip.schemas.base import ExtractionBase


class POLineItem(ExtractionBase):
    description: str = Field(description="Item or service description")
    quantity: float | None = Field(default=None, description="Quantity ordered")
    unit_price: float | None = Field(default=None, description="Price per unit")
    amount: float | None = Field(default=None, description="Line total")


class PurchaseOrder(ExtractionBase):
    """Buyer-issued purchase order (synthetic dataset; ground truth from generator)."""

    po_number: str = Field(description="Purchase order number")
    order_date: str | None = Field(default=None, description="Order date in ISO 8601 (YYYY-MM-DD)")
    vendor: str = Field(description="Vendor / supplier the PO is sent to")
    buyer: str | None = Field(default=None, description="Buying organisation")
    ship_to: str | None = Field(default=None, description="Ship-to address")
    line_items: list[POLineItem] = Field(default_factory=list)
    subtotal: float | None = Field(default=None, description="Sum of line items before tax")
    tax: float | None = Field(default=None, description="Tax amount")
    total: float | None = Field(default=None, description="Total order value")
    delivery_date: str | None = Field(default=None, description="Requested delivery date, ISO 8601")
    currency: str | None = Field(default=None, description="ISO currency code if shown")
