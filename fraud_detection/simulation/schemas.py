"""Strict versioned simulation contracts; no raw card or personal-data fields."""
from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Token = Annotated[str, Field(pattern=r"^[a-z]+_[a-zA-Z0-9]{3,64}$", max_length=80)]


def utcnow():
    return datetime.now(timezone.utc)


def identifier(prefix, value):
    return f"{prefix}_{uuid5(NAMESPACE_URL, value).hex}"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Envelope(Contract):
    schema_version: Literal[1] = 1
    simulation: Literal[True] = True
    event_id: Token
    transaction_id: Token
    trace_id: Token
    event_time: AwareDatetime


class ThreeDS(Contract):
    version: Literal["2.2"] = "2.2"
    authenticated: bool


class Authorization(Envelope):
    event_type: Literal["authorization_requested"] = "authorization_requested"
    source: Literal["ieee_replay", "synthetic"]
    customer_token: Token
    card_token: Token
    merchant_id: Token
    amount: Annotated[Decimal, Field(gt=0, le=10000000, max_digits=10, decimal_places=2)]
    currency: Literal["EUR", "USD", "GBP"]
    country: Literal["FR", "DE", "GB", "US", "ES", "IT", "NL"]
    channel: Literal["ecommerce", "pos", "mobile"]
    device_id: Token
    ip_hash: Token
    merchant_category: Annotated[str, Field(max_length=16, pattern=r"^[a-zA-Z0-9_]+$")]
    three_ds: ThreeDS
    identity_data_missing: bool = True
    feature_ref: Token | None = None

    @model_validator(mode="after")
    def source_contract(self):
        if (self.source == "ieee_replay") != (self.feature_ref is not None):
            raise ValueError("Only IEEE replay events must reference the private benchmark feature store")
        return self


class FraudLabel(Envelope):
    event_type: Literal["fraud_label_received"] = "fraud_label_received"
    label: Literal[0, 1]
    label_type: Literal["chargeback", "historical_confirmation"]
    label_time: AwareDatetime
    authorization_time: AwareDatetime

    @model_validator(mode="after")
    def delayed(self):
        if self.label_time <= self.authorization_time or self.event_time != self.label_time:
            raise ValueError("Labels must be released strictly after authorization at label_time")
        if (self.label == 1) != (self.label_type == "chargeback"):
            raise ValueError("Simulated chargebacks represent positive historical labels")
        return self


class Context(Contract):
    transactions_5m: int = 0
    amount_1h: float = 0
    failures_5m: int = 0
    cards_on_device_24h: int = 0
    last_country: str | None = None
    last_device: str | None = None
    last_time: float | None = None
    device_first_seen: float | None = None


class Thresholds(Contract):
    review: float = 0.55
    block: float = 0.98
    anomaly_review: float = 0.80

    @model_validator(mode="after")
    def ordered(self):
        if not 0 <= self.review < self.block <= 1:
            raise ValueError("Require 0 <= review < block <= 1")
        if not 0 <= self.anomaly_review <= 1:
            raise ValueError("Require 0 <= anomaly_review <= 1")
        return self


class Decision(Envelope):
    event_type: Literal["fraud_scored"] = "fraud_scored"
    request_event_id: Token
    decision: Literal["approve", "manual_review", "decline"]
    risk_score: Annotated[float, Field(ge=0, le=1)] | None
    anomaly_score: Annotated[float, Field(ge=0, le=1)] | None = None
    thresholds: Thresholds
    reasons: list[str]
    model_version: str
    feature_version: Literal["recent-context-v1+ieee-benchmark-v1"] = "recent-context-v1+ieee-benchmark-v1"
    received_at: AwareDatetime
    decided_at: AwareDatetime
    feature_timestamp: AwareDatetime
    decision_time_ms: float
    context: Context
    amount: float
    source: Literal["ieee_replay", "synthetic"]


class PaymentEvent(Envelope):
    event_type: Literal["authorization_approved", "authorization_declined", "manual_review_requested",
                        "payment_captured", "payment_refunded", "chargeback_received"]
    parent_event_id: Token
