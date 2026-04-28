"""SKU catalog. Reads `data/sku_catalog.csv` once on first access."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from .models import EnvCondition, FragilityClass, Vec3

_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "sku_catalog.csv"


@dataclass(frozen=True)
class ItemTemplate:
    sku: str
    name: str
    weight_kg: float
    dims_m: Vec3
    fragility: FragilityClass
    category: str
    default_env: EnvCondition


@cache
def load_catalog() -> dict[str, ItemTemplate]:
    out: dict[str, ItemTemplate] = {}
    with _CATALOG_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["sku"]] = ItemTemplate(
                sku=row["sku"],
                name=row["name"],
                weight_kg=float(row["weight_kg"]),
                dims_m=(float(row["length_m"]), float(row["width_m"]), float(row["height_m"])),
                fragility=FragilityClass(row["fragility"]),
                category=row["category"],
                default_env=EnvCondition(row["default_env"]),
            )
    return out


def get(sku: str) -> ItemTemplate:
    catalog = load_catalog()
    if sku not in catalog:
        raise KeyError(f"unknown SKU: {sku}")
    return catalog[sku]


def all_skus() -> list[str]:
    return sorted(load_catalog().keys())


def by_category(category: str) -> list[ItemTemplate]:
    return [t for t in load_catalog().values() if t.category == category]


def by_env(env: EnvCondition) -> list[ItemTemplate]:
    return [t for t in load_catalog().values() if t.default_env == env]
