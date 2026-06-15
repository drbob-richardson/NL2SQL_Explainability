"""Schema model for the restricted single-table fragment.

A schema is one typed relation. Types drive the context-sensitive constraints Phi
(numeric-only ops/aggregates, text-only LIKE) used when counting / generating the valid
query space. Running example: the DataCamp `airbnb_listings` table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ColType(str, Enum):
    NUM = "num"
    TEXT = "text"


@dataclass(frozen=True)
class Column:
    name: str
    type: ColType


@dataclass(frozen=True)
class Schema:
    table: str
    columns: tuple[Column, ...]

    @property
    def num_cols(self) -> tuple[Column, ...]:
        return tuple(c for c in self.columns if c.type is ColType.NUM)

    @property
    def text_cols(self) -> tuple[Column, ...]:
        return tuple(c for c in self.columns if c.type is ColType.TEXT)

    @property
    def m(self) -> int:
        return len(self.columns)


# The cheat-sheet schema.
AIRBNB = Schema(
    table="airbnb_listings",
    columns=(
        Column("id", ColType.NUM),
        Column("city", ColType.TEXT),
        Column("country", ColType.TEXT),
        Column("number_of_rooms", ColType.NUM),
        Column("year_listed", ColType.NUM),
    ),
)
