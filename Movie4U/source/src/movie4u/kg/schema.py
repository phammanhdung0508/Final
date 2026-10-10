"""Canonical node and relation types for the Movie4U knowledge graph."""

from enum import StrEnum


class NodeType(StrEnum):
    USER = "user"
    MOVIE = "movie"
    GENRE = "genre"


class RelationType(StrEnum):
    RATES = "rates"
    HAS_GENRE = "has_genre"
