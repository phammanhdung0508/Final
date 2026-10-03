from movie4u.kg.schema import NodeType, RelationType


def test_minimum_kg_schema() -> None:
    assert NodeType.USER == "user"
    assert NodeType.MOVIE == "movie"
    assert RelationType.RATES == "rates"
