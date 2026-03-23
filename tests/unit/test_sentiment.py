"""Tests for VADER sentiment analysis helpers."""

from depwatch.core.sentiment import compound_score, mean_compound


class TestCompoundScore:
    def test_positive_text(self) -> None:
        score = compound_score("This project is great and amazing!")
        assert score > 0.0

    def test_negative_text(self) -> None:
        score = compound_score("This is terrible and broken")
        assert score < 0.0

    def test_neutral_text(self) -> None:
        score = compound_score("Update version number")
        assert -0.5 < score < 0.5

    def test_empty_string(self) -> None:
        assert compound_score("") == 0.0

    def test_whitespace_only(self) -> None:
        assert compound_score("   ") == 0.0

    def test_score_range(self) -> None:
        score = compound_score("absolutely wonderful fantastic project")
        assert -1.0 <= score <= 1.0


class TestMeanCompound:
    def test_empty_list(self) -> None:
        assert mean_compound([]) == 0.0

    def test_single_text(self) -> None:
        result = mean_compound(["great"])
        expected = compound_score("great")
        assert result == expected

    def test_multiple_texts(self) -> None:
        texts = ["great", "terrible", "okay"]
        result = mean_compound(texts)
        scores = [compound_score(t) for t in texts]
        assert abs(result - sum(scores) / len(scores)) < 1e-9
