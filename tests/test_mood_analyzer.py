"""
Tests for MoodAnalyzer.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mood_analyzer import MoodAnalyzer


@pytest.fixture
def analyzer():
    return MoodAnalyzer()


# ---------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------

class TestPreprocess:
    def test_lowercase(self, analyzer):
        assert analyzer.preprocess("HAPPY") == ["happy"]

    def test_strips_whitespace(self, analyzer):
        assert analyzer.preprocess("  good  ") == ["good"]

    def test_strips_punctuation(self, analyzer):
        assert analyzer.preprocess("great!") == ["great"]
        assert analyzer.preprocess("sad,") == ["sad"]
        assert analyzer.preprocess("(awesome)") == ["awesome"]

    def test_splits_on_spaces(self, analyzer):
        assert analyzer.preprocess("I feel good") == ["i", "feel", "good"]

    def test_emoji_smiley_happy(self, analyzer):
        assert analyzer.preprocess(":)") == ["happy"]
        assert analyzer.preprocess(":-)") == ["happy"]

    def test_emoji_frown_sad(self, analyzer):
        assert analyzer.preprocess(":(") == ["sad"]
        assert analyzer.preprocess(":-(") == ["sad"]

    def test_unicode_emoji_happy(self, analyzer):
        assert analyzer.preprocess("😂") == ["happy"]

    def test_unicode_emoji_sad(self, analyzer):
        assert analyzer.preprocess("😭") == ["sad"]

    def test_unicode_emoji_mixed(self, analyzer):
        assert analyzer.preprocess("🥲") == ["mixed"]

    def test_normalizes_repeated_characters(self, analyzer):
        assert analyzer.preprocess("sooo") == ["so"]
        assert analyzer.preprocess("greeat") == ["great"]

    def test_empty_string(self, analyzer):
        assert analyzer.preprocess("") == []


# ---------------------------------------------------------------------
# score_text
# ---------------------------------------------------------------------

class TestScoreText:
    def test_positive_word_raises_score(self, analyzer):
        assert analyzer.score_text("I feel happy") > 0

    def test_negative_word_lowers_score(self, analyzer):
        assert analyzer.score_text("I feel sad") < 0

    def test_neutral_text_scores_zero(self, analyzer):
        assert analyzer.score_text("I went to the store") == 0

    def test_multiple_positive_words(self, analyzer):
        assert analyzer.score_text("great love happy") == 3

    def test_multiple_negative_words(self, analyzer):
        assert analyzer.score_text("sad bad terrible") == -3

    def test_mixed_words_cancel(self, analyzer):
        assert analyzer.score_text("happy sad") == 0

    def test_negation_not_happy(self, analyzer):
        assert analyzer.score_text("not happy") < 0

    def test_negation_not_sad(self, analyzer):
        assert analyzer.score_text("not sad") > 0

    def test_negation_never(self, analyzer):
        assert analyzer.score_text("never happy") < 0

    def test_negation_resets(self, analyzer):
        # "not happy good": negation applies to "happy" (-1), then "good" (+1) -> 0
        assert analyzer.score_text("not happy good") == 0

    def test_empty_text_scores_zero(self, analyzer):
        assert analyzer.score_text("") == 0

    def test_case_insensitive(self, analyzer):
        assert analyzer.score_text("HAPPY") > 0
        assert analyzer.score_text("SAD") < 0

    def test_punctuation_ignored(self, analyzer):
        assert analyzer.score_text("great!") > 0
        assert analyzer.score_text("awful.") < 0


# ---------------------------------------------------------------------
# predict_label
# ---------------------------------------------------------------------

class TestPredictLabel:
    def test_positive_text(self, analyzer):
        assert analyzer.predict_label("I love this so much") == "positive"

    def test_negative_text(self, analyzer):
        assert analyzer.predict_label("Today was terrible and awful") == "negative"

    def test_neutral_text(self, analyzer):
        assert analyzer.predict_label("I went outside") == "neutral"

    def test_negation_flips_to_negative(self, analyzer):
        assert analyzer.predict_label("not happy") == "negative"

    def test_negation_flips_to_positive(self, analyzer):
        assert analyzer.predict_label("not sad") == "positive"

    def test_returns_string(self, analyzer):
        assert isinstance(analyzer.predict_label("some text"), str)

    def test_label_in_known_set(self, analyzer):
        known = {"positive", "negative", "neutral"}
        for text in ["great", "hate", "meh"]:
            assert analyzer.predict_label(text) in known


# ---------------------------------------------------------------------
# Custom word lists
# ---------------------------------------------------------------------

class TestCustomWordLists:
    def test_custom_positive_words(self):
        analyzer = MoodAnalyzer(positive_words=["rad"], negative_words=[])
        assert analyzer.score_text("that was rad") == 1

    def test_custom_negative_words(self):
        analyzer = MoodAnalyzer(positive_words=[], negative_words=["bleak"])
        assert analyzer.score_text("feeling bleak") == -1

    def test_empty_word_lists(self):
        analyzer = MoodAnalyzer(positive_words=[], negative_words=[])
        assert analyzer.score_text("happy sad love hate") == 0

    def test_default_lists_loaded(self):
        analyzer = MoodAnalyzer()
        assert "happy" in analyzer.positive_words
        assert "sad" in analyzer.negative_words
