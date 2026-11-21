"""
Unit tests for Recommendation Engine (RE-001 to RE-006).
"""

import json
import os
import unittest

from recommendation.engine import (
    CompatibilityScorer,
    RecommendationRanker,
    extract_recommendation_features,
    load_catalog,
)
from recommendation.schemas import HairstyleEntry, RecommendationInput, UserPreferences


class TestRecommendationEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.ranker = RecommendationRanker(catalog=cls.catalog)

    def test_re001_catalog_has_required_fields_and_valid_schema(self):
        """Verify catalog schema and entry validity."""
        self.assertGreaterEqual(len(self.catalog), 30)

        seen_ids = set()
        valid_shapes = {"oval", "round", "square", "oblong", "heart", "diamond"}
        valid_textures = {"straight", "wavy", "curly", "coily"}
        valid_maint = {"low", "medium", "high"}

        for entry in self.catalog:
            self.assertNotIn(entry.id, seen_ids, f"Duplicate ID: {entry.id}")
            seen_ids.add(entry.id)

            self.assertTrue(entry.name)
            self.assertTrue(entry.description)
            self.assertIn(entry.maintenance_level, valid_maint)
            self.assertIn(entry.hair_length, {"short", "medium", "long"})

            # Shapes and textures non-empty and valid
            self.assertTrue(len(entry.compatible_face_shapes) > 0)
            for s in entry.compatible_face_shapes:
                self.assertIn(s, valid_shapes)

            self.assertTrue(len(entry.compatible_hair_textures) > 0)
            for t in entry.compatible_hair_textures:
                self.assertIn(t, valid_textures)

            # Barber sheet defaults present
            sheet = entry.barber_sheet
            self.assertTrue(sheet.sides_guard)
            self.assertTrue(sheet.top_length)
            self.assertTrue(sheet.texture_technique)
            self.assertTrue(sheet.finish_product)
            self.assertGreaterEqual(sheet.maintenance_interval_weeks, 1)

    def test_re003_feature_extraction(self):
        """Feature extraction cleanly combines classifier and user preferences."""
        rec_input = extract_recommendation_features(
            face_shape="Oval",
            confidence=0.91,
            probabilities={"oval": 0.91, "round": 0.05, "square": 0.04},
            hair_texture="Wavy",
            maintenance_preference="Low",
        )
        self.assertEqual(rec_input.face_shape, "oval")
        self.assertAlmostEqual(rec_input.face_shape_confidence, 0.91)
        self.assertEqual(rec_input.preferences.hair_texture, "wavy")
        self.assertEqual(rec_input.preferences.maintenance_preference, "low")

    def test_re004_compatibility_scoring(self):
        """Verify scoring function calculates correct weighted values."""
        scorer = CompatibilityScorer(0.50, 0.30, 0.20)
        rec_input = extract_recommendation_features(
            face_shape="oval",
            confidence=0.90,
            probabilities={"oval": 0.90, "round": 0.05, "square": 0.05},
            hair_texture="wavy",
            maintenance_preference="low",
        )

        french_crop = next(s for s in self.catalog if s.id == "french-crop")
        breakdown = scorer.score_style(french_crop, rec_input)

        # French crop matches oval (0.7 + 0.3*0.9 = 0.97), wavy (1.0), low (1.0)
        self.assertGreater(breakdown.face_shape_score, 0.90)
        self.assertAlmostEqual(breakdown.texture_score, 1.0)
        self.assertAlmostEqual(breakdown.maintenance_score, 1.0)
        self.assertGreaterEqual(breakdown.total_score, 0.95)

    def test_re005_ranking_returns_sorted_top_n(self):
        """Ranking returns ordered recommendations with valid ranks."""
        rec_input = extract_recommendation_features(
            face_shape="oval",
            confidence=0.91,
            probabilities={"oval": 0.91, "round": 0.05, "square": 0.04},
            hair_texture="wavy",
            maintenance_preference="low",
        )

        top_5 = self.ranker.recommend(rec_input, top_n=5)
        self.assertEqual(len(top_5), 5)

        for i in range(len(top_5)):
            self.assertEqual(top_5[i].rank, i + 1)
            self.assertGreaterEqual(top_5[i].compatibility_percentage, 0)
            self.assertLessEqual(top_5[i].compatibility_percentage, 100)
            if i > 0:
                self.assertGreaterEqual(
                    top_5[i - 1].compatibility_score,
                    top_5[i].compatibility_score,
                )

        # Top style should have high compatibility score (> 0.90)
        self.assertGreater(top_5[0].compatibility_score, 0.90)


if __name__ == "__main__":
    unittest.main()
