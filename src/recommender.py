from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    TFIDF_PATH,
    ITEM_VECTORS_PATH,
    CATALOG_PATH
)


class RecommendationEngine:
    """
    AI-powered learning resource recommendation engine.

    Core AI components:
    - TF-IDF for semantic text representation
    - Cosine similarity for interest-to-resource matching
    - Structured preference alignment
    - Weighted recommendation ranking
    - Human-readable recommendation explanations
    """

    # Recommendation weighting
    CONTENT_WEIGHT = 0.70
    PREFERENCE_WEIGHT = 0.30

    def __init__(self, model_dir=None):
        """
        Initialize the recommendation engine.

        Parameters
        ----------
        model_dir : optional
            Directory containing saved model artifacts.
            If omitted, project configuration is used.
        """

        self.model_dir = (
            Path(model_dir)
            if model_dir
            else TFIDF_PATH.parent
        )

        self.tfidf = None
        self.item_vectors = None
        self.catalog = None

        self.valid_difficulties = []
        self.valid_formats = []
        self.valid_goals = []

        self._load_artifacts()
        self._load_valid_preferences()

    # ---------------------------------------------------------
    # MODEL LOADING
    # ---------------------------------------------------------

    def _load_artifacts(self):
        """
        Load the trained recommendation artifacts.
        """

        tfidf_path = TFIDF_PATH
        vectors_path = ITEM_VECTORS_PATH
        catalog_path = CATALOG_PATH

        # Verify files exist before loading
        required_files = {
            "TF-IDF vectorizer": tfidf_path,
            "item vectors": vectors_path,
            "resource catalog": catalog_path
        }

        for name, path in required_files.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"{name} not found at: {path}"
                )

        # Load trained artifacts
        self.tfidf = joblib.load(tfidf_path)

        self.item_vectors = joblib.load(
            vectors_path
        )

        self.catalog = pd.read_pickle(
            catalog_path
        )

        # Basic consistency validation
        if len(self.catalog) != self.item_vectors.shape[0]:
            raise ValueError(
                "Catalog and item-vector dimensions do not match."
            )

    # ---------------------------------------------------------
    # PREFERENCE CONFIGURATION
    # ---------------------------------------------------------

    def _load_valid_preferences(self):
        """
        Extract valid preference options from the resource catalog.
        """

        self.valid_difficulties = sorted(
            self.catalog["difficulty"]
            .dropna()
            .unique()
            .tolist()
        )

        self.valid_formats = sorted(
            self.catalog["format"]
            .dropna()
            .unique()
            .tolist()
        )

        self.valid_goals = sorted(
            self.catalog["goal"]
            .dropna()
            .unique()
            .tolist()
        )

    # ---------------------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------------------

    def _validate_inputs(
        self,
        interests,
        difficulty,
        preferred_format,
        goal,
        top_n
    ):
        """
        Validate user-provided recommendation inputs.
        """

        # Interests
        if (
            not isinstance(interests, str)
            or not interests.strip()
        ):
            raise ValueError(
                "Interests cannot be empty. "
                "Please provide at least one interest."
            )

        # Difficulty
        if difficulty not in self.valid_difficulties:
            raise ValueError(
                f"Invalid difficulty: {difficulty}. "
                f"Choose from: {self.valid_difficulties}"
            )

        # Format
        if preferred_format not in self.valid_formats:
            raise ValueError(
                f"Invalid format: {preferred_format}. "
                f"Choose from: {self.valid_formats}"
            )

        # Goal
        if goal not in self.valid_goals:
            raise ValueError(
                f"Invalid goal: {goal}. "
                f"Choose from: {self.valid_goals}"
            )

        # Number of recommendations
        try:
            top_n = int(top_n)
        except (TypeError, ValueError):
            raise ValueError(
                "top_n must be a positive integer."
            )

        if top_n < 1:
            raise ValueError(
                "top_n must be at least 1."
            )

        # Never request more resources than exist
        return min(
            top_n,
            len(self.catalog)
        )

    # ---------------------------------------------------------
    # CONTENT SIMILARITY
    # ---------------------------------------------------------

    def _calculate_content_similarity(
        self,
        interests
    ):
        """
        Convert user interests into TF-IDF space
        and calculate similarity against every resource.
        """

        # Transform user interests using the
        # already-trained TF-IDF vectorizer
        user_vector = self.tfidf.transform(
            [interests]
        )

        # Compare user vector with every resource vector
        similarity_scores = cosine_similarity(
            user_vector,
            self.item_vectors
        ).flatten()

        return similarity_scores

    # ---------------------------------------------------------
    # PREFERENCE ALIGNMENT
    # ---------------------------------------------------------

    def _calculate_preference_alignment(
        self,
        results,
        difficulty,
        preferred_format,
        goal
    ):
        """
        Calculate how well each resource matches
        the user's structured preferences.
        """

        results = results.copy()

        # Difficulty match
        results["difficulty_match"] = (
            results["difficulty"] == difficulty
        ).astype(int)

        # Format match
        results["format_match"] = (
            results["format"] == preferred_format
        ).astype(int)

        # Goal match
        results["goal_match"] = (
            results["goal"] == goal
        ).astype(int)

        # Overall preference alignment
        results["preference_alignment"] = (
            results["difficulty_match"]
            + results["format_match"]
            + results["goal_match"]
        ) / 3

        return results

    # ---------------------------------------------------------
    # FINAL RANKING
    # ---------------------------------------------------------

    def _calculate_final_score(
        self,
        results
    ):
        """
        Combine semantic similarity and
        structured preference alignment.
        """

        results = results.copy()

        results["final_score"] = (
            self.CONTENT_WEIGHT
            * results["content_similarity"]
            +
            self.PREFERENCE_WEIGHT
            * results["preference_alignment"]
        )

        return results

    # ---------------------------------------------------------
    # EXPLANATIONS
    # ---------------------------------------------------------

    def _generate_explanation(
        self,
        row
    ):
        """
        Generate a human-readable explanation
        for an individual recommendation.
        """

        reasons = []

        # Interest relevance
        if row["content_similarity"] >= 0.60:
            reasons.append(
                "strong interest match"
            )

        elif row["content_similarity"] >= 0.30:
            reasons.append(
                "moderate interest match"
            )

        else:
            reasons.append(
                "some interest relevance"
            )

        # Difficulty
        if row["difficulty_match"] == 1:
            reasons.append(
                "matches your difficulty level"
            )

        # Format
        if row["format_match"] == 1:
            reasons.append(
                "matches your preferred format"
            )

        # Goal
        if row["goal_match"] == 1:
            reasons.append(
                "matches your learning goal"
            )

        return (
            "Recommended because it has "
            + ", ".join(reasons)
            + "."
        )

    # ---------------------------------------------------------
    # MAIN RECOMMENDATION METHOD
    # ---------------------------------------------------------

    def recommend(
        self,
        interests,
        difficulty,
        preferred_format,
        goal,
        top_n=5
    ):
        """
        Generate personalized learning-resource recommendations.

        Parameters
        ----------
        interests : str
            User's learning interests.

        difficulty : str
            Desired difficulty level.

        preferred_format : str
            Preferred resource format.

        goal : str
            User's learning goal.

        top_n : int
            Number of recommendations to return.

        Returns
        -------
        pandas.DataFrame
            Ranked personalized recommendations.
        """

        # Validate inputs
        top_n = self._validate_inputs(
            interests,
            difficulty,
            preferred_format,
            goal,
            top_n
        )

        # Start with catalog
        results = self.catalog.copy()

        # -----------------------------------------------------
        # STEP 1: AI CONTENT SIMILARITY
        # -----------------------------------------------------

        results["content_similarity"] = (
            self._calculate_content_similarity(
                interests
            )
        )

        # -----------------------------------------------------
        # STEP 2: STRUCTURED PREFERENCES
        # -----------------------------------------------------

        results = (
            self._calculate_preference_alignment(
                results,
                difficulty,
                preferred_format,
                goal
            )
        )

        # -----------------------------------------------------
        # STEP 3: FINAL AI RECOMMENDATION SCORE
        # -----------------------------------------------------

        results = self._calculate_final_score(
            results
        )

        # -----------------------------------------------------
        # STEP 4: RANK RESOURCES
        # -----------------------------------------------------

        results = results.sort_values(
            by="final_score",
            ascending=False
        )

        # Select top recommendations
        results = results.head(
            top_n
        ).copy()

        # -----------------------------------------------------
        # STEP 5: USER-FRIENDLY SCORE
        # -----------------------------------------------------

        results["match_percentage"] = (
            results["final_score"] * 100
        ).round(1)

        # -----------------------------------------------------
        # STEP 6: EXPLANATIONS
        # -----------------------------------------------------

        results["explanation"] = (
            results.apply(
                self._generate_explanation,
                axis=1
            )
        )

        # -----------------------------------------------------
        # STEP 7: FINAL OUTPUT
        # -----------------------------------------------------

        return results[
            [
                "item_id",
                "title",
                "category",
                "difficulty",
                "format",
                "goal",
                "content_similarity",
                "preference_alignment",
                "final_score",
                "match_percentage",
                "explanation"
            ]
        ].reset_index(
            drop=True
        )