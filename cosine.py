"""
DERMA / INSIGHT - Product recommendation engine.

This module is intentionally separate from the existing prediction pipeline.
It accepts the skin-type and characteristic labels already produced by Flask,
then ranks products independently inside each product category.

Current matching features (same as COSINE.ipynb):
    Skin Type, Oily, Dry, Sensitive

The workbook's "Rank" column is not used as a similarity score.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = ["Skin Type", "Oily", "Dry", "Sensitive"]

DISPLAY_COLUMNS = [
    "Name",
    "Brand",
    "Type Product",
    "Price",
    "Skin Type",
    "Oily",
    "Dry",
    "Sensitive",
    "Composition",
    "Similarity Score",
    "Image",
    "Link",
]

SKIN_TYPES = {"Oily", "Dry", "Normal", "Combination"}
INTENSITY_LEVELS = {"Low", "Medium", "High"}

# These are the exact category labels used in the supplied workbook.
PRODUCT_CATEGORIES = (
    "moisturizer",
    "facewash",
    "sunscreen",
    "serum",
    "toner",
    "treatment",
)

REQUIRED_COLUMNS = {
    "Type Product",
    "Name",
    "Brand",
    "Composition",
    "Price",
    "Oily",
    "Dry",
    "Sensitive",
    "Skin Type",
    "Image",
    "Link",
}


def _normalise_label(value: str, allowed: set, field_name: str) -> str:
    """Strip whitespace and normalise title case, then validate a label."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")

    normalised = value.strip().title()
    if normalised not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {choices}. Got: {value!r}")
    return normalised


def _default_dataset_path() -> Path:
    """
    Find the workbook in common project locations.

    Recommended project location:
        <project root>/dataset/product-recomendation.xlsx

    The misspelling 'recomendation' intentionally matches the supplied file.
    """
    project_root = Path(__file__).resolve().parent
    candidates = [
        project_root / "dataset" / "product-recomendation.xlsx",
        project_root / "product-recomendation.xlsx",
        project_root / "dataset" / "product-recomendation(1).xlsx",
        project_root / "product-recomendation(1).xlsx",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find the product workbook. Place it at "
        "'dataset/product-recomendation.xlsx' or pass dataset_path explicitly. "
        f"Locations checked:\n{searched}"
    )


class ProductRecommender:
    """
    Loads the product workbook and fits OneHotEncoder once.

    Create one instance when the Flask application starts and reuse it for
    requests. Do not instantiate this class inside the /predict route.
    """

    def __init__(self, dataset_path: Optional[str | os.PathLike] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else _default_dataset_path()

        if not self.dataset_path.is_file():
            raise FileNotFoundError(
                f"Product dataset not found: {self.dataset_path}"
            )

        self.products = self._load_and_validate_dataset()

        # Keep the same encoding approach used in COSINE.ipynb.
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.product_features = self.encoder.fit_transform(
            self.products[FEATURE_COLUMNS]
        )

    def _load_and_validate_dataset(self) -> pd.DataFrame:
        """Read the Products sheet, clean labels, and validate its schema."""
        try:
            frame = pd.read_excel(self.dataset_path, sheet_name="Products")
        except ValueError as exc:
            raise ValueError(
                "The product workbook must contain a sheet named 'Products'."
            ) from exc

        frame.columns = frame.columns.astype(str).str.strip()

        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(
                "Product dataset is missing required columns: "
                + ", ".join(missing)
            )

        # Normalise text fields without modifying the source workbook.
        text_columns = [
            "Type Product", "Name", "Brand", "Composition",
            "Oily", "Dry", "Sensitive", "Skin Type", "Image", "Link",
        ]
        for column in text_columns:
            frame[column] = frame[column].fillna("").astype(str).str.strip()

        # Empty required fields would make filtering or display unreliable.
        required_nonempty = [
            "Type Product", "Name", "Brand",
            "Oily", "Dry", "Sensitive", "Skin Type",
        ]
        for column in required_nonempty:
            if (frame[column] == "").any():
                raise ValueError(
                    f"Product dataset contains blank values in {column!r}."
                )

        # Validate dataset labels instead of silently accepting unexpected ones.
        frame["Skin Type"] = frame["Skin Type"].str.title()
        frame["Oily"] = frame["Oily"].str.title()
        frame["Dry"] = frame["Dry"].str.title()
        frame["Sensitive"] = frame["Sensitive"].str.title()
        frame["Type Product"] = frame["Type Product"].str.lower()

        invalid_skin_types = set(frame["Skin Type"]) - SKIN_TYPES
        if invalid_skin_types:
            raise ValueError(
                f"Unexpected Skin Type labels in product dataset: "
                f"{sorted(invalid_skin_types)}"
            )

        for column in ("Oily", "Dry", "Sensitive"):
            invalid_levels = set(frame[column]) - INTENSITY_LEVELS
            if invalid_levels:
                raise ValueError(
                    f"Unexpected labels in {column}: {sorted(invalid_levels)}"
                )

        unexpected_categories = set(frame["Type Product"]) - set(PRODUCT_CATEGORIES)
        if unexpected_categories:
            raise ValueError(
                "Unexpected product categories in dataset: "
                f"{sorted(unexpected_categories)}"
            )

        # Price must be numeric for the frontend to display it consistently.
        frame["Price"] = pd.to_numeric(frame["Price"], errors="coerce")
        if frame["Price"].isna().any():
            raise ValueError("Product dataset contains a non-numeric or blank Price.")

        # Preserve original row order as a stable tie-breaker.
        frame = frame.reset_index(drop=True)
        frame["_source_order"] = range(len(frame))

        if frame.empty:
            raise ValueError("The product dataset contains no products.")

        return frame

    def recommend_by_category(
        self,
        skin_type: str,
        oily: str,
        dry: str,
        sensitive: str,
        top_n: int = 3,
    ) -> Dict[str, List[dict]]:
        """
        Return up to top_n products for each of the six categories.

        Products are first restricted to the selected skin type and category.
        Cosine similarity is then calculated using the same four one-hot
        encoded profile features as the original notebook.

        The returned dictionary always contains all six category keys.
        A category can have an empty list if the workbook has no matching rows.
        """
        skin_type = _normalise_label(skin_type, SKIN_TYPES, "skin_type")
        oily = _normalise_label(oily, INTENSITY_LEVELS, "oily")
        dry = _normalise_label(dry, INTENSITY_LEVELS, "dry")
        sensitive = _normalise_label(sensitive, INTENSITY_LEVELS, "sensitive")

        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer.")

        user_profile = pd.DataFrame([{
            "Skin Type": skin_type,
            "Oily": oily,
            "Dry": dry,
            "Sensitive": sensitive,
        }])

        # Transform with the exact encoder fitted on the product dataset.
        user_features = self.encoder.transform(user_profile)

        recommendations: Dict[str, List[dict]] = {}

        for category in PRODUCT_CATEGORIES:
            # Filter by category AND the selected skin type first.
            candidates = self.products[
                (self.products["Type Product"] == category)
                & (self.products["Skin Type"] == skin_type)
            ].copy()

            if candidates.empty:
                recommendations[category] = []
                continue

            # Keep the same four-feature cosine similarity approach as the notebook.
            candidate_features = self.product_features[candidates.index]
            scores = cosine_similarity(
                user_features,
                candidate_features,
            ).ravel()

            candidates["Similarity Score"] = scores

            # Similarity is the primary ordering. Original row order breaks ties.
            candidates = candidates.sort_values(
                by=["Similarity Score", "_source_order"],
                ascending=[False, True],
                kind="mergesort",
            ).head(top_n)

            # Convert each row to ordinary Python values for safe Flask/Jinja use.
            category_results = []
            for row in candidates[DISPLAY_COLUMNS].to_dict(orient="records"):
                row["Price"] = float(row["Price"])
                row["Similarity Score"] = float(row["Similarity Score"])
                category_results.append(row)

            recommendations[category] = category_results

        return recommendations


# Convenience function for simple scripts/tests. In Flask, prefer creating
# ProductRecommender once at application startup and reusing that instance.
def recommend_products(
    skin_type: str,
    oily: str,
    dry: str,
    sensitive: str,
    top_n: int = 3,
    dataset_path: Optional[str | os.PathLike] = None,
) -> Dict[str, List[dict]]:
    recommender = ProductRecommender(dataset_path=dataset_path)
    return recommender.recommend_by_category(
        skin_type=skin_type,
        oily=oily,
        dry=dry,
        sensitive=sensitive,
        top_n=top_n,
    )


if __name__ == "__main__":
    # Quick local smoke test:
    # python cosine.py
    engine = ProductRecommender()
    result = engine.recommend_by_category(
        skin_type="Combination",
        oily="High",
        dry="Low",
        sensitive="Medium",
        top_n=3,
    )

    for category, products in result.items():
        print(f"\n{category.title()} ({len(products)} recommendations)")
        for product in products:
            print(
                f"- {product['Name']} | {product['Brand']} | "
                f"Similarity: {product['Similarity Score']:.4f}"
            )
