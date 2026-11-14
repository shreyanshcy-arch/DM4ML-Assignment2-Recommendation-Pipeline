
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from src.ingestion.ingest import ingest
from src.validation.validate import validate
from src.preparation.prepare import prepare
from src.transformation.transform import transform
from src.features.build_features import build_features
from src.training.train import train_models
from src.evaluation.evaluate import compare_models


def main():

    # --------------------------------------------------
    # 1. Create required project directories
    # --------------------------------------------------

    folders = [
        "data/raw",
        "data/processed",
        "data/errors",
        "data/features",
        "models",
        "reports"
    ]

    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

    log = []

    def stage(message):
        print(message)
        log.append(message)

    try:

        # --------------------------------------------------
        # 2. DATA INGESTION
        # --------------------------------------------------

        stage("[1/7] Data ingestion")

        users, movies, ratings = ingest()

        stage(
            f"Loaded {len(users):,} users, "
            f"{len(movies):,} movies, "
            f"{len(ratings):,} ratings"
        )

        # --------------------------------------------------
        # 3. DATA VALIDATION
        # --------------------------------------------------

        stage("[2/7] Data validation")

        valid, validation_report = validate(
            users,
            movies,
            ratings
        )

        stage(
            f"Validation: {validation_report['status']} | "
            f"valid ratings: {len(valid):,}"
        )

        # Stop if validation fails
        if validation_report["status"] != "PASS":
            raise ValueError(
                "Data validation failed. "
                "Check reports/validation_report.json"
            )

        # --------------------------------------------------
        # 4. DATA PREPARATION
        # --------------------------------------------------

        stage(
            "[3/7] Data preparation and chronological split"
        )

        train, test, all_data = prepare(valid, movies)

        stage(
            f"Train: {len(train):,} | "
            f"Test: {len(test):,}"
        )

        # --------------------------------------------------
        # 5. DATA TRANSFORMATION
        # --------------------------------------------------

        stage("[4/7] Data transformation")

        matrix, user_summary, item_summary = transform(
            train,
            test,
            movies
        )

        # --------------------------------------------------
        # 6. FEATURE MANAGEMENT
        # --------------------------------------------------

        stage("[5/7] Feature management")

        user_features, item_features, interaction_features = (
            build_features(train, movies)
        )

        # --------------------------------------------------
        # 7. MODEL TRAINING
        # --------------------------------------------------

        stage("[6/7] Model training")

        popularity, item_cf = train_models(train)

        stage(
            "Trained popularity and "
            "item-item collaborative filtering models"
        )

        # --------------------------------------------------
        # 8. MODEL EVALUATION
        # --------------------------------------------------

        stage("[7/7] Model evaluation and tracking")

        comparison = compare_models(
            popularity,
            item_cf,
            train,
            test,
            movies
        )

        stage(
            "Model comparison saved to "
            "reports/model_comparison.csv"
        )

        # --------------------------------------------------
        # 9. MLFLOW TRACKING
        # --------------------------------------------------

        try:

            import mlflow

            mlflow.set_tracking_uri(
                "sqlite:///mlflow.db"
            )

            mlflow.set_experiment(
                "DM4ML_Assignment2_Recommendation"
            )

            # Map model names to saved model files
            model_artifacts = {
                "Popularity": "models/popularity_model.joblib",
                "Item-item CF": "models/item_item_cf_model.joblib"
            }

            # Create a separate MLflow run for each model
            for _, row in comparison.iterrows():

                model_name = row["Model"]

                if model_name == "Popularity":

                    run_name = "Popularity_Recommender"

                    algorithm = (
                        "Rating count with "
                        "mean-rating tie breaking"
                    )

                elif model_name == "Item-item CF":

                    run_name = "Item_Item_CF_Recommender"

                    algorithm = (
                        "Item-item collaborative filtering "
                        "with cosine similarity"
                    )

                else:
                    raise ValueError(
                        f"Unknown model: {model_name}"
                    )

                with mlflow.start_run(
                    run_name=run_name
                ):

                    # --------------------------------------
                    # Model parameters
                    # --------------------------------------

                    mlflow.log_param(
                        "model_name",
                        model_name
                    )

                    mlflow.log_param(
                        "algorithm",
                        algorithm
                    )

                    mlflow.log_param(
                        "dataset",
                        "MovieLens ml-latest-small"
                    )

                    mlflow.log_param(
                        "train_rows",
                        len(train)
                    )

                    mlflow.log_param(
                        "test_rows",
                        len(test)
                    )

                    mlflow.log_param(
                        "test_fraction",
                        0.20
                    )

                    mlflow.log_param(
                        "k",
                        5
                    )

                    mlflow.log_param(
                        "relevance_threshold",
                        4.0
                    )

                    # --------------------------------------
                    # Evaluation metrics
                    # --------------------------------------

                    for metric in comparison.columns[1:]:

                        metric_name = (
                            metric.replace("@", "_at_")
                        )

                        mlflow.log_metric(
                            metric_name,
                            float(row[metric])
                        )

                    # --------------------------------------
                    # Model artifact
                    # --------------------------------------

                    model_path = model_artifacts[
                        model_name
                    ]

                    if Path(model_path).exists():

                        mlflow.log_artifact(
                            model_path,
                            artifact_path="model"
                        )

                    # --------------------------------------
                    # Supporting report artifacts
                    # --------------------------------------

                    supporting_artifacts = [
                        "reports/model_comparison.csv",
                        "reports/validation_report.json",
                        "reports/feature_catalog.csv"
                    ]

                    for artifact in supporting_artifacts:

                        if Path(artifact).exists():

                            mlflow.log_artifact(
                                artifact,
                                artifact_path="reports"
                            )

                    stage(
                        f"MLflow run completed: {run_name}"
                    )

            stage(
                "MLflow tracking completed "
                "for both recommendation models"
            )

        except Exception as exc:

            stage(
                f"MLflow tracking warning: {exc}"
            )

        # --------------------------------------------------
        # 10. SAVE PIPELINE EXECUTION LOG
        # --------------------------------------------------

        Path(
            "reports/pipeline_run.txt"
        ).write_text(
            "\n".join(log) + "\n",
            encoding="utf-8"
        )

        # --------------------------------------------------
        # 11. DISPLAY MODEL COMPARISON
        # --------------------------------------------------

        print("\nMODEL COMPARISON")

        print(
            comparison.to_string(index=False)
        )

        print(
            "\nPipeline completed. "
            "See reports/ and models/."
        )

    except Exception as exc:

        error_message = (
            f"PIPELINE FAILED: {type(exc).__name__}: {exc}"
        )

        print("\n" + error_message)

        log.append(error_message)

        Path(
            "reports/pipeline_run.txt"
        ).write_text(
            "\n".join(log) + "\n",
            encoding="utf-8"
        )

        raise


if __name__ == "__main__":
    main()
