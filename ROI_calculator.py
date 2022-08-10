import pandas
import json


# modelop.init
def init() -> None:
    """A function to load cost parameters and field names as global vars."""

    # Load cost constants and field names as global variables
    global COST_OF_ONE_SURPLUS, COST_OF_ONE_DEFICIT, COST_OF_ONE_MODEL_INFERENCE, COST_OF_ONE_HUMAN_INFERENCE
    global ACTUALS_FIELD, MODEL_PRED_FIELD, HUMAN_PRED_FIELD

    model_parameters = json.load(open("./model_parameters.json", "r"))
    COST_OF_ONE_SURPLUS = model_parameters.get("COST_OF_ONE_SURPLUS")
    COST_OF_ONE_DEFICIT = model_parameters.get("COST_OF_ONE_DEFICIT")
    COST_OF_ONE_MODEL_INFERENCE = model_parameters.get("COST_OF_ONE_MODEL_INFERENCE")
    COST_OF_ONE_HUMAN_INFERENCE = model_parameters.get("COST_OF_ONE_HUMAN_INFERENCE")

    ACTUALS_FIELD = model_parameters.get("ACTUALS_FIELD")
    MODEL_PRED_FIELD = model_parameters.get("MODEL_PRED_FIELD")
    HUMAN_PRED_FIELD = model_parameters.get("HUMAN_PRED_FIELD")


# modelop.metrics
def metrics(data: pandas.DataFrame) -> dict:
    """A function to compute ROI metrics on input data

    Args:
        data (pandas.DataFrame): Input data containing model and human forecasts,
            as well as actuals.

    Returns:
        dict: Cost-performance metrics and overall ROI.
    """

    # Calculate over/under predictions by model and humans
    data["model_surplus"] = (data[MODEL_PRED_FIELD] - data[ACTUALS_FIELD]).clip(0)
    data["model_deficit"] = (data[ACTUALS_FIELD] - data[MODEL_PRED_FIELD]).clip(0)

    data["human_surplus"] = (data[HUMAN_PRED_FIELD] - data[ACTUALS_FIELD]).clip(0)
    data["human_deficit"] = (data[ACTUALS_FIELD] - data[HUMAN_PRED_FIELD]).clip(0)

    # Assign a dollar amount to each forecast
    data[["model_surplus", "human_surplus"]] *= COST_OF_ONE_SURPLUS
    data[["model_deficit", "human_deficit"]] *= COST_OF_ONE_DEFICIT

    # Multiply by operation cost
    data[["model_surplus", "model_deficit"]] *= COST_OF_ONE_MODEL_INFERENCE
    data[["human_surplus", "human_deficit"]] *= COST_OF_ONE_HUMAN_INFERENCE

    # Compute totals
    sums = data.sum(axis=0)

    model_surplus_total = sums["model_surplus"]
    model_deficit_total = sums["model_deficit"]
    human_surplus_total = sums["human_surplus"]
    human_deficit_total = sums["human_deficit"]

    model_total_cost = model_surplus_total + model_deficit_total
    human_total_cost = human_surplus_total + human_deficit_total

    return {
        # Top-level key metrics
        "is_model_more_effective": bool(human_total_cost - model_total_cost >= 0),
        "cost_savings_by_model": human_total_cost - model_total_cost,
        # More in-depth view
        "business_value": [
            {
                "test_name": "Actual ROI",
                "test_category": "business_value",
                "test_type": "actual_roi",
                "test_id": "business_value_actual_roi",
                "values": {
                    "model_surplus_total": model_surplus_total,
                    "model_deficit_total": model_deficit_total,
                    "human_surplus_total": human_surplus_total,
                    "human_deficit_total": human_deficit_total,
                    "model_total_cost": model_total_cost,
                    "human_total_cost": human_total_cost,
                    "is_model_more_effective": bool(
                        human_total_cost - model_total_cost >= 0
                    ),
                    "cost_savings_by_model": human_total_cost - model_total_cost,
                },
            }
        ],
    }
