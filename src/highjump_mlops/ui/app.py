import pandas as pd
import streamlit as st
import altair as alt

from highjump_mlops.inference.service import get_athlete_history, list_predictable_athletes, predict_for_athlete


st.set_page_config(
    page_title="High Jump Prediction",
    page_icon="🏃",
    layout="wide",
)


@st.cache_data
def cached_predictable_athletes() -> list[str]:
    return list_predictable_athletes()


@st.cache_data
def cached_prediction(athlete: str) -> dict:
    return predict_for_athlete(athlete)


@st.cache_data
def cached_history(athlete: str) -> pd.DataFrame:
    return get_athlete_history(athlete)


def format_height(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.2f} m"


def format_number(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return str(int(value))


st.title("Men's Outdoor High Jump Prediction")
st.caption(
    "Live ML demo predicting an athlete's next competition mark "
    "from dynamic World Athletics result data."
)

st.markdown("## Select athlete")

try:
    athletes = cached_predictable_athletes()
except Exception as error:
    st.error("Could not load predictable athletes.")
    st.code(str(error))
    st.stop()

if not athletes:
    st.error(
        "No predictable athletes found. Run the feature pipeline and training pipeline first."
    )
    st.stop()

default_athlete = "Mutaz Essa BARSHIM"
default_index = athletes.index(default_athlete) if default_athlete in athletes else 0

selected_athlete = st.selectbox(
    "Athlete",
    athletes,
    index=default_index,
)

try:
    prediction = cached_prediction(selected_athlete)
    history = cached_history(selected_athlete)
except Exception as error:
    st.error(f"Could not create prediction for {selected_athlete}.")
    st.code(str(error))
    st.stop()

predicted_mark = round(float(prediction["prediction_next_competition_mark"]), 2)
latest_mark = round(float(prediction["latest_competition_mark"]), 2)
prediction_delta = predicted_mark - latest_mark

st.markdown("## Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted next competition mark",
        f"{predicted_mark:.2f} m",
        delta=f"{prediction_delta:+.2f} m vs latest",
    )

with col2:
    st.metric(
        "Latest competition mark",
        f"{latest_mark:.2f} m",
    )

with col3:
    st.metric(
        "Latest result rank",
        format_number(prediction["latest_result_rank"]),
    )

st.markdown("## Latest and previous competition context")

context_col1, context_col2, context_col3, context_col4 = st.columns(4)

with context_col1:
    st.metric("Latest date", prediction["latest_date"] or "N/A")

with context_col2:
    st.metric(
        "Previous competition mark",
        format_height(prediction["previous_competition_mark"]),
    )

with context_col3:
    st.metric(
        "Recent 3-competition mean",
        format_height(prediction["recent_3_competition_mark_mean"]),
    )

with context_col4:
    st.metric(
        "Days since latest competition",
        format_number(prediction["days_since_latest_competition"]),
    )

st.markdown("### Latest venue")
st.write(prediction["latest_venue"] or "N/A")

st.markdown("## Recent competition history")

display_history = history.copy()

if "date" in display_history.columns:
    display_history["date"] = pd.to_datetime(display_history["date"], errors="coerce")

chart_history = (
    display_history[["date", "competition_mark"]]
    .dropna()
    .sort_values("date")
    .tail(10)
    .reset_index(drop=True)
)

if not chart_history.empty:
    chart_history["order"] = chart_history.index
    chart_history["label"] = chart_history["date"].dt.strftime("%Y-%m-%d")
    chart_history["height"] = chart_history["competition_mark"].round(2)
    chart_history["series"] = "Actual competition mark"

    latest_chart_row = chart_history.iloc[-1]

    prediction_chart_data = pd.DataFrame(
        {
            "order": [
                int(latest_chart_row["order"]),
                int(latest_chart_row["order"]) + 1,
            ],
            "label": [
                str(latest_chart_row["label"]),
                "Next prediction",
            ],
            "height": [
                float(latest_chart_row["height"]),
                predicted_mark,
            ],
            "series": [
                "Predicted next competition mark",
                "Predicted next competition mark",
            ],
        }
    )

    y_axis = alt.Y(
        "height:Q",
        title="Height (m)",
        scale=alt.Scale(zero=False),
    )

    actual_chart = (
        alt.Chart(chart_history)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "label:N",
                title="Competition",
                sort=alt.SortField("order"),
            ),
            y=y_axis,
            color=alt.Color("series:N", title="Series"),
            tooltip=[
                alt.Tooltip("label:N", title="Date"),
                alt.Tooltip("height:Q", title="Height", format=".2f"),
                alt.Tooltip("series:N", title="Series"),
            ],
        )
    )

    prediction_chart = (
        alt.Chart(prediction_chart_data)
        .mark_line(point=True, strokeDash=[6, 4])
        .encode(
            x=alt.X(
                "label:N",
                title="Competition",
                sort=alt.SortField("order"),
            ),
            y=y_axis,
            color=alt.Color("series:N", title="Series"),
            tooltip=[
                alt.Tooltip("label:N", title="Date"),
                alt.Tooltip("height:Q", title="Height", format=".2f"),
                alt.Tooltip("series:N", title="Series"),
            ],
        )
    )

    chart = (actual_chart + prediction_chart).properties(height=350)

    st.altair_chart(chart, width="stretch")

    st.caption(
        "Solid line: recent actual competition marks. "
        "Dashed segment: model prediction for the next competition."
    )

table_history = display_history.copy()

for column in table_history.columns:
    if pd.api.types.is_float_dtype(table_history[column]):
        table_history[column] = table_history[column].round(2)

table_history = table_history.rename(
    columns={
        "date": "Date",
        "venue": "Venue",
        "competition_mark": "Competition mark (m)",
        "result_rank": "Result rank",
        "previous_competition_mark": "Previous mark (m)",
        "recent_3_competition_mark_mean": "Recent 3-competition mean (m)",
        "recent_5_competition_mark_mean": "Recent 5-competition mean (m)",
        "performance_change_from_previous": "Change from previous (m)",
        "days_since_previous_competition": "Days since previous",
        "season_best_so_far": "Season best so far (m)",
        "target_next_competition_mark": "Actual next competition mark (m)",
    }
)

st.dataframe(
    table_history,
    width="stretch",
    hide_index=True,
)

st.markdown("## Model evaluation")

metrics = prediction.get("metrics", {})

model_name = prediction.get("model_name") or metrics.get("model_name") or "N/A"
model_type = prediction.get("model_type") or metrics.get("model_type") or "N/A"

model_col1, model_col2 = st.columns(2)

with model_col1:
    st.metric("Deployed model", model_name)

with model_col2:
    st.metric("Model type", model_type)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    mae = metrics.get("mae")
    st.metric("MAE", "N/A" if mae is None else f"{mae:.4f} m")

with metric_col2:
    rmse = metrics.get("rmse")
    st.metric("RMSE", "N/A" if rmse is None else f"{rmse:.4f} m")

with metric_col3:
    training_rows = metrics.get("training_rows")
    st.metric("Training rows", "N/A" if training_rows is None else training_rows)

with metric_col4:
    test_rows = metrics.get("test_rows")
    st.metric("Test rows", "N/A" if test_rows is None else test_rows)

st.caption(
    "Target: next competition mark. "
    "The model predicts the height an athlete may clear in his next available competition result."
)