import pandas as pd
import streamlit as st
import altair as alt

from highjump_mlops.inference import get_athlete_history, list_predictable_athletes, predict_for_athlete


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


st.title("Men's Outdoor High Jump Prediction")
st.caption(
    "Live ML demo predicting an athlete's next season-best height "
    "from World Athletics season toplist data."
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

st.markdown("## Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted next season best",
        f"{prediction['prediction_next_season_best']:.2f} m",
    )

with col2:
    st.metric(
        "Latest season best",
        f"{prediction['latest_season_best']:.2f} m",
    )

with col3:
    st.metric(
        "Latest season rank",
        prediction["latest_season_rank"],
    )

st.markdown("## Latest athlete context")

context_col1, context_col2, context_col3, context_col4 = st.columns(4)

with context_col1:
    st.metric("Latest year", prediction["latest_year"])

with context_col2:
    previous_best = prediction["previous_season_best"]
    st.metric(
        "Previous season best",
        "N/A" if previous_best is None else f"{previous_best:.2f} m",
    )

with context_col3:
    performance_change = prediction["performance_change"]
    st.metric(
        "Performance change",
        "N/A" if performance_change is None else f"{performance_change:+.2f} m",
    )

with context_col4:
    st.metric(
        "Days since season best",
        prediction["days_since_season_best"],
    )

st.markdown("## Recent athlete history")

display_history = history.copy()

for column in display_history.columns:
    if display_history[column].dtype == "float64":
        display_history[column] = display_history[column].round(2)

chart_history = (
    history[["year", "season_best"]]
    .dropna()
    .sort_values("year")
    .copy()
)

if not chart_history.empty:
    historical_chart_data = chart_history.rename(
        columns={"season_best": "height"}
    )
    historical_chart_data["series"] = "Historical season best"

    latest_actual_row = chart_history.sort_values("year").iloc[-1]
    predicted_year = prediction["latest_year"] + 1

    prediction_chart_data = pd.DataFrame(
        {
            "year": [
                int(latest_actual_row["year"]),
                predicted_year,
            ],
            "height": [
                float(latest_actual_row["season_best"]),
                prediction["prediction_next_season_best"],
            ],
            "series": [
                "Predicted next season best",
                "Predicted next season best",
            ],
        }
    )

    y_axis = alt.Y(
        "height:Q",
        title="Season best height (m)",
        scale=alt.Scale(zero=False),
    )

    historical_chart = (
        alt.Chart(historical_chart_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=y_axis,
            color=alt.Color("series:N", title="Series"),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("height:Q", title="Season best", format=".2f"),
                alt.Tooltip("series:N", title="Series"),
            ],
        )
    )

    prediction_chart = (
        alt.Chart(prediction_chart_data)
        .mark_line(point=True, strokeDash=[6, 4])
        .encode(
            x=alt.X("year:O", title="Year"),
            y=y_axis,
            color=alt.Color("series:N", title="Series"),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("height:Q", title="Height", format=".2f"),
                alt.Tooltip("series:N", title="Series"),
            ],
        )
    )

    chart = (historical_chart + prediction_chart).properties(height=350)

    st.altair_chart(chart, width="stretch")

    st.caption(
        "Solid line: historical season-best heights. "
        "Dashed segment: model prediction for the next season."
    )

display_history = display_history.rename(
    columns={
        "year": "Year",
        "season_rank": "Season rank",
        "season_best": "Season best (m)",
        "previous_season_best": "Previous season best (m)",
        "recent_3_season_best_mean": "Recent 3-season mean (m)",
        "performance_change": "Performance change (m)",
        "days_since_season_best": "Days since season best",
    }
)

st.dataframe(
    display_history,
    width="stretch",
    hide_index=True,
)


st.markdown("## Model evaluation")

metrics = prediction.get("metrics", {})

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    mae = metrics.get("mae")
    st.metric("MAE", "N/A" if mae is None else f"{mae:.3f} m")

with metric_col2:
    rmse = metrics.get("rmse")
    st.metric("RMSE", "N/A" if rmse is None else f"{rmse:.3f} m")

with metric_col3:
    training_rows = metrics.get("training_rows")
    st.metric("Training rows", "N/A" if training_rows is None else training_rows)

with metric_col4:
    test_rows = metrics.get("test_rows")
    st.metric("Test rows", "N/A" if test_rows is None else test_rows)
