import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from collections import Counter, defaultdict
import random

# =============================
# Config
# =============================
st.set_page_config(page_title="Lottery Predictor", layout="wide")

GAME_LINKS = {
    "Star 6/50": "https://docs.google.com/spreadsheets/d/e/2PACX-1vS8b63hsr-rj0VCeviGFy36wVCEGRvKlUY98RSUMZtPyC6s9NxiUwuzA_ZvJ4pcujHoSpFhBCH4m2pJ/pub?gid=1694692461&single=true&output=csv",
    "Power 6/55": "https://docs.google.com/spreadsheets/d/e/2PACX-1vS8b63hsr-rj0VCeviGFy36wVCEGRvKlUY98RSUMZtPyC6s9NxiUwuzA_ZvJ4pcujHoSpFhBCH4m2pJ/pub?gid=1257988366&single=true&output=csv",
    "Supreme 6/58": "https://docs.google.com/spreadsheets/d/e/2PACX-1vS8b63hsr-rj0VCeviGFy36wVCEGRvKlUY98RSUMZtPyC6s9NxiUwuzA_ZvJ4pcujHoSpFhBCH4m2pJ/pub?gid=543531444&single=true&output=csv",
    "Magnum Life": "https://docs.google.com/spreadsheets/d/e/2PACX-1vS8b63hsr-rj0VCeviGFy36wVCEGRvKlUY98RSUMZtPyC6s9NxiUwuzA_ZvJ4pcujHoSpFhBCH4m2pJ/pub?gid=994502155&single=true&output=csv"
}

EXPECTED_COLS = {
    "Star 6/50": ["DrawDate"] + [f"DrawnNo{i}" for i in range(1,7)] + ["BonusNo"],
    "Power 6/55": ["DrawDate"] + [f"DrawnNo{i}" for i in range(1,7)],
    "Supreme 6/58": ["DrawDate"] + [f"DrawnNo{i}" for i in range(1,7)],
    "Magnum Life": ["Date"] + [f"Winning Number {i}" for i in range(1,9)] + ["Bonus Number 1", "Bonus Number 2"]
}

# =============================
# 🎛️ Weight Sliders
# =============================
st.sidebar.header("⚖️ Model Weights for Hybrid Ensemble")
w_rf = st.sidebar.slider("Random Forest Weight", 0.0, 5.0, 1.0, 0.1)
w_log = st.sidebar.slider("Logistic Regression Weight", 0.0, 5.0, 1.0, 0.1)
w_nb = st.sidebar.slider("Naive Bayes Weight", 0.0, 5.0, 1.0, 0.1)
w_mp = st.sidebar.slider("Markov Pairwise Weight", 0.0, 5.0, 1.0, 0.1)
w_mf = st.sidebar.slider("Markov Frequency Weight", 0.0, 5.0, 1.0, 0.1)

MODEL_WEIGHTS = {
    "Random Forest": w_rf,
    "Logistic": w_log,
    "Naive Bayes": w_nb,
    "Markov (Pairwise)": w_mp,
    "Markov (Frequency)": w_mf
}

# =============================
# Helpers
# =============================
def load_csv_from_url(url):
    r = requests.get(url)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def fix_magnum_headers(df):
    cols = list(df.columns)
    new_cols = []
    win_counter = 1
    bonus_counter = 1
    passed_bonus = False

    for c in cols:
        if "Winning Number" in c:
            new_cols.append(f"Winning Number {win_counter}")
            win_counter += 1
        elif c.strip().isdigit() and not passed_bonus:
            new_cols.append(f"Winning Number {win_counter}")
            win_counter += 1
        elif "Bonus Number" in c:
            new_cols.append(f"Bonus Number {bonus_counter}")
            bonus_counter += 1
            passed_bonus = True
        elif c.strip().isdigit() and passed_bonus:
            new_cols.append(f"Bonus Number {bonus_counter}")
            bonus_counter += 1
        else:
            new_cols.append(c)
    df.columns = new_cols

    if "Bonus Number 2" not in df.columns:
        df["Bonus Number 2"] = None

    return df

def validate_columns(game, df):
    expected = EXPECTED_COLS[game]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        st.warning(f"⚠️ {game} format changed. Missing columns: {missing}")
    else:
        st.success(f"{game} columns validated ✔️")

def enforce_filters(df, date_col, game):
    if game in ["Star 6/50", "Power 6/55", "Supreme 6/58"]:
        df[date_col] = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")
    elif game == "Magnum Life":
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df = df.dropna(subset=[date_col])

    if df.empty:
        return df

    min_year, max_year = int(df[date_col].dt.year.min()), int(df[date_col].dt.year.max())

    year = st.sidebar.slider(
        f"Filter Year ({game})", min_value=min_year, max_value=max_year,
        value=(min_year, max_year), key=f"{game}_year"
    )
    month = st.sidebar.multiselect(
        f"Filter Month ({game})", list(range(1, 13)), default=list(range(1, 13)),
        key=f"{game}_month"
    )
    weekday = st.sidebar.multiselect(
        f"Filter Weekday ({game})", list(range(0, 7)), default=list(range(0, 7)),
        key=f"{game}_weekday"
    )

    mask = (
        df[date_col].dt.year.between(year[0], year[1])
        & df[date_col].dt.month.isin(month)
        & df[date_col].dt.weekday.isin(weekday)
    )
    return df.loc[mask]

def train_simple_models(X, y):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=1, random_state=42),
        "Logistic": LogisticRegression(max_iter=1),
        "NaiveBayes": GaussianNB()
    }
    preds = {}
    for name, model in models.items():
        try:
            model.fit(X, y)
            preds[name] = model.predict_proba(X)[-1]
        except Exception:
            preds[name] = np.zeros(len(np.unique(y)))
    return preds

def markov_prediction(numbers, n_numbers, k=6, method="pairwise"):
    if method == "pairwise":
        transitions = defaultdict(Counter)
        for draw in numbers:
            for i in range(len(draw)-1):
                transitions[draw[i]][draw[i+1]] += 1
        current = random.choice(numbers)[0]
        result = [current]
        for _ in range(k-1):
            next_states = transitions[current]
            next_num = max(next_states, key=next_states.get) if next_states else random.randint(1, n_numbers)
            result.append(next_num)
            current = next_num
        return sorted(result)
    else:
        flat = [num for draw in numbers for num in draw]
        counter = Counter(flat)
        return sorted([x for x, _ in counter.most_common(k)])

def ensemble_predictions(preds_dict, strategy="vote", n_numbers=50, k=6):
    all_preds = []
    for probs in preds_dict.values():
        top = np.argsort(probs)[-k:]
        all_preds.append(top)
    if strategy=="vote":
        flat = [num for arr in all_preds for num in arr]
        counter = Counter(flat)
        return sorted([int(x)+1 for x, _ in counter.most_common(k)])
    else:
        avg_probs = np.mean(list(preds_dict.values()), axis=0)
        return sorted([int(x)+1 for x in np.argsort(avg_probs)[-k:]])

def constrain_predictions(preds, game):
    ranges = {"Star 6/50": 50, "Power 6/55": 55, "Supreme 6/58": 58, "Magnum Life": 36}
    max_num = ranges[game]
    k = 6 if game != "Magnum Life" else 8

    seen, unique = set(), []
    try:
        iterator = np.asarray(preds).ravel()
    except Exception:
        iterator = preds

    for item in iterator:
        if isinstance(item, (list, tuple, np.ndarray)):
            for sub in np.asarray(item).ravel():
                try:
                    val = int(sub) % max_num + 1
                except Exception:
                    continue
                if val not in seen:
                    seen.add(val); unique.append(val)
                    if len(unique) == k: break
            if len(unique) == k: break
        else:
            try:
                val = int(item) % max_num + 1
            except Exception:
                continue
            if val not in seen:
                seen.add(val); unique.append(val)
                if len(unique) == k: break

    while len(unique) < k:
        val = random.randint(1, max_num)
        if val not in seen:
            unique.append(val); seen.add(val)

    return sorted(unique)

def hybrid_ensemble_weighted(preds_dict, numbers, n_numbers=50, k=6):
    weighted_counter = Counter()

    # ML Models
    ml_vote = ensemble_predictions(preds_dict, "vote", n_numbers, k)
    for num in ml_vote: weighted_counter[num] += MODEL_WEIGHTS["Random Forest"]

    log_vote = ensemble_predictions({"log":preds_dict["Logistic"]}, "vote", n_numbers, k)
    for num in log_vote: weighted_counter[num] += MODEL_WEIGHTS["Logistic"]

    nb_vote = ensemble_predictions({"nb":preds_dict["NaiveBayes"]}, "vote", n_numbers, k)
    for num in nb_vote: weighted_counter[num] += MODEL_WEIGHTS["Naive Bayes"]

    # Markov
    mp = markov_prediction(numbers, n_numbers, k, "pairwise")
    for num in mp: weighted_counter[num] += MODEL_WEIGHTS["Markov (Pairwise)"]

    mf = markov_prediction(numbers, n_numbers, k, "freq")
    for num in mf: weighted_counter[num] += MODEL_WEIGHTS["Markov (Frequency)"]

    # Pick top k by weight
    final = [num for num, _ in weighted_counter.most_common(k)]

    # pad if needed
    while len(final) < k:
        cand = random.randint(1, n_numbers)
        if cand not in final: final.append(cand)

    return sorted(final[:k])

# =============================
# App
# =============================
st.title("Lottery Predictor – Malaysia (Toto & Magnum Life)")

tab1, tab2, tab3, tab4 = st.tabs(["Magnum Life", "Star 6/50", "Power 6/55", "Supreme 6/58"])

for game, tab in zip(GAME_LINKS.keys(), [tab4, tab2, tab3, tab1]):
    with tab:
        st.header(game)

        df = load_csv_from_url(GAME_LINKS[game])
        if game == "Magnum Life": df = fix_magnum_headers(df)
        validate_columns(game, df)

        date_col = "Date" if game=="Magnum Life" else "DrawDate"
        df = enforce_filters(df, date_col, game)

        if df.empty:
            st.warning("No draws available after applying filters."); continue

        st.write(f"Latest draw: {df[date_col].max().date()}")
        st.download_button("Download filtered data", df.to_csv(index=False), file_name=f"{game}.csv")

        if game=="Star 6/50":
            drawn_cols, bonus_col, n_numbers, k = [f"DrawnNo{i}" for i in range(1,7)], "BonusNo", 50, 6
        elif game=="Power 6/55":
            drawn_cols, bonus_col, n_numbers, k = [f"DrawnNo{i}" for i in range(1,7)], None, 55, 6
        elif game=="Supreme 6/58":
            drawn_cols, bonus_col, n_numbers, k = [f"DrawnNo{i}" for i in range(1,7)], None, 58, 6
        else:
            drawn_cols, bonus_col, n_numbers, k = [f"Winning Number {i}" for i in range(1,9)], ["Bonus Number 1", "Bonus Number 2"], 36, 8

        numbers = df[drawn_cols].values.tolist()
        if len(numbers) < 50:
            st.warning("Not enough draws for stable prediction (min 50 required)."); continue

        X, y = np.array(numbers), np.arange(len(numbers))
        preds_dict = train_simple_models(X, y)

        st.subheader("Predicted Numbers")
        raw_predictions = {
            "Random Forest": ensemble_predictions({"rf":preds_dict["RandomForest"]}, "vote", n_numbers, k),
            "Logistic": ensemble_predictions({"log":preds_dict["Logistic"]}, "vote", n_numbers, k),
            "Naive Bayes": ensemble_predictions({"nb":preds_dict["NaiveBayes"]}, "vote", n_numbers, k),
            "Markov (Pairwise)": markov_prediction(numbers, n_numbers, k, "pairwise"),
            "Markov (Frequency)": markov_prediction(numbers, n_numbers, k, "freq"),
            "Ensemble (Vote)": ensemble_predictions(preds_dict, "vote", n_numbers, k),
            "Ensemble (Prob Weighted)": ensemble_predictions(preds_dict, "prob", n_numbers, k),
            "Hybrid Ensemble (Weighted)": hybrid_ensemble_weighted(preds_dict, numbers, n_numbers, k)
        }

        predictions = {name: constrain_predictions(pred, game) for name, pred in raw_predictions.items()}
        for model_name, pred in predictions.items():
            st.write(f"{model_name}: {pred}")

        # Bonus handling
        bonus_predictions = None
        if bonus_col is not None:
            st.subheader("Bonus Predictions")
            if isinstance(bonus_col, str):
                bonus_vals = df[bonus_col].dropna().astype(int).tolist()
                if bonus_vals:
                    bonus_predictions = [Counter(bonus_vals).most_common(1)[0][0]]
                    st.write(f"Most frequent Bonus Prediction: {bonus_predictions}")
            elif isinstance(bonus_col, list):
                all_bonuses = []
                for col in bonus_col:
                    if col in df.columns:
                        all_bonuses += df[col].dropna().astype(int).tolist()
                if all_bonuses:
                    counts = Counter(all_bonuses)
                    deduped = list(dict.fromkeys([num for num, _ in counts.most_common(len(bonus_col)*2)]))
                    needed, max_bonus = len(bonus_col), {"Star 6/50": 50, "Power 6/55": 55, "Supreme 6/58": 58, "Magnum Life": 36}[game]
                    while len(deduped) < needed:
                        cand = random.randint(1, max_bonus)
                        if cand not in deduped: deduped.append(cand)
                    bonus_predictions = sorted(deduped[:needed])
                    st.write(f"Predicted Bonus Numbers: {bonus_predictions}")

        export_data = []
        for model_name, pred in predictions.items():
            export_data.append({
                "Game": game,
                "Model": model_name,
                "Predicted Numbers": ", ".join(map(str, pred)),
                "Bonus Prediction": ", ".join(map(str, bonus_predictions)) if bonus_predictions else "-"
            })
        st.download_button("Download Predictions CSV", pd.DataFrame(export_data).to_csv(index=False), file_name=f"{game}_predictions.csv")
