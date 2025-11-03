import pandas as pd
import numpy as np
import cvxpy as cp

# Load data
df = pd.read_csv("movie_data.csv")
saved_names = df.iloc[:, [0]].copy()
df = df.drop(df.columns[[0]], axis=1)

def matrix_completion(df_obs: pd.DataFrame) -> pd.DataFrame:
    # convert to numpy
    M_obs = df_obs.values
    mask = ~np.isnan(M_obs)

    m, n = df_obs.shape

    # CVXPY optimization variable
    X = cp.Variable((m, n))

    # constraints: match observed entries
    constraints = [X[mask] == M_obs[mask], X >= 0, X <= 10]

    # minimize nuclear norm (low-rank approximation)
    prob = cp.Problem(cp.Minimize(cp.normNuc(X)), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    # convert back to DataFrame
    M_completed = np.rint(X.value)
    df_completed = pd.DataFrame(M_completed, index=df_obs.index, columns=df_obs.columns)

    return df_completed

df_completed = matrix_completion(df)
saved_names.join(df_completed).to_csv("movie_data_completed.csv", index=False)


def topk_recommendations(df_obs: pd.DataFrame,
                         df_completed: pd.DataFrame,
                         k: int = 5,
                         min_score: float | None = None) -> pd.DataFrame:
    # entries that were NaN originally
    missing = df_obs.isna().values

    # exclude movies the student already rated by setting them to -inf
    scores = df_completed.values.copy()
    scores[~missing] = -np.inf

    # apply a minimum predicted score threshold
    if min_score is not None:
        scores[scores < min_score] = -np.inf

    # for each student (row), get indices of top-k columns by score
    m, n = scores.shape
    k_eff = min(k, n)  # guard if k > number of movies
    # use argpartition then sort those k by score
    topk_idx = np.argpartition(-scores, kth=np.clip(k_eff-1, 0, n-1), axis=1)[:, :k_eff]

    # sort within the chosen k
    row_indices = np.arange(m)[:, None]
    topk_sorted = topk_idx[row_indices, np.argsort(-scores[row_indices, topk_idx])]

    # build long-form (student, movie, score, rank)
    m = df_obs.shape[0]
    students = np.repeat(df_obs.index.values + 1, k_eff)
    movies = df_obs.columns.values
    movies_flat = movies[topk_sorted.ravel()]
    scores_flat = scores[row_indices, topk_sorted].ravel()

    # filter out rows where everything was -inf (no valid recs)
    valid = np.isfinite(scores_flat)
    out = pd.DataFrame({
        "students": students[valid],
        "movie": movies_flat[valid],
        "score": scores_flat[valid],
    })

    # add rank within each students
    out["rank"] = out.groupby("students")["score"].rank(ascending=False, method="first").astype(int)
    out = out.sort_values(["students", "rank"]).reset_index(drop=True)
    return out

recs = topk_recommendations(df, df_completed, k=5, min_score=7.5)  # e.g., only recommend if at least 7.5
recs.to_csv("movie_data_recommendations.csv", index=False)
