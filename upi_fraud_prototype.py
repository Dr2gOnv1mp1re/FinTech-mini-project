"""
Prototype: AI-based real-time fraud detection for UPI payments.
Uses a SYNTHETIC dataset (no real customer data) that mimics common Indian UPI scam patterns.
Run: python upi_fraud_prototype.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix, roc_curve)
from sklearn.inspection import permutation_importance

rng = np.random.default_rng(42)
N = 200_000
FRAUD_RATE = 0.008  # 0.8 % - heavily imbalanced, like real payment streams

# ---------------------------------------------------------------- 1. synthetic data
is_fraud = rng.random(N) < FRAUD_RATE
typ = np.where(is_fraud, rng.choice(["collect_scam", "vishing_remote", "mule_transfer"], N, p=[.35, .4, .25]), "legit")

# legit baseline for everything, then each scam TYPE overrides its own signature (non-linear, OR-of-AND pattern)
p_legit_h = np.array([1,1,1,1,1,2,3,5,7,8,8,8,8,8,7,7,7,8,9,9,8,6,4,2], float)
hour = rng.choice(24, N, p=p_legit_h / p_legit_h.sum())
user_avg = rng.lognormal(6.2, 0.8, N)                                  # user's typical ticket size (INR)
amount = user_avg * rng.lognormal(0, 0.8, N)
new_payee = (rng.random(N) < 0.22).astype(int)
payee_age_days = rng.exponential(350, N)
velocity_1h = rng.poisson(1.0, N)
device_change = (rng.random(N) < 0.05).astype(int)
sim_swap_7d = (rng.random(N) < 0.015).astype(int)
geo_km = rng.exponential(55, N)
collect_req = (rng.random(N) < 0.07).astype(int)
remote_app = (rng.random(N) < 0.015).astype(int)
acct_age = rng.exponential(1500, N)
failed_pin = rng.poisson(0.2, N)
merchant = rng.choice(["P2P", "grocery", "food", "bills", "travel", "gaming"], N, p=[.4, .2, .12, .13, .08, .07])

hard = rng.random(N) < 0.15                       # 15 % of fraud is camouflaged (looks normal)
cs = (typ == "collect_scam") & ~hard
vr = (typ == "vishing_remote") & ~hard
mu = (typ == "mule_transfer") & ~hard
# 1) collect-request scam: victim approves a fake collect request
collect_req[cs] = 1; new_payee[cs] = (rng.random(cs.sum()) < 0.9); amount[cs] = user_avg[cs] * rng.lognormal(0.9, 0.7, cs.sum())
# 2) vishing + remote-access app: victim guided by fake "bank officer"
remote_app[vr] = (rng.random(vr.sum()) < 0.75); device_change[vr] = (rng.random(vr.sum()) < 0.35)
hour[vr] = rng.choice(np.r_[np.arange(0, 6), np.arange(20, 24)], vr.sum()); amount[vr] = user_avg[vr] * rng.lognormal(1.5, 0.8, vr.sum())
failed_pin[vr] = rng.poisson(1.0, vr.sum())
# 3) mule / account-takeover: fast cash-out to young payee accounts
velocity_1h[mu] = rng.poisson(3.5, mu.sum()); payee_age_days[mu] = rng.exponential(12, mu.sum()); new_payee[mu] = 1
sim_swap_7d[mu] = (rng.random(mu.sum()) < 0.25); geo_km[mu] = rng.exponential(250, mu.sum())
amount = np.clip(amount, 1, 100000)

df = pd.DataFrame(dict(
    amount=amount, amount_ratio=amount / user_avg, hour=hour, is_night=((hour < 6)).astype(int),
    new_payee=new_payee, payee_age_days=payee_age_days, velocity_1h=velocity_1h,
    device_change=device_change, sim_swap_7d=sim_swap_7d, geo_km=geo_km,
    collect_request=collect_req, remote_access_app=remote_app, account_age_days=acct_age,
    failed_pin=failed_pin, merchant=merchant, fraud=is_fraud.astype(int)))
df = pd.get_dummies(df, columns=["merchant"], dtype=int)
df.to_csv("synthetic_upi_transactions.csv", index=False)

# ---------------------------------------------------------------- 2. stratified train/validation/test split
# The synthetic data has no timestamps, so a chronological split is not meaningful.
X, y = df.drop(columns="fraud"), df["fraud"]
X_trainval, Xte, y_trainval, yte = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
Xtr, Xval, ytr, yval = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=43
)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=1),
    "Gradient Boosting (HGB)": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, class_weight="balanced", random_state=1),
}
res, curves, probs = {}, {}, {}
validation_pr_auc = {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    pval = m.predict_proba(Xval)[:, 1]
    validation_pr_auc[name] = average_precision_score(yval, pval)
    prec_val, rec_val, thresholds = precision_recall_curve(yval, pval)
    f1s_val = 2 * prec_val * rec_val / (prec_val + rec_val + 1e-12)
    best = np.argmax(f1s_val[:-1])
    t = thresholds[best]

    p = m.predict_proba(Xte)[:, 1]
    probs[name] = p
    prec, rec, _ = precision_recall_curve(yte, p)
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    fpr_, tpr_, _ = roc_curve(yte, p)
    res[name] = dict(roc_auc=roc_auc_score(yte, p), pr_auc=average_precision_score(yte, p),
                     threshold=float(t), precision=precision_score(yte, pred), recall=recall_score(yte, pred),
                     f1=f1_score(yte, pred), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                     fpr=fp / (fp + tn), validation_pr_auc=validation_pr_auc[name])
    curves[name] = (prec, rec)

# ---------------------------------------------------------------- 3. figures
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
cols = {"Logistic Regression": "#8b95a5", "Random Forest": "#2a9d8f", "Gradient Boosting (HGB)": "#e76f51"}
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1))
for n in models:
    ax[0].plot(*curves[n][::-1], label=f"{n} (AP={res[n]['pr_auc']:.2f})", color=cols[n], lw=1.8)
ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision"); ax[0].set_title("Precision-Recall curves (test set)")
ax[0].legend(fontsize=7, loc="lower left"); ax[0].grid(alpha=.25)
best_name = max(validation_pr_auc, key=validation_pr_auc.get)
bm = models[best_name]
sub = np.random.default_rng(0).choice(len(Xte), 40000, replace=False)
pi = permutation_importance(bm, Xte.iloc[sub], yte.iloc[sub], scoring="average_precision", n_repeats=3, random_state=0, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values().tail(8)
ax[1].barh(imp.index, imp.values, color="#264653")
ax[1].set_title(f"Top features ({best_name})"); ax[1].set_xlabel("Drop in PR-AUC when shuffled")
ax[1].grid(alpha=.25, axis="x")
plt.tight_layout(); plt.savefig("fig_results.png", dpi=200); plt.close()

# ---- architecture diagram
def box(ax, x, y, w, h, text, fc, ec="#222", fs=7.2, bold=False):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, fontweight="bold" if bold else "normal")
def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#333"), zorder=1)

fig, ax = plt.subplots(figsize=(7.6, 4.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")
box(ax, 1, 40, 14, 12, "UPI app\n(PhonePe, GPay,\nPaytm, BHIM)", "#dfe7fd")
box(ax, 19, 40, 13, 12, "PSP Bank\n(Payer)", "#dfe7fd")
box(ax, 36, 40, 13, 12, "NPCI UPI\nSwitch", "#dfe7fd")
box(ax, 53, 40, 13, 12, "Beneficiary\nBank", "#dfe7fd")
for a, b in ((15, 19), (32, 36), (49, 53)): arrow(ax, a, 46, b, 46)
ax.text(40, 55, "Existing payment rails", fontsize=8, style="italic", ha="center")
box(ax, 1, 21, 17, 13, "Event stream\n(Kafka)", "#fff3cd")
box(ax, 22, 21, 19, 13, "Feature store\n(Redis)\nvelocity, baseline", "#fff3cd")
box(ax, 45, 21, 19, 13, "ML scoring\nGBM + rules\n+ graph score", "#ffd6cc", bold=True)
box(ax, 68, 21, 17, 13, "Decision\nengine: allow /\nstep-up / block", "#d8f3dc", bold=True, fs=7)
for a, b in ((18, 22), (41, 45), (64, 68)): arrow(ax, a, 27.5, b, 27.5)
arrow(ax, 25.5, 40, 9.5, 34); ax.text(19, 35.5, "real-time copy", fontsize=6.5)
arrow(ax, 82, 34, 85, 40.5); ax.text(86, 36.5, "action <100 ms", fontsize=6.5)
box(ax, 77, 40, 21, 12, "Alert, MPIN\nre-auth,\ncooling-off", "#d8f3dc", fs=7)
box(ax, 1, 3, 20, 12, "Data lake\n(labelled\noutcomes)", "#e9ecef")
box(ax, 26, 3, 20, 12, "Model training\n& back-testing\n(MLflow)", "#e9ecef")
box(ax, 51, 3, 21, 12, "Governance:\nSHAP, bias and\ndrift monitoring", "#e9ecef")
box(ax, 77, 3, 21, 12, "Analyst console\n& regulatory\nreporting", "#e9ecef")
arrow(ax, 21, 9, 26, 9); arrow(ax, 46, 9, 51, 9); arrow(ax, 72, 9, 77, 9)
arrow(ax, 36, 15, 52, 21); ax.text(37, 17, "deploy model", fontsize=6.5)
arrow(ax, 87, 15, 80, 21); ax.text(86, 18, "feedback", fontsize=6.5)
ax.text(50, 0, "Fig. 1  Proposed real-time UPI fraud detection architecture", ha="center", fontsize=7.5, style="italic")
plt.savefig("fig_architecture.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- workflow cost simulation at the best model's threshold
n_te = len(yte); avg_fraud_amt = float(df.loc[y == 1, "amount"].mean())
r = res[best_name]
saved = r["tp"] * avg_fraud_amt; missed = r["fn"] * avg_fraud_amt
summary = dict(res=res, best=best_name, n_train=len(Xtr), n_validation=len(Xval), n_test=n_te,
               fraud_train=int(ytr.sum()), fraud_validation=int(yval.sum()), fraud_test=int(yte.sum()),
               selection_metric="validation PR-AUC",
               avg_fraud_amt=avg_fraud_amt, saved=saved, missed=missed,
               alerts_per_10k=float((r["tp"] + r["fp"]) / n_te * 10000),
               top_features=list(imp.index[::-1]))
json.dump(summary, open("results.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
