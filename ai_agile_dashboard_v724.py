import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                               RandomForestClassifier, VotingClassifier,
                               AdaBoostClassifier, IsolationForest)
from sklearn.linear_model import LogisticRegression, LinearRegression
import os, pathlib, subprocess

# ── Spark setup ────────────────────────────────────────────────────────────
def _find_java():
    if os.environ.get("JAVA_HOME"): return os.environ["JAVA_HOME"]
    for base in [r"C:\Program Files\Eclipse Adoptium", r"C:\Program Files\Java",
                 "/usr/lib/jvm", "/usr/local/opt", "/Library/Java/JavaVirtualMachines"]:
        p = pathlib.Path(base)
        if p.exists():
            for child in sorted(p.iterdir(), reverse=True):
                jb = child / "bin" / ("java.exe" if os.name == "nt" else "java")
                if not jb.exists(): jb = child / "Contents" / "Home" / "bin" / "java"
                if jb.exists(): return str(jb.parent.parent)
    return None

_jh = _find_java()
if _jh:
    os.environ["JAVA_HOME"] = _jh
    os.environ["PATH"] = str(pathlib.Path(_jh)/"bin") + os.pathsep + os.environ.get("PATH","")
os.environ.setdefault("PYSPARK_PYTHON", "python3")
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

@st.cache_resource
def get_spark():
    if not SPARK_AVAILABLE: return None
    try:
        spark = (SparkSession.builder
                 .appName("AgileAI").master("local[*]")
                 .config("spark.driver.memory", "2g")
                 .config("spark.sql.shuffle.partitions", "4")
                 .config("spark.ui.enabled", "false")
                 .config("spark.driver.host", "127.0.0.1")
                 .config("spark.driver.bindAddress", "127.0.0.1")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        return spark
    except: return None

# ── Feature engineering ────────────────────────────────────────────────────
def engineer_features(pdf, spark=None):
    if spark:
        try:
            sdf = spark.createDataFrame(pdf)
            sdf = sdf.withColumn("Velocity_Efficiency",
                F.when(F.col("Planned_Story_Points_Sprint")>0,
                       F.col("Historical_Velocity")/F.col("Planned_Story_Points_Sprint")).otherwise(1.0))
            sdf = sdf.withColumn("Completion_Gap",
                F.col("Planned_Story_Points_Sprint")-F.col("Completed_Story_Points"))
            sdf = sdf.withColumn("Blocker_Severity",
                F.col("Blocked_Stories")*F.when(F.col("Days_Remaining_Sprint")>0,
                    1/F.col("Days_Remaining_Sprint")).otherwise(1.0))
            sdf = sdf.withColumn("Scope_Pressure",
                F.when(F.col("Planned_Story_Points_Sprint")>0,
                       F.col("Scope_Change")/F.col("Planned_Story_Points_Sprint")).otherwise(0.0))
            sdf = sdf.withColumn("Sprint_Momentum",
                F.when(F.col("Historical_Velocity")>0,
                       F.col("Completed_Story_Points")/F.col("Historical_Velocity")).otherwise(0.0))
            sdf = sdf.withColumn("Recovery_Index",
                F.when((F.col("Planned_Story_Points_Sprint")-F.col("Completed_Story_Points")>0)&
                       (F.col("Days_Remaining_Sprint")>0),
                       (F.col("Historical_Velocity")*F.col("Days_Remaining_Sprint")/10)/
                       (F.col("Planned_Story_Points_Sprint")-F.col("Completed_Story_Points"))
                ).otherwise(1.0))
            sdf = sdf.withColumn("Workload_Stress",
                (F.col("Current_Workload_Percent")/100)*F.col("Consecutive_Overloads"))
            return sdf.toPandas().fillna(0)
        except: pass
    df = pdf.copy()
    df["Velocity_Efficiency"] = (df["Historical_Velocity"]/df["Planned_Story_Points_Sprint"].replace(0,1)).clip(0,3)
    df["Completion_Gap"]      = df["Planned_Story_Points_Sprint"] - df["Completed_Story_Points"]
    df["Blocker_Severity"]    = df["Blocked_Stories"]*(1/df["Days_Remaining_Sprint"].replace(0,1).abs())
    df["Scope_Pressure"]      = (df["Scope_Change"]/df["Planned_Story_Points_Sprint"].replace(0,1)).clip(-1,2)
    df["Sprint_Momentum"]     = (df["Completed_Story_Points"]/df["Historical_Velocity"].replace(0,1)).clip(0,2)
    df["Recovery_Index"]      = ((df["Historical_Velocity"]*df["Days_Remaining_Sprint"]/10)/
                                  (df["Planned_Story_Points_Sprint"]-df["Completed_Story_Points"]).replace(0,.001)).clip(0,5)
    df["Workload_Stress"]     = (df["Current_Workload_Percent"]/100)*df.get("Consecutive_Overloads",pd.Series(0,index=df.index))
    return df.fillna(0)

SPARK_FEATS = ["Velocity_Efficiency","Completion_Gap","Blocker_Severity",
               "Scope_Pressure","Sprint_Momentum","Recovery_Index","Workload_Stress"]

# ── Cached training — obj tabs (no Spark feats, max speed) ─────────────────
@st.cache_data(show_spinner="⚡ Training models...")
def train_objectives(df):
    """Train all 5 objectives. Metrics stored here — never recomputed."""
    import warnings; warnings.filterwarnings("ignore")
    R = {}

    # Obj 1: Sprint — Decision Tree
    try:
        f = [c for c in ["Planned_Story_Points_Sprint","Completed_Story_Points",
             "Percent_Done","Days_Remaining_Sprint","Historical_Velocity",
             "Blocked_Stories","Scope_Change"] if c in df.columns]
        X, y = df[f].fillna(0), df["Success_Label"]
        if y.nunique() > 1:
            sc = StandardScaler(); Xs = sc.fit_transform(X)
            Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42)
            m = DecisionTreeClassifier(max_depth=6,class_weight="balanced",
                                       random_state=42,min_samples_leaf=10)
            m.fit(Xtr,ytr); yp = m.predict(Xte)
            R["sprint"] = {"model":m,"scaler":sc,"features":f,"algo":"Decision Tree",
                           "acc":accuracy_score(yte,yp),"report":classification_report(yte,yp)}
    except: pass

    # Obj 2: Workload — Naive Bayes
    try:
        f = [c for c in ["Planned_Story_Points_Resource","Current_Assigned_SP","Historical_Avg_SP",
             "Remaining_Days_Resource","High_Priority_Tasks_Resource","Current_Workload_Percent"]
             if c in df.columns]
        X, y = df[f].fillna(0), df["Expected_Overload"]
        if y.nunique() > 1:
            sc = StandardScaler(); Xs = sc.fit_transform(X)
            Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42)
            m = GaussianNB(); m.fit(Xtr,ytr); yp = m.predict(Xte)
            R["workload"] = {"model":m,"scaler":sc,"features":f,"algo":"Naive Bayes",
                             "acc":accuracy_score(yte,yp),"report":classification_report(yte,yp)}
    except: pass

    # Obj 3: TTR — Ridge Regression
    try:
        Xb = pd.get_dummies(df[["Issue_Type","Priority"]],drop_first=False)
        ex = [c for c in ["Original_Estimate_Hours","Story_Points_Issue"] if c in df.columns]
        X  = pd.concat([Xb,df[ex]],axis=1).fillna(0)
        y  = df["Resolution_Time_Hours"]
        sc = StandardScaler(); Xs = sc.fit_transform(X)
        Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42)
        m = Ridge(alpha=1.0); m.fit(Xtr,ytr); yp = m.predict(Xte)
        R["ttr"] = {"model":m,"scaler":sc,"features":X.columns.tolist(),"X3":X,
                    "algo":"Ridge Regression",
                    "r2":r2_score(yte,yp),"mse":mean_squared_error(yte,yp)}
    except: pass

    # Obj 4: Burnout — Decision Tree
    try:
        f = [c for c in ["Total_SP_This_Sprint","Historical_Avg_SP_Burnout",
             "High_Priority_Tasks_Burnout","Consecutive_Overloads"] if c in df.columns]
        X, y = df[f].fillna(0), df["Risk_Flag"]
        if y.nunique() > 1:
            sc = StandardScaler(); Xs = sc.fit_transform(X)
            Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42)
            m = DecisionTreeClassifier(max_depth=5,class_weight="balanced",
                                       random_state=42,min_samples_leaf=10)
            m.fit(Xtr,ytr); yp = m.predict(Xte)
            R["burnout"] = {"model":m,"scaler":sc,"features":f,"algo":"Decision Tree",
                            "acc":accuracy_score(yte,yp),"report":classification_report(yte,yp)}
    except: pass

    # Obj 5: Allocation — KNN
    try:
        df2 = df.copy()
        le_s = LabelEncoder(); le_l = LabelEncoder()
        df2["Summary_enc"] = le_s.fit_transform(df2["Summary"].astype(str))
        df2["Labels_enc"]  = le_l.fit_transform(df2["Labels"].astype(str))
        X = df2[["Summary_enc","Labels_enc","Original_Estimate_Resource",
                 "Story_Points_Resource"]].fillna(0)
        y = df2["Assignee_Resource"]
        sc = StandardScaler(); Xs = sc.fit_transform(X)
        Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42)
        m = KNeighborsClassifier(n_neighbors=5,weights="distance",n_jobs=-1)
        m.fit(Xtr,ytr)
        R["alloc"] = {"model":m,"scaler":sc,"features":X.columns.tolist(),
                      "le_summary":le_s,"le_labels":le_l,"algo":"KNN",
                      "acc":accuracy_score(yte,m.predict(Xte))}
    except: pass

    return R

# ── Cached Spark ML tab training (ensemble + spark feats) ──────────────────
@st.cache_data(show_spinner=False)
def train_spark_ml(df):
    """Spark ML tab: all 5 obj with ensemble models + Spark features. Cached."""
    import warnings; warnings.filterwarnings("ignore")
    R = {}
    sf = [c for c in SPARK_FEATS if c in df.columns]

    # Sprint — Ensemble Voting
    try:
        f = [c for c in ["Planned_Story_Points_Sprint","Completed_Story_Points","Percent_Done",
             "Days_Remaining_Sprint","Historical_Velocity","Blocked_Stories","Scope_Change"]+sf
             if c in df.columns]
        X,y = df[f].fillna(0), df["Success_Label"]
        if y.nunique()>1:
            Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
            lr  = LogisticRegression(max_iter=300,class_weight="balanced",random_state=42)
            gbt = GradientBoostingClassifier(n_estimators=60,random_state=42)
            rf  = RandomForestClassifier(n_estimators=60,random_state=42,n_jobs=-1)
            ada = AdaBoostClassifier(n_estimators=60,random_state=42)
            ens = VotingClassifier([("lr",lr),("gbt",gbt),("rf",rf),("ada",ada)],voting="soft")
            ens.fit(Xtr,ytr)
            ind = {}
            for nm,clf in [("LR",lr),("GBT",gbt),("RF",rf),("AdaBoost",ada)]:
                clf.fit(Xtr,ytr); ind[nm]=accuracy_score(yte,clf.predict(Xte))
            gbt.fit(Xtr,ytr)
            imp = pd.Series(gbt.feature_importances_,index=f).sort_values(ascending=False)
            R["sprint"] = {"ens_acc":accuracy_score(yte,ens.predict(Xte)),
                           "ind":ind,"feat":f,"sf":sf,"imp":imp.to_dict()}
    except: pass

    # Workload — LR + Spark
    try:
        f = [c for c in ["Planned_Story_Points_Resource","Current_Assigned_SP","Historical_Avg_SP",
             "Remaining_Days_Resource","High_Priority_Tasks_Resource","Current_Workload_Percent"]+sf
             if c in df.columns]
        X,y = df[f].fillna(0), df["Expected_Overload"]
        if y.nunique()>1:
            sc = StandardScaler(); Xs = sc.fit_transform(X)
            Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42)
            m = LogisticRegression(max_iter=300,class_weight="balanced",random_state=42)
            m.fit(Xtr,ytr); yp=m.predict(Xte)
            R["workload"] = {"acc":accuracy_score(yte,yp),"report":classification_report(yte,yp),
                             "feat":f,"sf":sf}
    except: pass

    # TTR — GBT Regressor + Spark
    try:
        Xb = pd.get_dummies(df[["Issue_Type","Priority"]],drop_first=False)
        ex = [c for c in ["Original_Estimate_Hours","Story_Points_Issue"]+sf if c in df.columns]
        X  = pd.concat([Xb,df[ex]],axis=1).fillna(0)
        y  = df["Resolution_Time_Hours"]
        sc = StandardScaler(); Xs = sc.fit_transform(X)
        Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42)
        gbr = GradientBoostingRegressor(n_estimators=60,random_state=42)
        lr3 = LinearRegression()
        gbr.fit(Xtr,ytr); lr3.fit(Xtr,ytr)
        R["ttr"] = {"gb_r2":r2_score(yte,gbr.predict(Xte)),
                    "lr_r2":r2_score(yte,lr3.predict(Xte)),
                    "gb_mse":mean_squared_error(yte,gbr.predict(Xte)),
                    "feat":X.columns.tolist(),"sf":sf}
    except: pass

    # Burnout — GBT + RF Ensemble + Spark
    try:
        f = [c for c in ["Total_SP_This_Sprint","Historical_Avg_SP_Burnout",
             "High_Priority_Tasks_Burnout","Consecutive_Overloads"]+sf if c in df.columns]
        X,y = df[f].fillna(0), df["Risk_Flag"]
        if y.nunique()>1:
            sc = StandardScaler(); Xs = sc.fit_transform(X)
            Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42)
            gbt4 = GradientBoostingClassifier(n_estimators=60,random_state=42)
            rf4  = RandomForestClassifier(n_estimators=60,class_weight="balanced",random_state=42,n_jobs=-1)
            ens4 = VotingClassifier([("gbt",gbt4),("rf",rf4)],voting="soft")
            ens4.fit(Xtr,ytr); yp=ens4.predict(Xte)
            R["burnout"] = {"acc":accuracy_score(yte,yp),"report":classification_report(yte,yp),
                            "feat":f,"sf":sf}
    except: pass

    # Allocation — AdaBoost
    try:
        df2=df.copy()
        le_s=LabelEncoder(); le_l=LabelEncoder()
        df2["Summary_enc"]=le_s.fit_transform(df2["Summary"].astype(str))
        df2["Labels_enc"] =le_l.fit_transform(df2["Labels"].astype(str))
        X=df2[["Summary_enc","Labels_enc","Original_Estimate_Resource","Story_Points_Resource"]].fillna(0)
        y=df2["Assignee_Resource"]
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
        m=AdaBoostClassifier(n_estimators=60,random_state=42); m.fit(Xtr,ytr)
        R["alloc"] = {"acc":accuracy_score(yte,m.predict(Xte)),"model":m,
                      "le_summary":le_s,"le_labels":le_l,"feat":X.columns.tolist()}
    except: pass

    # K-Means Clustering
    try:
        if "Assignee" in df.columns:
            agg={}
            if "Current_Workload_Percent" in df.columns: agg["Workload"]=("Current_Workload_Percent","mean")
            if "Risk_Flag" in df.columns:               agg["Burnout"] =("Risk_Flag","mean")
            if "Success_Label" in df.columns:           agg["SprintRisk"]=("Success_Label",lambda x:(x==0).mean())
            if "Consecutive_Overloads" in df.columns:  agg["ConsecOL"]=("Consecutive_Overloads","mean")
            if len(agg)>=2:
                adf=df.groupby("Assignee").agg(**agg).fillna(0)
                Xc=StandardScaler().fit_transform(adf)
                km=KMeans(n_clusters=min(3,len(adf)),random_state=42,n_init=10)
                adf["Cluster"]=km.fit_predict(Xc)
                nc=adf.select_dtypes(include="number").columns.tolist()
                R["cluster"]={"agg":adf.to_dict(),"num_cols":nc}
    except: pass

    # Anomaly Detection
    try:
        af=[c for c in ["Current_Workload_Percent","Blocked_Stories","Consecutive_Overloads",
                         "Completion_Gap","Blocker_Severity","Workload_Stress"] if c in df.columns]
        if len(af)>=2:
            iso=IsolationForest(contamination=0.05,random_state=42,n_estimators=100)
            sc=iso.fit_predict(df[af].fillna(0)); cf=iso.score_samples(df[af].fillna(0))
            R["anomaly"]={"count":int((sc==-1).sum()),"feats":af,
                          "scores":sc.tolist(),"confs":cf.tolist()}
    except: pass

    return R

# ── Sample dataset ──────────────────────────────────────────────────────────
def make_sample(n=500):
    np.random.seed(42)
    asgn=["Alice","Bob","Carol","David","Eve","Frank","Grace","Henry"]
    itp =["Bug","Story","Task","Epic","Sub-task"]
    pri =["High","Medium","Low"]
    pln =np.random.randint(20,80,n).astype(float)
    cmp =np.clip(pln*np.random.uniform(0.3,1.1,n),0,pln)
    pct =(cmp/pln*100).round(1)
    dr  =np.random.randint(0,14,n).astype(float)
    hv  =np.random.randint(20,70,n).astype(float)
    blk =np.random.randint(0,5,n).astype(float)
    sc  =np.random.randint(-5,10,n).astype(float)
    wl  =np.random.uniform(60,160,n).round(1)
    co  =np.random.randint(0,5,n)
    eh  =np.random.exponential(6,n).clip(1,40).round(1)
    sp  =np.random.choice([1,2,3,5,8],n)
    ar  =np.random.choice(asgn,n)
    return pd.DataFrame({
        "Sprint_Number":          np.random.randint(1,21,n),
        "Planned_Story_Points_Sprint": pln,
        "Completed_Story_Points": cmp.round(1),
        "Percent_Done":           pct,
        "Days_Remaining_Sprint":  dr,
        "Historical_Velocity":    hv,
        "Blocked_Stories":        blk,
        "Scope_Change":           sc,
        "Success_Label":          ((pct>60)&(blk<3)&(dr>1)).astype(int),
        "Assignee":               ar,
        "Planned_Story_Points_Resource": pln*0.8,
        "Current_Assigned_SP":    pln*np.random.uniform(0.7,1.3,n),
        "Historical_Avg_SP":      hv*0.9,
        "Remaining_Days_Resource":dr,
        "High_Priority_Tasks_Resource": np.random.randint(0,5,n).astype(float),
        "Current_Workload_Percent": wl,
        "Expected_Overload":      (wl>110).astype(int),
        "Issue_Type":             np.random.choice(itp,n,p=[0.3,0.3,0.2,0.1,0.1]),
        "Priority":               np.random.choice(pri,n,p=[0.3,0.5,0.2]),
        "Original_Estimate_Hours": eh,
        "Story_Points_Issue":     sp,
        "Resolution_Time_Hours":  (eh*np.random.uniform(0.5,1.8,n)).round(1),
        "Total_SP_This_Sprint":   pln,
        "Historical_Avg_SP_Burnout": hv*0.85,
        "High_Priority_Tasks_Burnout": np.random.randint(0,5,n).astype(float),
        "Consecutive_Overloads":  co,
        "Risk_Flag":              ((co>=2)|(wl>130)).astype(int),
        "Summary":                np.random.choice(["Fix login","Add search","Refactor API","Update DB"],n),
        "Labels":                 np.random.choice(["feature","bug","tech-debt","hotfix"],n),
        "Original_Estimate_Resource": eh,
        "Story_Points_Resource":  sp,
        "Assignee_Resource":      ar,
    })

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Agile Dashboard", layout="wide", page_icon="🚀")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0a0d14; color: #e2e8f0; }

.dash-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.dash-sub { color: #64748b; font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; }

.card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.card.critical { border-left: 4px solid #f43f5e; background: #1c0f14; }
.card.warning  { border-left: 4px solid #f59e0b; background: #1c1508; }
.card.success  { border-left: 4px solid #10b981; background: #0c1c17; }
.card.info     { border-left: 4px solid #38bdf8; background: #0c1825; }

.card-title  { font-weight: 700; font-size: 0.95rem; margin-bottom: 0.3rem; }
.card-detail { font-size: 0.82rem; color: #94a3b8; line-height: 1.6; }
.card-action { font-size: 0.8rem; color: #7dd3fc; font-style: italic; margin-top: 0.4rem; }

.metric-box {
    background: #111827; border: 1px solid #1e293b;
    border-radius: 8px; padding: 0.9rem 1rem; text-align: center;
}
.metric-val  { font-family: 'Space Mono',monospace; font-size: 1.8rem; font-weight: 700; }
.metric-lbl  { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

.badge {
    display: inline-block; padding: 2px 10px; border-radius: 4px;
    font-size: 0.72rem; font-weight: 700; margin: 2px;
    font-family: 'Space Mono', monospace;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #38bdf8;
    border-bottom: 1px solid #1e293b; padding-bottom: 6px;
    margin: 1.2rem 0 0.8rem;
}

[data-testid="stMetricValue"]  { font-family: 'Space Mono',monospace; font-size: 1.4rem !important; color: #e2e8f0 !important; }
[data-testid="stMetricLabel"]  { font-size: 0.72rem !important; color: #64748b !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricDelta"]  { font-size: 0.78rem !important; }

div[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
}

.stButton > button {
    background: linear-gradient(135deg,#1d4ed8,#4f46e5) !important;
    color: white !important; border: none !important;
    border-radius: 6px !important; font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #111827 !important; color: #e2e8f0 !important;
    border: 1px solid #1e293b !important; border-radius: 6px !important;
}

.bar-wrap { background:#1e293b; border-radius:4px; height:7px; margin-top:3px; }
.bar-fill  { height:7px; border-radius:4px; }

[data-testid="stExpander"] { background: #111827 !important; border: 1px solid #1e293b !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("<div class='dash-title'>🚀 AI Agile Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='dash-sub'>Autonomous ML • Sprint Intelligence • Team Health</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Upload / Sample ─────────────────────────────────────────────────────────
uc1, uc2 = st.columns([4,1])
with uc1:
    uploaded = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
with uc2:
    use_sample = st.button("🎲 Sample Data", use_container_width=True)

if "sample_loaded" not in st.session_state:
    st.session_state.sample_loaded = False
if use_sample:
    st.session_state.sample_loaded = True

df_raw = None
if uploaded:
    df_raw = pd.read_csv(uploaded)
    st.session_state.sample_loaded = False
elif st.session_state.sample_loaded:
    df_raw = make_sample(500)

if df_raw is None:
    st.markdown("""
    <div class='card info' style='margin-top:2rem;text-align:center;padding:2.5rem;'>
        <div style='font-size:2.5rem;margin-bottom:0.8rem;'>📂</div>
        <div class='card-title' style='font-size:1.1rem;'>Upload your Agile CSV or use Sample Data</div>
        <div class='card-detail'>Supports agile_dataset_large.csv, agile_dataset_healthy.csv, or any agile CSV</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Prep data ───────────────────────────────────────────────────────────────
df = df_raw.copy().fillna(0)
for col, thresh in [("Success_Label",0.5),("Expected_Overload",0.5),("Risk_Flag",0.3)]:
    if col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map({"No":0,"Yes":1}).fillna(0).astype(int)
        else:
            df[col] = (df[col]>thresh).astype(int)

_spark = get_spark()
df = engineer_features(df, _spark)
_engine = "⚡ Spark" if (SPARK_AVAILABLE and _spark) else "🐼 Pandas"
_sf_found = [c for c in SPARK_FEATS if c in df.columns]

# Summary strip
sc1,sc2,sc3,sc4,sc5 = st.columns(5)
sc1.metric("Records",  f"{len(df):,}")
sc2.metric("Features", len(df.columns))
sc3.metric("Engine",   _engine)
if "Success_Label" in df.columns: sc4.metric("At Risk", int((df["Success_Label"]==0).sum()))
if "Risk_Flag"     in df.columns: sc5.metric("Burnout Flags", int(df["Risk_Flag"].sum()))

if _sf_found:
    st.success(f"✅ {len(_sf_found)} Spark features engineered: {', '.join(_sf_found)}")

# ── Train models ─────────────────────────────────────────────────────────────
M  = train_objectives(df)
SM = train_spark_ml(df)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🤖 Agentic AI",
                "1️⃣ Sprint","2️⃣ Workload","3️⃣ TTR","4️⃣ Burnout","5️⃣ Allocation",
                "⚡ Spark ML"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — Agentic AI
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("<div class='section-header'>Autonomous Scan</div>", unsafe_allow_html=True)
    findings, chain = [], []

    # Sprint risk
    if "sprint" in M:
        try:
            m=M["sprint"]; X=df[m["features"]].fillna(0); Xs=m["scaler"].transform(X)
            preds=m["model"].predict(Xs); probs=m["model"].predict_proba(Xs)[:,1]
            at_risk=int((preds==0).sum()); pct=at_risk/len(preds)
            if pct>0.15:
                sev="critical" if pct>0.5 else "warning"
                findings.append({"sev":sev,"obj":"Sprint","icon":"🔴" if sev=="critical" else "🟡",
                    "title":f"{at_risk}/{len(preds)} sprints ({pct:.0%}) at risk of spillover",
                    "detail":f"Avg risk prob: {probs[preds==0].mean():.0%} | Avg blocked: {df.loc[preds==0,'Blocked_Stories'].mean():.1f}" if 'Blocked_Stories' in df.columns else "",
                    "action":"Reduce scope or unblock stories before sprint closes."})
            else:
                findings.append({"sev":"success","obj":"Sprint","icon":"✅",
                    "title":"All sprints on track","detail":"No spillover risk detected.","action":""})
        except: pass

    # Workload
    if "workload" in M:
        try:
            m=M["workload"]; X=df[m["features"]].fillna(0); Xs=m["scaler"].transform(X)
            preds=m["model"].predict(Xs); cnt=int((preds==1).sum())
            if cnt > len(df)*0.2:
                sev="critical" if cnt>len(df)*0.45 else "warning"
                findings.append({"sev":sev,"obj":"Workload","icon":"🔴" if sev=="critical" else "🟡",
                    "title":f"{cnt} resource(s) projected overloaded",
                    "detail":f"Avg workload: {df.loc[preds==1,'Current_Workload_Percent'].mean():.0f}%" if 'Current_Workload_Percent' in df.columns else "",
                    "action":"Redistribute story points to under-capacity members."})
            else:
                findings.append({"sev":"success","obj":"Workload","icon":"✅",
                    "title":"Workloads within capacity","detail":"No overload signals.","action":""})
        except: pass

    # Burnout
    if "burnout" in M:
        try:
            m=M["burnout"]; X=df[m["features"]].fillna(0); Xs=m["scaler"].transform(X)
            preds=m["model"].predict(Xs); cnt=int(preds.sum()); pct=cnt/len(preds)
            if pct>0.25:
                sev="critical" if pct>0.5 else "warning"
                findings.append({"sev":sev,"obj":"Burnout","icon":"🔴" if sev=="critical" else "🟡",
                    "title":f"{cnt} member(s) at burnout risk",
                    "detail":f"Avg consecutive overloads: {df.loc[preds==1,'Consecutive_Overloads'].mean():.1f}" if 'Consecutive_Overloads' in df.columns else "",
                    "action":"Lighten next sprint load, schedule 1:1s."})
            else:
                findings.append({"sev":"success","obj":"Burnout","icon":"✅",
                    "title":"No burnout risk detected","detail":"Team load looks sustainable.","action":""})
        except: pass

    # Health score
    score = max(0, min(100, 100 - sum(25 if f["sev"]=="critical" else 10 if f["sev"]=="warning" else 0 for f in findings)))
    sc_color = "#10b981" if score>=75 else "#f59e0b" if score>=50 else "#f43f5e"
    sc_label = "🟢 Healthy" if score>=75 else "🟡 Attention" if score>=50 else "🔴 At Risk"

    hc1, hc2, hc3 = st.columns([1,2,1])
    with hc1:
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-val' style='color:{sc_color};'>{score}</div>
            <div class='metric-lbl'>Health Score</div>
            <div style='margin-top:6px;font-size:0.85rem;'>{sc_label}</div>
        </div>""", unsafe_allow_html=True)
    with hc2:
        st.markdown(f"""<div style='margin-top:1.5rem;'>
            <div class='bar-wrap'><div class='bar-fill' style='background:{sc_color};width:{score}%;'></div></div>
            <div style='display:flex;justify-content:space-between;font-size:0.7rem;color:#475569;margin-top:4px;'>
                <span>0 Critical</span><span>50 Attention</span><span>100 Healthy</span></div>
        </div>""", unsafe_allow_html=True)
    with hc3:
        crits = sum(1 for f in findings if f["sev"]=="critical")
        warns = sum(1 for f in findings if f["sev"]=="warning")
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-val' style='color:#f43f5e;'>{crits}</div>
            <div class='metric-lbl'>Critical</div>
            <div class='metric-val' style='color:#f59e0b;margin-top:8px;'>{warns}</div>
            <div class='metric-lbl'>Warnings</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Findings</div>", unsafe_allow_html=True)
    sev_order = {"critical":0,"warning":1,"info":2,"success":3}
    for f in sorted(findings, key=lambda x: sev_order.get(x["sev"],9)):
        act = f"<div class='card-action'>→ {f['action']}</div>" if f["action"] else ""
        st.markdown(f"""<div class='card {f["sev"]}'>
            <div class='card-title'>{f["icon"]} [{f["obj"]}] {f["title"]}</div>
            <div class='card-detail'>{f["detail"]}{act}</div>
        </div>""", unsafe_allow_html=True)

    # Per-assignee breakdown
    acol = "Assignee" if "Assignee" in df.columns else ("Assignee_Resource" if "Assignee_Resource" in df.columns else None)
    if acol:
        st.markdown("<div class='section-header'>Per-Assignee Risk</div>", unsafe_allow_html=True)
        people = sorted(df[acol].unique())
        cols_p = st.columns(min(len(people), 8))
        for i, person in enumerate(people):
            sub = df[df[acol]==person]
            sr  = sub["Success_Label"].eq(0).mean()   if "Success_Label" in df.columns else 0
            ol  = sub["Expected_Overload"].mean()      if "Expected_Overload" in df.columns else 0
            br  = sub["Risk_Flag"].mean()              if "Risk_Flag" in df.columns else 0
            wl  = sub["Current_Workload_Percent"].mean() if "Current_Workload_Percent" in df.columns else 0
            ps  = max(0,min(100, 100-(sr*35)-(ol*30)-(br*20)-max(0,(wl-100)/2)))
            pc  = "#10b981" if ps>=60 else "#f59e0b" if ps>=40 else "#f43f5e"
            pl  = "🟢" if ps>=60 else "🟡" if ps>=40 else "🔴"
            with cols_p[i % len(cols_p)]:
                st.markdown(f"""<div class='card {"success" if ps>=60 else "warning" if ps>=40 else "critical"}' style='text-align:center;padding:0.8rem;'>
                    <div class='metric-val' style='color:{pc};font-size:1.5rem;'>{ps:.0f}</div>
                    <div style='font-size:0.9rem;font-weight:700;margin:4px 0;'>{person}</div>
                    <div style='font-size:0.72rem;'>{pl}</div>
                    <div style='font-size:0.72rem;color:#64748b;text-align:left;margin-top:8px;line-height:1.9;'>
                        🏃 Sprint risk: <b>{sr:.0%}</b><br>
                        📦 Overload: <b>{ol:.0%}</b><br>
                        🔥 Burnout: <b>{br:.0%}</b><br>
                        ⚡ Workload: <b>{wl:.0f}%</b>
                    </div>
                </div>""", unsafe_allow_html=True)

    # Action table
    action_rows=[]
    for f in sorted(findings, key=lambda x: sev_order.get(x["sev"],9)):
        if f.get("action"):
            p = "🔴 P1 — Immediate" if f["sev"]=="critical" else "🟡 P2 — This Sprint"
            action_rows.append({"Priority":p,"Objective":f["obj"],"Issue":f["title"],"Action":f["action"]})
    if action_rows:
        st.markdown("<div class='section-header'>Action Priority</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)

    # Trend charts
    if "Sprint_Number" in df.columns:
        st.markdown("<div class='section-header'>Trends</div>", unsafe_allow_html=True)
        tr = df.copy()
        tr["at_risk"] = (tr["Success_Label"]==0).astype(int)
        agg = tr.groupby("Sprint_Number").agg(
            sprint_risk=("at_risk","mean"),
            avg_workload=("Current_Workload_Percent","mean"),
            burnout=("Risk_Flag","mean")).reset_index()
        agg[["sprint_risk","avg_workload","burnout"]] = (agg[["sprint_risk","avg_workload","burnout"]]*100).round(1)
        tc1,tc2,tc3 = st.columns(3)
        with tc1:
            st.caption("Sprint Risk %")
            st.line_chart(agg.set_index("Sprint_Number")[["sprint_risk"]],height=180,use_container_width=True)
        with tc2:
            st.caption("Avg Workload %")
            st.line_chart(agg.set_index("Sprint_Number")[["avg_workload"]],height=180,use_container_width=True)
        with tc3:
            st.caption("Burnout Flag %")
            st.line_chart(agg.set_index("Sprint_Number")[["burnout"]],height=180,use_container_width=True)

    # Report download
    report = f"""# AI Agile Project Health Report\n\n**Health Score:** {score}/100 — {sc_label}\n\n## Findings\n"""
    for f in findings:
        report += f"\n- **{f['icon']} [{f['obj']}]** {f['title']}: {f['detail']}"
        if f["action"]: report += f"\n  → *{f['action']}*"
    st.download_button("⬇️ Download Report (.md)", data=report,
                       file_name="agile_report.md", mime="text/markdown")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Sprint
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-header'>Objective 1 — Sprint Completion</div>", unsafe_allow_html=True)
    st.caption("Decision Tree Classifier | base features only | max speed")
    if "sprint" not in M:
        st.warning("Sprint model unavailable — check Success_Label column.")
    else:
        m=M["sprint"]
        c1,c2,c3 = st.columns(3)
        c1.metric("Accuracy",   f"{m['acc']:.2%}")
        c2.metric("Algorithm",  m["algo"])
        c3.metric("Features",   len(m["features"]))
        with st.expander("Classification Report"):
            st.text(m["report"])
        st.markdown("<div class='section-header'>Predict</div>", unsafe_allow_html=True)
        p1,p2 = st.columns(2)
        with p1:
            psp = st.number_input("Planned Story Points",   1,100,40, key="s_psp")
            csp = st.number_input("Completed Story Points", 0,100,30, key="s_csp")
            pct = st.slider("% Done", 0.0,100.0,75.0,        key="s_pct")
            drs = st.number_input("Days Remaining",          0, 30, 5, key="s_drs")
        with p2:
            hv  = st.number_input("Historical Velocity",    0,100,35, key="s_hv")
            bs  = st.number_input("Blocked Stories",        0, 10, 1, key="s_bs")
            sc  = st.number_input("Scope Change",         -20, 20, 0, key="s_sc")
        if st.button("🔮 Predict Sprint Success", key="s_btn"):
            row = pd.DataFrame([{"Planned_Story_Points_Sprint":psp,"Completed_Story_Points":csp,
                "Percent_Done":pct,"Days_Remaining_Sprint":drs,
                "Historical_Velocity":hv,"Blocked_Stories":bs,"Scope_Change":sc}])
            row = row.reindex(columns=m["features"],fill_value=0)
            p   = m["model"].predict(m["scaler"].transform(row))[0]
            prob= m["model"].predict_proba(m["scaler"].transform(row))[0][1]
            if p: st.success(f"✅ Likely to Complete — {prob:.2%} confidence")
            else: st.warning(f"⚠️ Risk of Spillover — {prob:.2%} confidence")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Workload
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-header'>Objective 2 — Workload Projection</div>", unsafe_allow_html=True)
    st.caption("Naive Bayes (Gaussian) | fastest binary classifier")
    if "workload" not in M:
        st.warning("Workload model unavailable — check Expected_Overload column.")
    else:
        m=M["workload"]
        c1,c2 = st.columns(2)
        c1.metric("Accuracy",  f"{m['acc']:.2%}")
        c2.metric("Algorithm", m["algo"])
        with st.expander("Classification Report"):
            st.text(m["report"])
        st.markdown("<div class='section-header'>Predict</div>", unsafe_allow_html=True)
        p1,p2 = st.columns(2)
        with p1:
            psp2 = st.number_input("Planned SP",          1,100, 35, key="w_psp")
            casp = st.number_input("Current Assigned SP", 0,100, 40, key="w_casp")
            hasp = st.number_input("Historical Avg SP",   1,100, 30, key="w_hasp")
        with p2:
            rdr  = st.number_input("Remaining Days",      1, 30,  5, key="w_rdr")
            hpt  = st.number_input("High Priority Tasks", 0, 10,  2, key="w_hpt")
            cwp  = st.number_input("Current Workload %",  0,200,125, key="w_cwp")
        if st.button("🔮 Predict Overload", key="w_btn"):
            row = pd.DataFrame([{"Planned_Story_Points_Resource":psp2,"Current_Assigned_SP":casp,
                "Historical_Avg_SP":hasp,"Remaining_Days_Resource":rdr,
                "High_Priority_Tasks_Resource":hpt,"Current_Workload_Percent":cwp}])
            row  = row.reindex(columns=m["features"],fill_value=0)
            pred = m["model"].predict(m["scaler"].transform(row))[0]
            prob = m["model"].predict_proba(m["scaler"].transform(row))[0][1]
            if pred: st.warning(f"⚠️ Overload Risk — {prob:.2%} confidence")
            else:    st.success(f"✅ Within Capacity — {prob:.2%} confidence")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TTR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='section-header'>Objective 3 — Time to Resolve</div>", unsafe_allow_html=True)
    st.caption("Ridge Regression | regularised linear model | fast hour prediction")
    if "ttr" not in M:
        st.warning("TTR model unavailable — check Resolution_Time_Hours column.")
    else:
        m=M["ttr"]
        c1,c2,c3 = st.columns(3)
        c1.metric("R²",  f"{m['r2']:.3f}")
        c2.metric("MSE", f"{m['mse']:.2f}")
        c3.metric("Algorithm", m["algo"])
        st.markdown("<div class='section-header'>Predict</div>", unsafe_allow_html=True)
        p1,p2 = st.columns(2)
        with p1:
            itype = st.selectbox("Issue Type",["Bug","Story","Task"], key="t_it")
            pri   = st.selectbox("Priority",  ["Low","Medium","High"], key="t_pri")
        with p2:
            oe = st.number_input("Original Estimate (h)",1,50,8, key="t_oe")
            sp = st.number_input("Story Points",         1,20,5, key="t_sp")
        if st.button("🔮 Estimate Resolution Time", key="t_btn"):
            row={c:0 for c in m["features"]}
            if f"Issue_Type_{itype}" in row: row[f"Issue_Type_{itype}"]=1
            if f"Priority_{pri}"     in row: row[f"Priority_{pri}"]    =1
            row["Original_Estimate_Hours"]=oe
            if "Story_Points_Issue" in row: row["Story_Points_Issue"]=sp
            pred_t = max(0, m["model"].predict(m["scaler"].transform(pd.DataFrame([row])[m["features"]]))[0])
            delta  = pred_t - oe
            st.metric("Estimated Resolution Time", f"{pred_t:.1f} hours", f"{delta:+.1f}h vs estimate")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Burnout
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("<div class='section-header'>Objective 4 — Burnout Risk</div>", unsafe_allow_html=True)
    st.caption("Decision Tree Classifier | threshold-based rules | explainable")
    if "burnout" not in M:
        st.warning("Burnout model unavailable — check Risk_Flag column.")
    else:
        m=M["burnout"]
        c1,c2,c3 = st.columns(3)
        c1.metric("Accuracy",  f"{m['acc']:.2%}")
        c2.metric("Algorithm", m["algo"])
        c3.metric("Features",  len(m["features"]))
        with st.expander("Classification Report"):
            st.text(m["report"])
        st.markdown("<div class='section-header'>Predict</div>", unsafe_allow_html=True)
        p1,p2 = st.columns(2)
        with p1:
            tsp  = st.number_input("Total SP This Sprint",  0,100,40, key="b_tsp")
            hasp = st.number_input("Historical Avg SP",     1,100,25, key="b_hasp")
        with p2:
            hpt  = st.number_input("High Priority Tasks",  0, 10, 2, key="b_hpt")
            co   = st.number_input("Consecutive Overloads",0,  5, 2, key="b_co")
        if st.button("🔮 Check Burnout Risk", key="b_btn"):
            row={f:0 for f in m["features"]}
            row.update({"Total_SP_This_Sprint":tsp,"Historical_Avg_SP_Burnout":hasp,
                        "High_Priority_Tasks_Burnout":hpt,"Consecutive_Overloads":co})
            row_s = m["scaler"].transform(pd.DataFrame([row])[m["features"]])
            pred  = m["model"].predict(row_s)[0]
            prob  = m["model"].predict_proba(row_s)[0][1]
            if pred: st.warning(f"⚠️ Burnout Risk Detected — {prob:.2%} confidence")
            else:    st.success(f"✅ Workload Healthy — {prob:.2%} confidence")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Allocation
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("<div class='section-header'>Objective 5 — Resource Allocation</div>", unsafe_allow_html=True)
    st.caption("KNN Classifier | groups similar issues to similar assignees")
    if "alloc" not in M:
        st.warning("Allocation model unavailable.")
    else:
        m=M["alloc"]
        c1,c2 = st.columns(2)
        c1.metric("Accuracy",  f"{m['acc']:.2%}")
        c2.metric("Algorithm", m["algo"])
        st.info("ℹ️ ~12-15% accuracy expected — needs skill-tag and component-owner features for higher accuracy.")
        st.markdown("<div class='section-header'>Suggest Assignee</div>", unsafe_allow_html=True)
        p1,p2 = st.columns(2)
        with p1:
            summary = st.text_input("Issue Summary","Fix login bug", key="a_sum")
            label   = st.text_input("Label",        "Bug",            key="a_lbl")
        with p2:
            oe5 = st.number_input("Original Estimate (h)",1,50,8, key="a_oe")
            sp5 = st.number_input("Story Points",         1,20,5, key="a_sp")
        if st.button("🔮 Suggest Assignee", key="a_btn"):
            le_s=m["le_summary"]; le_l=m["le_labels"]
            try:    se=le_s.transform([summary])[0]
            except: se=0
            try:    le=le_l.transform([label])[0]
            except: le=0
            row=pd.DataFrame([{"Summary_enc":se,"Labels_enc":le,
                                "Original_Estimate_Resource":oe5,"Story_Points_Resource":sp5}])
            row=row.reindex(columns=m["features"],fill_value=0)
            st.success(f"✅ Recommended Assignee: **{m['model'].predict(m['scaler'].transform(row))[0]}**")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Spark ML (all 5 obj + clustering + anomaly)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("<div class='section-header'>Spark ML — All 5 Objectives with Ensemble Models + Spark Features</div>", unsafe_allow_html=True)
    if SPARK_AVAILABLE and _spark:
        st.success("⚡ Apache Spark active — distributed mode")
    else:
        st.info("🐼 Pandas mode — install Java 17 + pyspark for Spark acceleration")
    if _sf_found:
        st.success(f"✅ {len(_sf_found)} Spark features active: {', '.join(_sf_found)}")

    # 1 Sprint Ensemble
    st.markdown("<div class='section-header'>1️⃣ Sprint — Ensemble Voting (LR+GBT+RF+AdaBoost)</div>", unsafe_allow_html=True)
    if "sprint" in SM:
        s=SM["sprint"]
        ea=s["ens_acc"]; ind=s["ind"]; best=max(ind,key=ind.get)
        c1,c2,c3 = st.columns(3)
        c1.metric("Ensemble Accuracy", f"{ea:.2%}")
        c2.metric("Best Single Model", f"{ind[best]:.2%}", best)
        c3.metric("Spark Features",    str(len([f for f in s["feat"] if f in s["sf"]])))
        with st.expander("Model Comparison"):
            for nm,acc in sorted(ind.items(),key=lambda x:-x[1]):
                diff=ea-acc; dc="#10b981" if diff>=0 else "#f43f5e"
                st.markdown(f"""<div style='margin-bottom:5px;'>
  <div style='display:flex;justify-content:space-between;font-size:0.8rem;color:#cbd5e1;'>
    <span>{nm}</span><span>{acc:.2%} <span style='color:{dc};'>({diff:+.2%})</span></span></div>
  <div class='bar-wrap'><div class='bar-fill' style='background:#38bdf8;width:{acc*100:.1f}%;'></div></div>
</div>""", unsafe_allow_html=True)
        with st.expander("Feature Importance"):
            imp=pd.Series(s["imp"]).sort_values(ascending=False)
            for feat,v in imp.items():
                bw=v/imp.max()*100; tag=" ⚡" if feat in s["sf"] else ""
                c="#10b981" if v>imp.mean() else "#38bdf8"
                st.markdown(f"""<div style='margin-bottom:4px;'>
  <div style='display:flex;justify-content:space-between;font-size:0.75rem;color:#cbd5e1;'>
    <span>{feat}{tag}</span><span style='color:{c};'>{v:.3f}</span></div>
  <div class='bar-wrap'><div class='bar-fill' style='background:{c};width:{bw:.0f}%;'></div></div>
</div>""", unsafe_allow_html=True)
            st.caption("⚡ = Spark-engineered feature")
    else:
        st.info("Sprint ensemble unavailable.")

    # 2 Workload
    st.markdown("<div class='section-header'>2️⃣ Workload — Logistic Regression + Spark Features</div>", unsafe_allow_html=True)
    if "workload" in SM:
        w=SM["workload"]
        c1,c2=st.columns(2)
        c1.metric("Accuracy",       f"{w['acc']:.2%}")
        c2.metric("Spark Features", str(len([f for f in w["feat"] if f in w["sf"]])))
        with st.expander("Classification Report"): st.text(w["report"])
    else: st.info("Workload unavailable.")

    # 3 TTR
    st.markdown("<div class='section-header'>3️⃣ TTR — GBT Regressor + Spark Features</div>", unsafe_allow_html=True)
    if "ttr" in SM:
        t=SM["ttr"]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("GBT R²",  f"{t['gb_r2']:.3f}")
        c2.metric("LR R²",   f"{t['lr_r2']:.3f}", f"{t['gb_r2']-t['lr_r2']:+.3f}")
        c3.metric("GBT MSE", f"{t['gb_mse']:.2f}")
        c4.metric("Spark Features", str(len([f for f in t["feat"] if f in t["sf"]])))
    else: st.info("TTR unavailable.")

    # 4 Burnout
    st.markdown("<div class='section-header'>4️⃣ Burnout — GBT + RF Ensemble + Spark Features</div>", unsafe_allow_html=True)
    if "burnout" in SM:
        b=SM["burnout"]
        c1,c2,c3=st.columns(3)
        c1.metric("Ensemble Accuracy", f"{b['acc']:.2%}")
        c2.metric("Spark Features",    str(len([f for f in b["feat"] if f in b["sf"]])))
        c3.metric("Algorithm",         "GBT + RF Voting")
        with st.expander("Classification Report"): st.text(b["report"])
    else: st.info("Burnout unavailable.")

    # 5 Allocation
    st.markdown("<div class='section-header'>5️⃣ Allocation — AdaBoost</div>", unsafe_allow_html=True)
    if "alloc" in SM:
        a=SM["alloc"]
        c1,c2=st.columns(2)
        c1.metric("Accuracy",  f"{a['acc']:.2%}")
        c2.metric("Algorithm", "AdaBoost")
        st.info("ℹ️ Low accuracy expected — needs skill-tag features.")
    else: st.info("Allocation unavailable.")

    # Clustering
    st.markdown("<div class='section-header'>👥 Team Segmentation — K-Means</div>", unsafe_allow_html=True)
    if "cluster" in SM:
        adf=pd.DataFrame(SM["cluster"]["agg"]); nc=SM["cluster"]["num_cols"]
        cl1,cl2,cl3=st.columns(3)
        for cw,cid,lbl,col in [(cl1,0,"🟢 High Performers","#10b981"),
                                (cl2,1,"🟡 Mid Performers","#f59e0b"),
                                (cl3,2,"🔴 Overloaded","#f43f5e")]:
            members=adf[adf["Cluster"]==cid].index.tolist()
            with cw:
                st.markdown(f"""<div class='card' style='text-align:center;'>
  <div style='font-size:0.8rem;font-weight:700;color:{col};margin-bottom:6px;'>{lbl}</div>
  {''.join([f"<span class='badge' style='background:{col}22;color:{col};'>{m}</span>" for m in members])}
</div>""", unsafe_allow_html=True)
        ddf=adf.drop(columns=["Cluster"],errors="ignore")
        fmt={c:"{:.2f}" for c in nc if c in ddf.columns}
        st.dataframe(ddf.style.format(fmt,na_rep="--"),use_container_width=True)
    else: st.info("Clustering unavailable.")

    # Anomaly
    st.markdown("<div class='section-header'>🔍 Anomaly Detection — Isolation Forest</div>", unsafe_allow_html=True)
    if "anomaly" in SM:
        an=SM["anomaly"]
        sc_s=pd.Series(an["scores"]); cf_s=pd.Series(an["confs"])
        mask=sc_s==-1
        anom_df=df[mask].copy(); anom_df["Anomaly Score"]=cf_s[mask].values
        anom_df=anom_df.sort_values("Anomaly Score")
        c1,c2,c3=st.columns(3)
        c1.metric("Anomalies",   an["count"])
        c2.metric("Rate",        "5.0%")
        c3.metric("Features",    len(an["feats"]))
        dcols=(['Assignee'] if 'Assignee' in df.columns else [])+an["feats"]+["Anomaly Score"]
        dcols=[c for c in dcols if c in anom_df.columns]
        st.dataframe(anom_df[dcols].head(10),use_container_width=True,hide_index=True)
    else: st.info("Anomaly detection unavailable.")

    with st.expander("⚡ Spark MLlib Pipeline Code"):
        st.code("""
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.clustering import KMeans

spark = SparkSession.builder.appName("AgileML").master("local[*]").getOrCreate()
df = spark.read.csv("agile_dataset.csv", header=True, inferSchema=True)

# Spark feature engineering
df = df.withColumn("Velocity_Efficiency", F.col("Historical_Velocity")/F.col("Planned_Story_Points_Sprint"))
df = df.withColumn("Completion_Gap",      F.col("Planned_Story_Points_Sprint")-F.col("Completed_Story_Points"))
df = df.withColumn("Blocker_Severity",    F.col("Blocked_Stories")/F.col("Days_Remaining_Sprint"))
df = df.withColumn("Sprint_Momentum",     F.col("Completed_Story_Points")/F.col("Historical_Velocity"))
df = df.withColumn("Workload_Stress",     (F.col("Current_Workload_Percent")/100)*F.col("Consecutive_Overloads"))

assembler = VectorAssembler(inputCols=["Planned_Story_Points_Sprint","Completed_Story_Points",
    "Percent_Done","Velocity_Efficiency","Completion_Gap","Blocker_Severity","Sprint_Momentum"],
    outputCol="features")
train, test = df.randomSplit([0.8, 0.2], seed=42)
model = Pipeline(stages=[assembler, GBTClassifier(labelCol="Success_Label", maxIter=50)]).fit(train)
""", language="python")