import streamlit as st
import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import io
import numpy as np # 必须导入 numpy

st.set_page_config(page_title="Clinical Decision Support System", layout="wide")
st.title("🩺 Clinical Decision Support System")

# ==========================================
# 1. 模型加载 (Model Loading)
# ==========================================
@st.cache_resource
def load_models():
    # 加载 12 特征模型
    models_12 = {
        1: joblib.load('./selected_features/rf_1yr.pkl'),
        3: joblib.load('./selected_features/rf_3yr.pkl'),
        5: joblib.load('./selected_features/rf_5yr.pkl')
    }

    # 加载 9 特征模型
    try:
        models_9 = {
            1: joblib.load('./nine_features/catboost_op_sm_1yr.pkl'),
            3: joblib.load('./nine_features/catboost_op_sm_3yr.pkl'),
            5: joblib.load('./nine_features/catboost_op_sm_5yr.pkl')
        }
    except FileNotFoundError:
        st.warning("⚠️ 未找到9特征模型文件，演示模式下暂时使用12特征模型替代。")
        models_9 = models_12

    return models_12, models_9

models_12, models_9 = load_models()

# ==========================================
# 2. 模式选择 (Mode Selection)
# ==========================================
st.markdown("### ⚙️ Settings")
model_mode = st.radio(
    "Select Feature Input Mode:",
    ("12 Features (Full)", "9 Features (Reduced)"),
    horizontal=True
)
is_full_mode = (model_mode == "12 Features (Full)")

# ==========================================
# 3. 动态输入界面 (Dynamic UI)
# ==========================================
left_col, right_col = st.columns([2, 3], gap="large")

cakut_subphenotype_list = {
    'renal hypodysplasia associated with puv': 1,
    'solitary kidney': 2,
    'bilateral renal hypodysplasia': 3,
    'unilateral renal hypodysplasia': 4,
    'multicystic dysplastic kidney': 5,
    'horseshoe kidney': 6,
    'others': 7
}

with left_col:
    st.subheader("🏥 Patient Characteristics")
    col1, col2 = st.columns(2, gap='medium')

    with col1:
        age_first_diagnose = st.number_input("Age At First Diagnose(yr)", min_value=0.0, max_value=18.0, value=0.0)
        gender = st.selectbox("Gender", ["Female", "Male"])
        if is_full_mode:
            family_history = st.selectbox("Family history", ["No", "Yes"])
        else:
            family_history = "No"
        ckd_stage_first_diagnose = st.selectbox("CKD Stage At First Diagnose", [1, 2, 3, 4, 5])
        short_stature = st.selectbox("Short Stature", ["No", "Yes"])
        cakut_subphenotype = st.selectbox("CAKUT Subphenotype", cakut_subphenotype_list.keys())

    with col2:
        if is_full_mode:
            pax2 = st.selectbox("PAX2", ["No", "Yes"])
        else:
            pax2 = "No"
        if is_full_mode:
            prenatal_phenotype = st.selectbox("Prenatal Phenotype", ["No", "Yes"])
        else:
            prenatal_phenotype = "No"
        congenital_heart_disease = st.selectbox("Congenital Heart Disease", ["No", "Yes"])
        ocular = st.selectbox("Ocular", ["No", "Yes"])
        preterm_birth = st.selectbox("Preterm Birth", ["No", "Yes"])
        behavioral_cognitive_abnormalities = st.selectbox("Behavioral Cognitive Abnormalities", ["No", "Yes"])

    predict_btn = st.button("PREDICT")

# ==========================================
# 4. 数据构建 (Data Construction)
# ==========================================
def get_binary(val):
    return 0 if val == 'No' or val == 'Female' else 1

data_dict = {
    "gender (1/0)": [get_binary(gender)],
    "preterm_birth (1/0)": [get_binary(preterm_birth)],
    "cakut_subphenotype": [cakut_subphenotype_list[cakut_subphenotype]],
    "behavioral_cognitive_abnormalities (1/0)": [get_binary(behavioral_cognitive_abnormalities)],
    "congenital_heart_disease (1/0)": [get_binary(congenital_heart_disease)],
    "ocular (1/0)": [get_binary(ocular)],
    "age_first_diagnose": [age_first_diagnose],
    "ckd_stage_first_diagnose": [ckd_stage_first_diagnose],
    "short_stature (1/0)": [get_binary(short_stature)]
}

if is_full_mode:
    data_dict.update({
        'PAX2': [get_binary(pax2)],
        'family_history (1/0)': [get_binary(family_history)],
        'prenatal_phenotype (1/0)': [get_binary(prenatal_phenotype)]
    })

input_data = pd.DataFrame(data_dict)

# ==========================================
# 5. 预测与渲染逻辑 (Core Logic)
# ==========================================
def render_prediction(model, input_data, year):
    # 【必须步骤 1】使用副本，防止影响其他年份的预测
    input_data = input_data.copy()

    # 【必须步骤 2】识别核心模型 (解决 Pipeline 报错问题)
    if hasattr(model, 'steps'):
        estimator = model.steps[-1][1] # Pipeline 取最后一步
    else:
        estimator = model # 普通模型

    # 【必须步骤 3】自动修正特征顺序 (解决 Feature names mismatch 问题)
    # 我们不手动去猜顺序，直接问模型“你想要什么顺序？”然后照做
    try:
        # 获取模型期待的特征
        if hasattr(estimator, 'feature_names_in_'):
            expected_features = estimator.feature_names_in_
        elif hasattr(estimator, 'feature_names_'):
            expected_features = estimator.feature_names_
        else:
            expected_features = None
        
        # 如果模型有明确的特征顺序要求，我们就强制对齐
        if expected_features is not None:
            # 防止列缺失报错，如果缺了就补0
            for col in expected_features:
                if col not in input_data.columns:
                    input_data[col] = 0
            # 关键：按模型要求的顺序重新排列
            input_data = input_data[list(expected_features)]
            
    except Exception as e:
        print(f"Warning in alignment: {e}")

    # --- 预测 ---
    try:
        esrd_prob = model.predict_proba(input_data)[0][1]
        st.write(f"Probability of kidney failure within {year} year: **{esrd_prob:.2%}**")
    except Exception as e:
        st.error(f"Prediction Error ({year} yr): {e}")
        return

    # --- SHAP 绘图 (仅支持树模型) ---
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(input_data)

        # 兼容处理：RF 返回 list，XGB/CatBoost 返回 array
        if isinstance(shap_values, list):
            base_value = explainer.expected_value[1]
            shap_values_plot = shap_values[1]
        else:
            base_value = explainer.expected_value
            shap_values_plot = shap_values

        force_plot = shap.force_plot(
            base_value,
            shap_values_plot,
            input_data,
            matplotlib=False
        )
        
        html_buffer = io.StringIO()
        shap.save_html(html_buffer, force_plot)
        html_content = html_buffer.getvalue()
        
        wrapped = f"<div style='width:100%; overflow-x:auto;'>{html_content}</div>"
        components.html(wrapped, height=150, scrolling=True)

    except Exception:
        # 遇到不支持 SHAP 的模型 (如 SVM/KNN) 优雅跳过，不报错
        st.caption("ℹ️ (Details not available for this model type)")

with right_col:
    st.subheader("🤖 Predicted Results")
    if predict_btn:
        current_models = models_12 if is_full_mode else models_9
        
        # 依次调用
        render_prediction(current_models[1], input_data, 1)
        render_prediction(current_models[3], input_data, 3)
        render_prediction(current_models[5], input_data, 5)
