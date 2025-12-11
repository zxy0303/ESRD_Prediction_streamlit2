import streamlit as st
import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import io
import numpy as np
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

    # 加载 9 特征模型 (请确保你有这些文件，且文件名正确)
    try:
        models_9 = {
            1: joblib.load('./nine_features/catboost_op_sm_1yr.pkl'),
            3: joblib.load('./nine_features/catboost_op_sm_3yr.pkl'),
            5: joblib.load('./nine_features/catboost_op_sm_5yr.pkl')
        }
    except FileNotFoundError:
        st.warning("⚠️ 未找到9特征模型文件 (gbm_Xyr_9.pkl)，演示模式下暂时使用12特征模型替代。")
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

st.markdown(
    f"Current Mode: **{model_mode}**. "
    f"{'Includes all clinical features.' if is_full_mode else 'Excludes PAX2, Family History, and Prenatal Phenotype.'}"
)

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

# ==========================================
# 3. 动态输入界面 (Dynamic UI)
# ==========================================
with left_col:
    st.subheader("🏥 Patient Characteristics")
    col1, col2 = st.columns(2, gap='medium')

    # --- 第一列输入 ---
    with col1:
        # [保留] 核心特征
        age_first_diagnose = st.number_input("Age At First Diagnose(yr)", min_value=0.0, max_value=18.0, value=0.0)
        gender = st.selectbox("Gender", ["Female", "Male"])

        # [移除] 仅在 12 特征模式下显示
        if is_full_mode:
            family_history = st.selectbox("Family history", ["No", "Yes"])
        else:
            family_history = "No"  # 默认填充，不参与9特征预测

        # [保留] 核心特征
        ckd_stage_first_diagnose = st.selectbox("CKD Stage At First Diagnose", [1, 2, 3, 4, 5])
        short_stature = st.selectbox("Short Stature", ["No", "Yes"])  # 这次保留了
        cakut_subphenotype = st.selectbox("CAKUT Subphenotype", cakut_subphenotype_list.keys())

    # --- 第二列输入 ---
    with col2:
        # [移除] 仅在 12 特征模式下显示
        if is_full_mode:
            pax2 = st.selectbox("PAX2", ["No", "Yes"])
        else:
            pax2 = "No"

        # [移除] 仅在 12 特征模式下显示
        if is_full_mode:
            prenatal_phenotype = st.selectbox("Prenatal Phenotype", ["No", "Yes"])
        else:
            prenatal_phenotype = "No"

        # [保留] 核心特征
        congenital_heart_disease = st.selectbox("Congenital Heart Disease", ["No", "Yes"])
        ocular = st.selectbox("Ocular", ["No", "Yes"])  # 这次保留了
        preterm_birth = st.selectbox("Preterm Birth", ["No", "Yes"])
        behavioral_cognitive_abnormalities = st.selectbox("Behavioral Cognitive Abnormalities", ["No", "Yes"])

    predict_btn = st.button("PREDICT")


# ==========================================
# 4. 数据构建 (Data Construction)
# ==========================================
def get_binary(val):
    return 0 if val == 'No' or val == 'Female' else 1


# 1. 首先构建 9 个核心特征 (这是你指定的列表)
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

# 2. 如果是 12 特征模式，追加另外 3 个
if is_full_mode:
    data_dict.update({
        'PAX2': [get_binary(pax2)],
        'family_history (1/0)': [get_binary(family_history)],
        'prenatal_phenotype (1/0)': [get_binary(prenatal_phenotype)]
    })

input_data = pd.DataFrame(data_dict)


# ==========================================
# 5. 预测与渲染逻辑 (Prediction Logic)
# ==========================================
def render_prediction(model, input_data, year):
    # [修复1] 创建数据副本！防止修改原始数据影响后续的 3年/5年 预测
    input_data = input_data.copy()
    
    # =================================================
    # 1. 提取核心模型 (Handle Pipeline)
    # =================================================
    try:
        if hasattr(model, 'steps'):
            # 如果是 Pipeline，取出最后一步的分类器
            estimator = model.steps[-1][1]
        else:
            estimator = model
    except Exception as e:
        st.error(f"⚠️ Year {year}: 模型解析失败 - {e}")
        return

    # =================================================
    # 2. 自动对齐特征顺序 (Feature Alignment)
    # =================================================
    try:
        # 获取模型特征名称
        if hasattr(estimator, 'feature_names_'): 
            model_features = estimator.feature_names_
        elif hasattr(estimator, 'feature_names_in_'): 
            model_features = estimator.feature_names_in_
        else:
            model_features = None

        if model_features is not None:
            model_features = list(model_features) # 确保是列表
            # 补全缺失列
            missing_cols = set(model_features) - set(input_data.columns)
            if missing_cols:
                for c in missing_cols:
                    input_data[c] = 0
            
            # 强制重排
            input_data = input_data[model_features]

    except Exception as e:
        st.warning(f"Feature alignment warning: {e}")

    # =================================================
    # 3. 预测 (Prediction)
    # =================================================
    try:
        # 必须使用完整 model (包含Pipeline) 进行预测
        if hasattr(model, "predict_proba"):
            esrd_prob = model.predict_proba(input_data)[0][1]
            st.write(f"Probability of kidney failure within {year} year: **{esrd_prob:.2%}**")
        else:
            st.warning(f"⚠️ Year {year}: 模型不支持 predict_proba")
            return

    except Exception as e:
        st.error(f"❌ Year {year} 预测出错: {str(e)}")
        # 调试信息：展开查看列名
        with st.expander(f"Debug Info (Year {year})"):
            st.write("Input Columns:", input_data.columns.tolist())
        return

    # =================================================
    # 4. SHAP 解释 (仅针对树模型)
    # =================================================
    try:
        # SHAP 解释器必须用核心模型 (estimator)
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(input_data)

        # [修复2] 兼容不同的 SHAP 返回格式 (List vs Array)
        # Random Forest 通常返回 list [class0, class1]，我们需要 class1
        if isinstance(shap_values, list):
            # 对应的 expected_value 通常也是 list
            base_value = explainer.expected_value[1]
            shap_values_to_plot = shap_values[1]
        else:
            # XGBoost/CatBoost 通常直接返回 array
            base_value = explainer.expected_value
            shap_values_to_plot = shap_values

        # 绘图
        force_plot = shap.force_plot(
            base_value,
            shap_values_to_plot,
            input_data,
            matplotlib=False,
            link="logit" # 可选：如果是概率输出，有时需要 logit link，视模型而定
        )

        html_buffer = io.StringIO()
        shap.save_html(html_buffer, force_plot)
        html_content = html_buffer.getvalue()

        # 渲染
        wrapped = f"""
        <div style='width: 100%; overflow-x: auto; overflow-y: hidden;'>
            <style>
                .shap-force-plot {{ width: 100% !important; }}
                .js-plotly-plot {{ width: 100% !important; }}
            </style>
            {html_content}
        </div>
        """
        components.html(wrapped, height=150, scrolling=True)

    except Exception:
        # 如果是 SVM/KNN 等不支持 SHAP 的模型，或者绘图失败
        # 我们捕获异常但不报错，避免影响概率值的显示
        st.caption(f"ℹ️ (SHAP plot not available for {type(estimator).__name__})")






