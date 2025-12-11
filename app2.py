import streamlit as st
import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import io
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
    # =================================================
    # 1. 提取核心模型 (Handle Pipeline)
    # =================================================
    # 如果模型是 Pipeline (包含 steps 属性)，我们需要取出最后一步的分类器
    # 因为 SHAP TreeExplainer 无法直接解释 Pipeline
    if hasattr(model, 'steps'):
        # 取出 pipeline 的最后一步，格式通常是 [('step_name', step_object), ...]
        estimator = model.steps[-1][1]
    else:
        estimator = model

    # =================================================
    # 2. 自动对齐特征顺序 (Feature Alignment)
    # =================================================
    try:
        # 优先从核心模型(estimator)中获取特征名称
        # CatBoost/XGBoost 使用 feature_names_, Sklearn 使用 feature_names_in_
        if hasattr(estimator, 'feature_names_'): 
            model_features = estimator.feature_names_
        elif hasattr(estimator, 'feature_names_in_'): 
            model_features = estimator.feature_names_in_
        else:
            # 如果找不到特征名属性，暂时跳过对齐（可能会在预测时报错）
            model_features = None

        if model_features is not None:
            # 补齐输入数据中缺失的列（如果有的话），填 0
            missing_cols = set(model_features) - set(input_data.columns)
            if missing_cols:
                for c in missing_cols:
                    input_data[c] = 0
            
            # 关键：强制按照模型训练时的特征顺序重排输入数据
            input_data = input_data[model_features]

    except AttributeError:
        st.warning("⚠️ 无法读取模型特征顺序，正在尝试使用默认顺序。")
    except KeyError as e:
        st.error(f"❌ 数据对齐失败，缺少特征: {e}")
        return

    # =================================================
    # 3. 预测与 SHAP 解释
    # =================================================
    try:
        # 1. 预测概率：必须使用完整的 model (Pipeline)，以保证预处理步骤（如存在）被执行
        esrd = model.predict_proba(input_data)[0][1]
        
        st.write(f"Probability of kidney failure within {year} year: **{esrd:.2%}**")

        # 2. SHAP 解释：必须使用核心模型 (estimator)，因为 TreeExplainer 只认树模型
        explainer = shap.TreeExplainer(estimator)
        
        # 计算 SHAP 值
        shap_values = explainer.shap_values(input_data)

        # 3. 绘图
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[0] if isinstance(shap_values, list) else shap_values, # 兼容不同版本返回值
            input_data,
            matplotlib=False,
        )

        html_buffer = io.StringIO()
        shap.save_html(html_buffer, force_plot)
        html_content = html_buffer.getvalue()

        component_height = 140
        wrapped = f"""
        <div style='width: 100%; overflow-x: auto; overflow-y: hidden;'>
            <style>
                .shap-force-plot {{ width: 100% !important; }}
                .js-plotly-plot {{ width: 100% !important; }}
            </style>
            {html_content}
        </div>
        """
        components.html(wrapped, height=component_height, scrolling=True)

    except Exception as e:
        st.error(f"Error in processing: {e}")
        # 调试信息：如果报错，显示当前处理的模型类型和列名，方便排查
        st.write(f"Debug Info - Model Type: {type(model)}")
        st.write(f"Debug Info - Estimator Type: {type(estimator)}")
        st.write("Debug Info - Input Columns:", input_data.columns.tolist())





