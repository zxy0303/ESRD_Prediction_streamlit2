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
        1: joblib.load('./selected_features/catboost_1yr.pkl'),
        3: joblib.load('./selected_features/catboost_3yr.pkl'),
        5: joblib.load('./selected_features/catboost_5yr.pkl')
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
    ("12 Features (Specialized model)", "9 Features (General model)"),
    horizontal=True
)

is_full_mode = (model_mode == "12 Features (Specialized model)")

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
    # 1. 自动对齐特征顺序
    # =================================================
    model_features = None
    if hasattr(model, 'feature_names_'): model_features = model.feature_names_
    elif hasattr(model, 'feature_names_in_'): model_features = model.feature_names_in_
    elif hasattr(model, 'steps'): # Pipeline
        try:
            final_estimator = model.steps[-1][1]
            if hasattr(final_estimator, 'feature_names_'): model_features = final_estimator.feature_names_
            elif hasattr(final_estimator, 'feature_names_in_'): model_features = final_estimator.feature_names_in_
        except: pass

    if model_features is not None:
        try:
            missing_cols = set(model_features) - set(input_data.columns)
            if missing_cols:
                for c in missing_cols: input_data[c] = 0
            input_data = input_data[model_features]
        except KeyError as e:
            st.error(f"❌ 特征对齐失败: {e}")
            return

    # =================================================
    # 2. 预测概率
    # =================================================
    try:
        esrd = model.predict_proba(input_data)[0][1]
        st.write(f"Probability of kidney failure within {year} year: **{esrd:.2%}**")
    except Exception as e:
        st.error(f"预测发生错误: {e}")
        return

    # =================================================
    # 3. 计算 SHAP 值
    # =================================================
    shap_model = model
    if hasattr(model, 'steps'): shap_model = model.steps[-1][1]

    try:
        explainer = shap.TreeExplainer(shap_model)
        # check_additivity=False 可以防止某些浮点数精度报错
        shap_values_raw = explainer.shap_values(input_data, check_additivity=False)
        base_value_raw = explainer.expected_value
    except Exception as e:
        st.warning(f"无法生成 SHAP 数据: {e}")
        return

    # =================================================
    # 4. 智能维度清洗 (Smart Shape Handling)
    # =================================================
    try:
        # --- A. 准备特征元数据 ---
        final_feature_values = input_data.iloc[0].values.flatten().astype(float)
        final_feature_names = input_data.columns.tolist()
        n_features = len(final_feature_names)

        # --- B. 处理 SHAP Values (核心修复) ---
        shap_val = shap_values_raw
        
        # 1. 预处理 List (RandomForest 常见)
        # 如果是 [array_class0, array_class1]，取第二个
        if isinstance(shap_val, list):
            if len(shap_val) > 1:
                shap_val = shap_val[1]
            else:
                shap_val = shap_val[0]
        
        # 转换为 Numpy以便操作
        shap_val = np.array(shap_val)

        # 2. 精确形状匹配 (不再盲目 Flatten)
        # 情况 1: (1, 12, 2) -> 1样本, 12特征, 2类别 (Sklearn RF 常见)
        if shap_val.ndim == 3 and shap_val.shape[-1] == 2:
            shap_val = shap_val[0, :, 1] # 取第0样本，所有特征，第1类别
        
        # 情况 2: (1, 12) -> 标准二分类输出
        elif shap_val.ndim == 2 and shap_val.shape[0] == 1:
            shap_val = shap_val[0, :]
            
        # 情况 3: (12, 2) -> 某些特殊情况 squeeze 后
        elif shap_val.ndim == 2 and shap_val.shape[1] == 2 and shap_val.shape[0] == n_features:
            shap_val = shap_val[:, 1] # 所有特征，第1类别

        # 情况 4: (2, 12) -> 类别在前，特征在后
        elif shap_val.ndim == 2 and shap_val.shape[0] == 2 and shap_val.shape[1] == n_features:
            shap_val = shap_val[1, :]

        # 3. 最终兜底：强制拉平并截断
        shap_val = np.squeeze(shap_val)
        if shap_val.ndim > 1: 
             shap_val = shap_val.flatten()
        
        # 如果经过上述处理，长度还是不对 (比如 24)，说明上面没匹配到，做最后的抢救
        if shap_val.size == 2 * n_features:
             # 假设是 (Features, 2) 这种交替排列
             # 取后半部分通常不安全，我们尝试取奇数位 (0, 2, 4 是类0; 1, 3, 5 是类1)
             # 但为了安全，这里还是建议检查前面的逻辑。
             # 作为一个强力补丁：
             shap_val = shap_val[-n_features:] 

        final_shap_values = shap_val

        # --- C. 处理 Base Value ---
        base_val = base_value_raw
        if isinstance(base_val, (list, np.ndarray)):
            base_val = np.array(base_val).flatten()
            if base_val.size > 1:
                base_val = base_val[-1] # 取正类
            else:
                base_val = base_val[0]
        final_base_value = float(base_val)

        # --- D. 维度校验 ---
        if final_shap_values.shape[0] != n_features:
             st.warning(f"⚠️ 维度校验未通过: SHAP shape {final_shap_values.shape} != Features {n_features}")
             return

        # =================================================
        # 5. 绘图
        # =================================================
        force_plot = shap.force_plot(
            final_base_value,
            final_shap_values,
            features=final_feature_values,
            feature_names=final_feature_names,
            matplotlib=False
        )

        html_buffer = io.StringIO()
        shap.save_html(html_buffer, force_plot)
        html_content = html_buffer.getvalue()

        wrapped = f"""
        <div style='width: 100%; overflow-x: auto; overflow-y: hidden;'>
            <style>
                .shap-force-plot {{ width: 100% !important; }}
                .js-plotly-plot {{ width: 100% !important; }}
            </style>
            {html_content}
        </div>
        """
        components.html(wrapped, height=140, scrolling=True)
    
    except Exception as e:
        st.error(f"⚠️ 图表渲染失败: {e}")


with right_col:
    st.subheader("🤖 Predicted Results")
    if predict_btn:
        try:
            current_models = models_12 if is_full_mode else models_9

            render_prediction(current_models[1], input_data, 1)
            render_prediction(current_models[3], input_data, 3)
            render_prediction(current_models[5], input_data, 5)

        except Exception as e:
            st.error(f"Error: {e}")
            # 调试辅助：如果报错，打印当前 DataFrame 的列名，方便对比模型需求

            st.write("Current Input Columns:", input_data.columns.tolist())











