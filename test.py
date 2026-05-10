import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt

# ---------------- 1. 页面配置 ----------------
st.set_page_config(page_title="AKI 风险预测系统", layout="wide", page_icon="🏥")

# ---------------- 2. 加载模型 ----------------
@st.cache_resource
def load_model_and_meta():
    model = CatBoostClassifier()
    model.load_model("CatBoost_model_py.cbm")  # 模型文件与脚本在同目录
    expected_features = model.feature_names_
    cat_features = model.get_cat_feature_indices()
    return model, expected_features, cat_features

try:
    model, expected_features, cat_indices = load_model_and_meta()
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()

# ---------------- 3. 侧边栏输入控件（带单位） ----------------
st.sidebar.header("输入患者临床参数")

def get_input():
    inputs = {}

    # 连续变量
    inputs["BUN"] = st.sidebar.number_input("BUN (mg/dL)", value=20.0, step=1.0)
    inputs["Cr"] = st.sidebar.number_input("Cr (Creatinine, mg/dL)", value=1.0, step=0.1)
    inputs["SOFA"] = st.sidebar.number_input("SOFA 评分", value=2.0, step=1.0)
    inputs["Lactate"] = st.sidebar.number_input("Lactate (mmol/L)", value=2.0, step=0.1)
    inputs["RR"] = st.sidebar.number_input("呼吸频率 (breaths/min)", value=20, step=1)
    inputs["Alb"] = st.sidebar.number_input("白蛋白 (g/dL)", value=3.5, step=0.1)
    inputs["Glu"] = st.sidebar.number_input("血糖 (mmol/mL)", value=5.5, step=0.1)
    inputs["Weight"] = st.sidebar.number_input("体重 (kg)", value=65.0, step=1.0)
    inputs["SpO2"] = st.sidebar.number_input("血氧饱和度 (%)", value=98, step=1)

    # 二分类变量
    inputs["T2DM"] = str(int(st.sidebar.checkbox("是否存在 T2DM?", value=False)))
    inputs["HF"] = str(int(st.sidebar.checkbox("是否存在心力衰竭 (HF)?", value=False)))

    # 严格按照模型训练特征顺序生成 DataFrame
    df = pd.DataFrame([inputs])[expected_features]
    return df

input_df = get_input()

# ---------------- 4. 主界面展示 ----------------
st.title("🏥 老年肠梗阻患者 AKI 风险预测")
st.write("左侧录入体征数据，点击下方按钮获取 AI 风险评估及模型解释。")

if st.button("开始预测 (Predict)", type="primary"):
    try:
        # 预测概率
        prob = model.predict_proba(input_df)[0][1]

        # 结果看板
        st.divider()
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("评估结果")
            st.metric("AKI 风险概率", f"{prob:.2%}")
            # 风险分级
            if prob > 0.7:
                st.error("⚠️ 高风险 (High Risk)")
            elif prob > 0.3:
                st.warning("⚠️ 中风险 (Medium Risk)")
            else:
                st.success("✅ 低风险 (Low Risk)")

            st.write("**当前输入参数预览:**")
            st.dataframe(input_df.T.rename(columns={0: "数值"}), use_container_width=True)

        with col2:
            st.subheader("模型决策解释 (SHAP)")
            # SHAP 解释
            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)

            # 绘制瀑布图
            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            st.pyplot(fig)
            st.info("💡 红色条表示该指标增加了 AKI 风险，蓝色条表示降低了风险。")

    except Exception as e:
        st.error(f"预测过程中出错: {e}")
        st.info("建议检查输入数据类型是否与模型训练时一致。")

# ---------------- 5. 开发者调试信息 ----------------
with st.expander("🔍 开发者调试信息"):
    st.write("模型期望的特征顺序:", expected_features)
    st.write("分类特征索引:", cat_indices)
    st.write("当前输入 DataFrame 结构:", input_df)

# ---------------- 6. 底部页脚 ----------------
st.markdown("---")
st.caption("注：本工具仅供临床科研参考，不作为最终诊断依据。")