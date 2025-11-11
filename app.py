import streamlit as st
import pandas as pd
import joblib
import time
import warnings
import os

# 页面配置
st.set_page_config(
    page_title="直肠术后LARS风险预测工具",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 初始化session_state
if 'show_form' not in st.session_state:
    st.session_state.show_form = False
if 'has_predicted' not in st.session_state:
    st.session_state.has_predicted = False
if 'pred_result' not in st.session_state:
    st.session_state.pred_result = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None

# -------------------------- 页面标题与身份选择 --------------------------
st.title("直肠术后LARS风险预测工具")
st.divider()

st.subheader("请选择您的身份")
user_type = st.radio(
    "身份类型",
    ["👨‍⚕️ 医护工作者", "👨‍👩‍👧‍👦 患者/家属"],
    horizontal=True,
    key="user_type"
)
st.divider()

# -------------------------- 应用介绍 --------------------------
st.subheader("应用介绍")
st.markdown("""
我们是**四川大学华西医院直肠术后低位前切除综合征（LARS）医护研究团队**，基于临床大数据与随机森林机器学习算法，开发了本风险预测工具。

<font size="4">工具核心用途：预测直肠癌患者<b>术后6个月</b>内发生LARS的风险，为临床医护工作者提供术后管理决策支持，也帮助患者及家属提前了解康复风险，辅助制定个性化自我护理方案。</font>
""", unsafe_allow_html=True)  # 启用HTML支持

# 参考文献
st.markdown("#### 团队相关研究成果")  # 将###改为####减小标题字体
st.markdown("""
<span style="font-size:14px">1. 汪晓东, 黄明君, 李立, 等. 结直肠癌术后LARS预测模型的构建方法及预测系统[P]. 中国专利: ZL 2023 1 0088636.5, 2023-05-02.</span>  
<span style="font-size:14px">2. Ye L, Huang MJ, Huang YW, et al. Risk factors of postoperative low anterior resection syndrome for colorectal cancer: A meta-analysis[J]. Asian Journal of Surgery, 2022, 45: 39-50.</span>  
<span style="font-size:14px">3. 张纯, 林雨昕, 李琳, 等. 直肠癌术后低位前切除综合征的风险预测模型构建：基于随机森林算法[J]. 中国普外基础与临床杂志, 2025, 32(7): 845-852.</span>
""", unsafe_allow_html=True)  # 统一设置文献内容字体大小
st.divider()

# 后续代码保持不变...
# -------------------------- 模型加载 --------------------------
st.subheader("模型加载状态")
model_path = 'lars_risk_model.pkl'
model_loaded = False
model = None

try:
    with st.spinner("模型正在加载中，请稍候..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(model_path)
    st.success("✅ 模型加载成功！点击下方「开始预测」按钮进入输入界面")
    model_loaded = True
    if hasattr(model, 'n_features_in_'):
        st.write(f"🔧 模型要求特征数量：{model.n_features_in_}（当前输入为8个特征，已匹配）")
except Exception as e:
    st.error(f"❌ 模型加载失败：{str(e)}")
    st.warning("预测功能暂时不可用，请检查模型文件是否存在或联系开发者")
    if st.checkbox("查看当前目录文件（调试用）"):
        st.write("当前目录文件列表：", os.listdir('.'))
st.divider()

# 以下代码与原代码一致，省略...
