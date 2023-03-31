import streamlit as st

st.set_page_config(layout="wide")

st.sidebar.title("沼气学术会议")
st.sidebar.info(
    """
    中国沼气学会学术年会:     
    <http://www.biogaschina.com.cn>         
    能源系统与电气工程会议:       
    <https://www.ncsti.gov.cn>
    """
)

st.sidebar.title("友情链接")
st.sidebar.info(
    """
    国际沼气网: <http://www.biogasintel.com>       
    中国农业农村部: <http://www.moa.gov.cn>          
    香港可再生能源网: <https://re.emsd.gov.hk>
    
    """
)

st.title("沼气工程智能计算平台")

st.markdown(
    """
        由于沼气工程的实际运行数据量很大，两个国家的沼气工程的数据结构、数据单位、标准范围等等也可能不同，为此，必须建立用计算机来分析比对海量非结构化数据的方法。
    为此，应构建基于机器学习算法的典型重要指标的聚类分析算法，使程序能够快速给出主要参数的范围，并且按照输入的要求的不同，给出对应的结果。
    这部分是将单一数据和判定依据（判据）以及输出结果相关联（或者说是形成数学上的直接映射的关系）的部分。

    """
)

st.info("使用清洁能源，守护地球家园人人有责")

st.subheader("全球气候变化趋势")
st.markdown(
    """
    我们正处在悬崖边缘，很可能错失实现1.5℃温控目标的机会"——《2019年排放差距报告》。尽管COVID-19的全球蔓延导致二氧化碳排放量出现短暂下降，但本世纪世界仍在朝着3℃以上的灾难性温度上升，各国仍需加速二氧化碳减排，以实现《巴黎协定》的承诺。"""
)

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.image("https://github.com/giswqs/data/raw/main/timelapse/spain.gif")
    st.image("https://github.com/giswqs/data/raw/main/timelapse/las_vegas.gif")

with row1_col2:
    st.image("https://github.com/giswqs/data/raw/main/timelapse/goes.gif")
    st.image("https://github.com/giswqs/data/raw/main/timelapse/fire.gif")
