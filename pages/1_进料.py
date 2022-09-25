import streamlit as st
import time

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

st.header("进料计算模块")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.subheader('进料数值输入')

pot1 = st.number_input("发酵罐1 - 沼气产量：",value=1)
pot2 = st.number_input("发酵罐2 - 沼气产量：",value=1)
pot3 = st.number_input("发酵罐3 - 沼气产量：",value=1)
two_level_pot = st.number_input("二级发酵罐 - 沼气产量：",value=1)
bio_store_pot = st.number_input("沼液储存池 - 沼气产量：",value=1)
two_level_pot_dry = st.number_input("二级发酵罐 - 干物质含量 (%)：",value=1)

st.subheader('产沼气效率')
with st.form("cal1"):
    product_biogas_efficiency = (pot1+pot2+pot3+two_level_pot+bio_store_pot)/two_level_pot_dry
    product_biogas_efficiency = round(product_biogas_efficiency, 2)
    message1 = product_biogas_efficiency

    if st.form_submit_button("计算"):
        #st.subheader(message1)
        st.success("success 🎉")

        r24c1, r24c2, r24c3,r24c4, r24c5, r24c6 = st.columns(6)
        with r24c1:
            st.info("发酵罐1 - 沼气产量")
        with r24c2:
            st.info("发酵罐2 - 沼气产量")
        with r24c3:
            st.info("发酵罐3 - 沼气产量")
        with r24c4:
            st.info("二级发酵罐 - 沼气产量")
        with r24c5:
            st.info("沼液储存池 - 沼气产量")
        with r24c6:
            st.info("二级发酵罐 - 干物质含量 (%)")

        r24b1, r24b2, r24b3,r24b4, r24b5, r24b6 = st.columns(6)
        with r24b1:
            st.info(pot1)
        with r24b2:
            st.info(pot2)
        with r24b3:
            st.info(pot3)
        with r24b4:
            st.info(two_level_pot)
        with r24b5:
            st.info(bio_store_pot)
        with r24b6:
            st.info(two_level_pot_dry)

        n24b1, n24b2, n24b3,n24b4, n24b5, n24b6 = st.columns(6)
        with n24b1:
            st.info("产沼气效率")

        with n24b1:
            st.info(message1)

#=(工艺流程!G21*工艺流程!G22+工艺流程!H21*工艺流程!H22+工艺流程!I21*工艺流程!I22+工艺流程!J21*工艺流程!J22+工艺流程!K21*工艺流程!K22)/K17
st.subheader('产甲烷效率')
with st.form("cal2"):
    st.subheader('甲烷含量数值输入')

    pot1_methane = st.number_input("发酵罐1 - 甲烷含量：", value=1)
    pot2_methane = st.number_input("发酵罐2 - 甲烷含量：", value=1)
    pot3_methane = st.number_input("发酵罐3 - 甲烷含量：", value=1)
    two_level_pot_methane = st.number_input("二级发酵罐 - 甲烷含量：", value=1)
    bio_store_pot_methane = st.number_input("沼液储存池 - 甲烷含量：", value=1)
    two_level_pot_dry_methane = st.number_input("沼液储存池 - 干物质含量 (%)：", value=1)

    product_biogas_efficiency = (pot1*pot1_methane+pot2*pot2_methane+pot3*pot3_methane+two_level_pot*two_level_pot_methane+bio_store_pot*bio_store_pot_methane)/two_level_pot_dry_methane
    product_biogas_efficiency = round(product_biogas_efficiency, 2)
    message2 = product_biogas_efficiency

    if st.form_submit_button("计算",help="点击它"):
        st.success("success 🎉")

        r24c1, r24c2, r24c3, r24c4, r24c5, r24c6 = st.columns(6)
        with r24c1:
            st.info("发酵罐1 - 沼气产量")
        with r24c2:
            st.info("发酵罐2 - 沼气产量")
        with r24c3:
            st.info("发酵罐3 - 沼气产量")
        with r24c4:
            st.info("二级发酵罐 - 沼气产量")
        with r24c5:
            st.info("沼液储存池 - 沼气产量")
        with r24c6:
            st.info("二级发酵罐 - 干物质含量 (%)")

        r24b1, r24b2, r24b3, r24b4, r24b5, r24b6 = st.columns(6)
        with r24b1:
            st.info(pot1)
        with r24b2:
            st.info(pot2)
        with r24b3:
            st.info(pot3)
        with r24b4:
            st.info(two_level_pot)
        with r24b5:
            st.info(bio_store_pot)
        with r24b6:
            st.info(two_level_pot_dry)


        m24c1, m24c2, m24c3, m24c4, m24c5, m24c6 = st.columns(6)
        with m24c1:
            st.info("发酵罐1 - 甲烷含量")
        with m24c2:
            st.info("发酵罐2 - 甲烷含量")
        with m24c3:
            st.info("发酵罐3 - 甲烷含量")
        with m24c4:
            st.info("二级发酵罐 - 甲烷含量")
        with m24c5:
            st.info("沼液储存池 - 甲烷含量")
        with m24c6:
            st.info("二级发酵罐 - 干物质含量 (%)")

        m24b1, m24b2, m24b3, m24b4, m24b5, m24b6 = st.columns(6)
        with m24b1:
            st.info(pot1_methane)
        with m24b2:
            st.info(pot2_methane)
        with m24b3:
            st.info(pot3_methane)
        with m24b4:
            st.info(two_level_pot_methane)
        with m24b5:
            st.info(bio_store_pot_methane)
        with m24b6:
            st.info(two_level_pot_dry_methane)

        # n24c1 = st.columns([1])
        # with n24c1:
        #     st.info("产甲烷效率")
        #
        # n24c1 = st.columns([1])
        # with n24c1:
        #     st.info(message2)
        m24c1, m24c2, m24c3, m24c4, m24c5, m24c6 = st.columns(6)
        with m24c1:
            st.info("产甲烷效率")


        m24b1, m24b2, m24b3, m24b4, m24b5, m24b6 = st.columns(6)
        with m24b1:
            st.info(message2)

