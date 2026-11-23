import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os, re, json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 提取json工具函数
def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise Exception("没有找到JSON")

class AgentState(TypedDict):
    df_raw: pd.DataFrame
    df_clean: pd.DataFrame | None
    detect_report: dict
    clean_plan: dict
    review_result: str
    log: list[str]

def detector_agent(state: AgentState):
    df = state["df_raw"]
    report = {}
    report["missing"] = df.isnull().sum().to_dict()
    report["duplicate_count"] = int(df.duplicated().sum())
    report["columns"] = df.columns.tolist()
    prompt = f"""
你是数据检测专家，根据下面脏数据检测报告，输出JSON格式清洗方案。
报告：{report}
允许操作：drop_duplicate, fill_median, fill_const, drop_row, filter_outlier
输出严格只返回JSON，格式：{{"action_list": [{{"col":"xxx","op":"xxx"}}]}}
"""
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role":"user","content":prompt}]
    )
    plan = extract_json(resp.choices[0].message.content)
    state["detect_report"] = report
    state["log"].append("【侦察智能体】完成脏数据扫描")
    return state

def planner_agent(state: AgentState):
    report = state["detect_report"]
    prompt = f"""
基于脏数据报告 {report}，生成清洗方案JSON，action_list里面每个元素包含col、op。
可选op：drop_duplicate, fill_median, fill_const, drop_row, filter_outlier
只输出JSON，不要解释。
"""
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role":"user","content":prompt}]
    )
    plan = extract_json(resp.choices[0].message.content)
    state["clean_plan"] = plan
    state["log"].append("【决策智能体】生成清洗方案")
    return state

def executor_agent(state: AgentState):
    df = state["df_raw"].copy()
    plan = state["clean_plan"]
    for act in plan["action_list"]:
        col = act["col"]
        op = act["op"]
        if op == "drop_duplicate":
            df = df.drop_duplicates()
        elif op == "fill_median":
            df[col] = df[col].fillna(df[col].median())
        elif op == "fill_const":
            df[col] = df[col].fillna("未知")
        elif op == "drop_row":
            df = df.dropna(subset=[col])
        elif op == "filter_outlier":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]
    state["df_clean"] = df
    state["log"].append("【执行智能体】执行清洗操作")
    return state

def review_agent(state: AgentState):
    df = state["df_clean"]
    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        state["review_result"] = "fail"
        state["log"].append(f"【审核智能体】质检失败，还有缺失值{total_missing}个，返回重新规划")
    else:
        state["review_result"] = "pass"
        state["log"].append("【审核智能体】质检通过，无缺失值")
    return state

def decide_next(state: AgentState):
    if state["review_result"] == "pass":
        return "end"
    else:
        return "planner"

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("detector", detector_agent)
    graph.add_node("planner", planner_agent)
    graph.add_node("executor", executor_agent)
    graph.add_node("review", review_agent)
    graph.set_entry_point("detector")
    graph.add_edge("detector", "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "review")
    graph.add_conditional_edges(
        "review",
        decide_next,
        {"end": END, "planner": "planner"}
    )
    return graph.compile()

st.set_page_config(page_title="多智能体数据清洗平台", layout="wide")
st.title("基于LangGraph+智谱GLM 多智能体数据清洗")
st.subheader("上传CSV → AI自动检测 → 生成方案 → 清洗 → 质检 → 下载")

uploaded_file = st.file_uploader("上传你的csv文件", type=["csv"])
if uploaded_file is not None:
     df_raw = pd.read_csv(uploaded_file)
     st.markdown("### 原始数据")
     st.dataframe(df_raw, use_container_width=True)
     if st.button("启动AI清洗流水线"):
         runner = build_graph()
         init_state = {
             "df_raw": df_raw,
             "df_clean": None,
             "detect_report": {},
             "clean_plan": {},
             "review_result": "",
             "log": []
         }
         with st.spinner("智能体正在运行，请稍等..."):
             res = runner.invoke(init_state)
         st.success("流水线执行完成！")
         st.markdown("### 运行日志")
         for line in res["log"]:
             st.write("- " + line)
         st.markdown("### 清洗后数据")
         st.dataframe(res["df_clean"], use_container_width=True)
         csv_download = res["df_clean"].to_csv(index=False, encoding="utf-8-sig")
         st.download_button("下载清洗后的CSV", data=csv_download, file_name="cleaned_result.csv", mime="text/csv")