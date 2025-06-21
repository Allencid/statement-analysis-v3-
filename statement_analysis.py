import streamlit as st
import openai
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud
from snownlp import SnowNLP
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
import ast
import os

# 📌 建立 OpenAI 客戶端
from dotenv import load_dotenv
import os
load_dotenv()  # 讀取 .env 檔案
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

# 📌 Streamlit 標題
st.title("📊 中文陳述資料分析工具")

# 📌 輸入陳述資料
statement_text = st.text_area("請貼上你的中文陳述資料：")

# 📌 選擇主題數目
num_topics = st.number_input("請輸入要分幾個主題（至少 3 個）", min_value=3, value=3, step=1)

# 📌 定義 LLM 分析函式（必須放在 if 外面）
def analyze_statement_to_timeline(statement, num_topics):
    prompt = f"""
你是一位陳述資料分析師，以下是一份中文陳述資料。
請根據內容與時間順序，將資料歸類為 {num_topics} 個主題。
每個主題請自動命名，並列出該主題下依照時序的事件摘要。

⚠️ 回覆時只提供符合範例格式的 JSON 字串，勿加任何說明文字。

回覆格式：
[
  {{
    "主題": "主題名稱",
    "事件列表": [
      "時間 + 事件內容",
      ...
    ]
  }},
  ...
]

陳述資料：
{statement}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# 📌 執行分析按鈕
if st.button("分析陳述資料") and statement_text:

    # 📌 執行分析
    result_text = analyze_statement_to_timeline(statement_text, num_topics)

    # 📌 處理回傳內容
    if result_text.startswith("```json"):
        result_text = result_text[7:]
    if result_text.endswith("```"):
        result_text = result_text[:-3]

    result_data = ast.literal_eval(result_text)

    # 📌 整理成時序表 DataFrame
    rows = []
    for topic in result_data:
        for event in topic['事件列表']:
            rows.append({"主題": topic['主題'], "事件": event})

    df = pd.DataFrame(rows)
    st.subheader("📋 時序表")
    st.dataframe(df)

    # 📊 彙總每個主題的文字內容與字數、詞數
    topic_text_summary = df.groupby('主題')['事件'].apply(lambda x: ' '.join(x)).reset_index()
    topic_text_summary['總字數'] = topic_text_summary['事件'].apply(len)
    topic_text_summary['總詞數'] = topic_text_summary['事件'].apply(lambda x: len(SnowNLP(x).words))

    st.subheader("📊 主題字數與詞數統計")
    st.dataframe(topic_text_summary)

    # 📈 繪圖：字數折線圖
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=topic_text_summary['主題'],
        y=topic_text_summary['總字數'],
        mode='lines+markers',
        line=dict(color='royalblue')
    ))
    fig_line.update_layout(title='主題總字數折線圖', xaxis_title='主題', yaxis_title='總字數')
    st.plotly_chart(fig_line)

    # 📈 繪圖：詞數折線圖
    fig_wordcount = go.Figure()
    fig_wordcount.add_trace(go.Scatter(
        x=topic_text_summary['主題'],
        y=topic_text_summary['總詞數'],
        mode='lines+markers',
        line=dict(color='orange')
    ))
    fig_wordcount.update_layout(title='主題總詞數折線圖', xaxis_title='主題', yaxis_title='總詞數')
    st.plotly_chart(fig_wordcount)

    # 📌 CKIP 斷詞 + 詞性標註
    ws = CkipWordSegmenter(model="bert-base")
    pos = CkipPosTagger(model="bert-base")
    word_segments = ws([statement_text])
    pos_tags = pos(word_segments)

    words = word_segments[0]
    tags = pos_tags[0]
    df_pos = pd.DataFrame({"詞": words, "詞性": tags})

    st.subheader("📊 詞性分布統計表")
    pos_count = df_pos["詞性"].value_counts().reset_index()
    pos_count.columns = ["詞性", "數量"]
    st.dataframe(pos_count)

    # 📊 詞性分布直方圖
    st.subheader("📈 詞性分布直方圖")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(pos_count["詞性"], pos_count["數量"], color='skyblue')
    plt.xticks(rotation=45)
    st.pyplot(fig)

else:
    st.info("請貼上陳述資料，並點選【分析陳述資料】。")