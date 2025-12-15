# ==================================================
# 0. 라이브러리
# ==================================================
import streamlit as st
import time
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from wordcloud import WordCloud, STOPWORDS
from konlpy.tag import Okt
from itertools import combinations
from collections import Counter

import koreanize_matplotlib


# 1. 공통 설정
MY_STOPWORDS = {
    "기자", "뉴스", "관련", "이번", "통해", "대한",
    "케데헌", "넷플릭스", "케이팝", "데몬", "헌터스"
}


# 2. WordCloud 함수
def draw_wordcloud(text):
    wc = WordCloud(
        font_path="/System/Library/Fonts/AppleGothic.ttf",
        max_words=50,
        width=800,
        height=800,
        stopwords=STOPWORDS | MY_STOPWORDS,
        background_color="black",
        colormap="coolwarm"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


# 3. 네트워크 시각화 함수
def draw_keyword_network(
    df,
    text_col="description",
    stopwords_path="korean_stopwords.txt",
    min_count=20
):
    texts = df[text_col].dropna().astype(str).tolist()

    # 불용어 로드
    with open(stopwords_path, "r", encoding="utf-8") as f:
        stopwords = set(f.read().splitlines())
    stopwords |= MY_STOPWORDS

    # 명사 추출
    okt = Okt()
    noun_docs = []

    for text in texts:
        text = re.sub(r"[^가-힣\s]", "", text)
        nouns = okt.nouns(text)
        nouns = [n for n in set(nouns) if len(n) > 1 and n not in stopwords]
        noun_docs.append(nouns)

    # Edge 생성
    edges = []
    for nouns in noun_docs:
        if len(nouns) > 1:
            edges.extend(combinations(sorted(nouns), 2))

    edge_counts = Counter(edges)
    filtered_edges = {
        e: c for e, c in edge_counts.items() if c >= min_count
    }

    # Graph
    G = nx.Graph()
    for (n1, n2), w in filtered_edges.items():
        G.add_edge(n1, n2, weight=w)

    # 시각화
    pos = nx.spring_layout(G, seed=42, k=0.3)
    node_sizes = [G.degree(n) * 100 for n in G.nodes()]
    edge_widths = [G[u][v]["weight"] * 0.05 for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(15, 15))
    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        width=edge_widths,
        node_color="skyblue",
        edge_color="gray",
        font_size=12,
        alpha=0.8
    )
    ax.set_title("키워드 네트워크", fontsize=20)
    ax.axis("off")

    return fig



# 4. Seaborn 키워드 빈도 Bar 그래프
def plot_keyword_freq_bar(texts, top_n=20):
    words = " ".join(texts).split()
    freq = Counter(words).most_common(top_n)

    df_freq = pd.DataFrame(freq, columns=["keyword", "count"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_freq,
        x="count",
        y="keyword",
        ax=ax
    )
    ax.set_title("키워드 빈도 Top {}".format(top_n))
    ax.set_xlabel("빈도")
    ax.set_ylabel("키워드")

    return fig

import altair as alt


def plot_article_trend_line(
    df,
    date_col="date"
):
    # 날짜 컬럼 datetime 변환
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # 날짜별 기사 수 집계
    trend_df = (
        df
        .groupby(df[date_col].dt.date)
        .size()
        .reset_index(name="count")
        .rename(columns={date_col: "date"})
    )

    # Altair Line Chart
    chart = (
        alt.Chart(trend_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="날짜"),
            y=alt.Y("count:Q", title="기사 수"),
            tooltip=["date:T", "count:Q"]
        )
        .properties(
            title="날짜별 기사 수 추이",
            width=700,
            height=400
        )
    )

    return chart


import plotly.express as px
from collections import Counter


def plot_keyword_freq_bubble(
    texts,
    top_n=30
):
    # 키워드 빈도 계산
    words = " ".join(texts).split()
    freq = Counter(words).most_common(top_n)

    df_freq = pd.DataFrame(freq, columns=["keyword", "count"])

    # 버블차트
    fig = px.scatter(
        df_freq,
        x="keyword",
        y="count",
        size="count",
        color="count",
        hover_name="keyword",
        size_max=60,
        title="키워드 빈도 Bubble Chart"
    )

    fig.update_layout(
        xaxis_title="키워드",
        yaxis_title="빈도"
    )

    return fig


# 5. Streamlit 페이지 설정
st.set_page_config(
    page_title="송송송의 Streamlit",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("케데헌 데이터 분석")
st.sidebar.divider()

menu = st.sidebar.radio(
    "메뉴",
    ["홈", "워드클라우드", "네트워크시각화", "다양한 그래프"]
)



# 6. 홈
if menu == "홈":
    st.title("😈 K-pop Demon Hunters 😈")

    if st.button("풍선을 띄워보세요"):
        st.balloons()



# 7. 워드클라우드
elif menu == "워드클라우드":
    st.title("워드클라우드")

    if st.button("로드"):
        with st.spinner("로딩 중..."):
            df = pd.read_csv("Demons.csv")
            text = " ".join(df["title"].dropna().astype(str))
            fig = draw_wordcloud(text)
            time.sleep(1)

        st.success("완료")
        st.pyplot(fig)



# 8. 네트워크 시각화
elif menu == "네트워크시각화":
    st.title("네트워크 시각화")

    if st.button("로드"):
        with st.spinner("네트워크 생성 중..."):
            df = pd.read_csv("Demons.csv")
            fig = draw_keyword_network(df)
            time.sleep(1)

        st.success("완료")
        st.pyplot(fig)


# 9. 다양한 그래프
elif menu == "다양한 그래프":
    st.title("다양한 그래프")

    # 데이터 로드
    df = pd.read_csv("Demons.csv")
    texts = df["title"].dropna().astype(str).tolist()


    # 키워드 빈도 Bar 그래프
    st.subheader("키워드 빈도 Bar 그래프")
    st.pyplot(
        plot_keyword_freq_bar(
            texts,
            top_n=20
        )
    )

    # 날짜별 기사 수 Line 그래프
    st.subheader(" 날짜별 기사 수 추이")
    st.altair_chart(
        plot_article_trend_line(
            df,
            date_col="pubDate"  
        ),
        use_container_width=True
    )

    # 3️⃣ 키워드 빈도 Bubble 그래프
    st.subheader("🫧 키워드 빈도 Bubble 그래프")
    st.plotly_chart(
        plot_keyword_freq_bubble(
            texts,
            top_n=30
        ),
        use_container_width=True
    )
