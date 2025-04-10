import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly


st.set_page_config(page_title="Dashboard Barbearia", layout="wide")
st.title("📊 Dashboard de Vendas - Barbearia")

# --- Carrega os dados ---
df = pd.read_csv("dados.csv", parse_dates=["data"])

@st.cache_data
def treinar_modelo_prophet(df):
    df_treino = df[["data", "vendas"]].rename(columns={"data": "ds", "vendas": "y"})
    modelo = Prophet(yearly_seasonality=False)  # Removendo sazonalidade anual
    modelo.fit(df_treino)
    
    futuro = modelo.make_future_dataframe(periods=30)
    previsao = modelo.predict(futuro)
    
    return modelo, previsao



# --- Pré-processamento para gráficos mensais ---
df["mes"] = df["data"].dt.to_period("M").astype(str)

# --- Menu lateral ---
grafico = st.sidebar.selectbox(
    "Selecione o gráfico:",
    (
        "Vendas por Mês",
        "Média de Clientes por Mês (% Promoção)",
        "Clientes por Dia da Semana",
        "Vendas: Com x Sem Promoção",
        'Previsão de Vendas (Prophet)'
    )
)

# --- Gráfico 1: Vendas por mês ---
if grafico == "Vendas por Mês":
    vendas_mensais = df.groupby("mes")["vendas"].sum().reset_index()
    fig = px.bar(vendas_mensais, x="mes", y="vendas", title="Vendas Totais por Mês", labels={"mes": "Mês", "vendas": "Vendas (R$)"})
    st.plotly_chart(fig, use_container_width=True)

# --- Gráfico 2: Média de clientes por mês com cor por promoção ---
elif grafico == "Média de Clientes por Mês (% Promoção)":
    mensal = df.groupby("mes").agg({"clientes": "mean", "promocao": "mean"}).reset_index()
    mensal["promo_percent"] = (mensal["promocao"] * 100).round(1)
    fig = px.bar(mensal, x="mes", y="clientes", color="promo_percent",
                 color_continuous_scale="Blues",
                 title="Média de Clientes por Mês (Cor = % Promoções)",
                 labels={"clientes": "Clientes", "promo_percent": "% Promoção"})
    st.plotly_chart(fig, use_container_width=True)

# --- Gráfico 3: Clientes por dia da semana ---
elif grafico == "Clientes por Dia da Semana":
    df["dia_semana"] = df["data"].dt.day_name()
    media_dia = df.groupby("dia_semana")["clientes"].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    fig = px.bar(x=media_dia.index, y=media_dia.values, title="Média de Clientes por Dia da Semana", labels={"x": "Dia", "y": "Clientes"})
    st.plotly_chart(fig, use_container_width=True)

# --- Gráfico 4: Vendas com e sem promoção ---
elif grafico == "Vendas: Com x Sem Promoção":
    df["tipo_dia"] = df["promocao"].map({1: "Com Promoção", 0: "Sem Promoção"})
    fig = px.box(df, x="tipo_dia", y="vendas", title="Distribuição de Vendas - Com ou Sem Promoção")
    st.plotly_chart(fig, use_container_width=True)

# --- Gráfico 5: Previsão de Vendas ---
elif grafico == "Previsão de Vendas (Prophet)":
    with st.spinner("Treinando modelo..."):
        modelo, previsao = treinar_modelo_prophet(df)

    st.subheader("Previsão de Vendas para os Próximos 30 Dias")
    fig = plot_plotly(modelo, previsao)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Componentes da Previsão")
    fig2 = modelo.plot_components(previsao)
    st.pyplot(fig2)
