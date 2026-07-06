# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 11:29:09 2026

@author: milos
"""

from datetime import datetime, timedelta

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# External dependency
# pip install yfinance
import yfinance as yf # yahoo finance
import pandas as pd

# Use local Ollama model, e.g., 'llama3'
local_llm = LLM(model="ollama/gpt-oss:120b-cloud", base_url="http://localhost:11434")

# -----------------------------
# Custom Tool: Stock Data Fetcher
# -----------------------------
@tool("Stock Price Analysis Tool")
def stock_price_analysis(ticker: str) -> str:
    """
    Fetches historical stock data and computes:
    - 30-day moving average
    - 90-day moving average
    - Volatility (std dev of returns)
    - Percentage return over 6 months
    """

    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=180)

        stock = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if stock.empty:
            return f"No data found for ticker {ticker}."

        stock["Daily Return"] = stock["Close"].pct_change()

        current_price = stock["Close"].iloc[-1]
        ma_30 = stock["Close"].rolling(window=30).mean().iloc[-1]
        ma_90 = stock["Close"].rolling(window=90).mean().iloc[-1]
        volatility = stock["Daily Return"].std() * (252 ** 0.5)  # annualized
        pct_return = (
            (stock["Close"].iloc[-1] - stock["Close"].iloc[0])
            / stock["Close"].iloc[0]
        ) * 100

        summary = f"""
Stock Analysis for {ticker.upper()}:

Current Price: ${current_price:.2f}
30-Day Moving Average: ${ma_30:.2f}
90-Day Moving Average: ${ma_90:.2f}
6-Month Return: {pct_return:.2f}%
Annualized Volatility: {volatility:.2%}

Interpretation Hints:
- If current price > moving averages → upward momentum
- High volatility → higher risk
- Strong positive return → bullish trend
"""

        return summary.strip()

    except Exception as e:
        return f"Error analyzing stock {ticker}: {str(e)}"


# -----------------------------
# Agents
# -----------------------------
market_analyst = Agent(
    role="Senior Quantitative Market Analyst",
    goal="Perform deep technical analysis on {ticker} using historical stock data.",
    backstory=(
        "You are a Wall Street quantitative analyst with 15 years of experience "
        "analyzing equities, volatility patterns, and price momentum. "
        "You rely strictly on data-driven insights."
    ),
    tools=[stock_price_analysis],
    verbose=True,
    memory=True,
    llm=local_llm
)

financial_strategist = Agent(
    role="Investment Strategy Advisor",
    goal="Provide clear investment recommendations for {ticker} based on the analysis.",
    backstory=(
        "You are a seasoned portfolio strategist who translates complex market "
        "analysis into actionable investment recommendations for retail and "
        "institutional investors."
    ),
    verbose=True,
    memory=True,
    allow_delegation=False,
    llm=local_llm
)


# -----------------------------
# Tasks
# -----------------------------
analysis_task = Task(
    description=(
        "Use the Stock Price Analysis Tool to analyze {ticker}. "
        "Explain the trend, volatility level, and price momentum clearly.\n\n"
        "Your final answer MUST include:\n"
        "- Trend direction (Bullish / Bearish / Sideways)\n"
        "- Volatility assessment (Low / Moderate / High)\n"
        "- Momentum interpretation\n"
    ),
    expected_output=(
        "A structured technical analysis report with trend, volatility, "
        "and momentum clearly identified."
    ),
    agent=market_analyst,
)

strategy_task = Task(
    description=(
        "Based on the technical analysis provided, create a structured "
        "investment recommendation for {ticker}.\n\n"
        "Your final answer MUST include:\n"
        "- Risk level (Low / Medium / High)\n"
        "- Short-term outlook (1–3 months)\n"
        "- Long-term outlook (6–12 months)\n"
        "- Clear recommendation: Buy / Hold / Sell\n"
        "- Justification based strictly on the analysis\n"
    ),
    expected_output=(
        "A professional investment recommendation report including risk level, "
        "outlook, and a clear Buy/Hold/Sell decision."
    ),
    agent=financial_strategist,
)


# -----------------------------
# Crew
# -----------------------------
crew = Crew(
    agents=[market_analyst, financial_strategist],
    tasks=[analysis_task, strategy_task],
    process=Process.sequential,
    verbose=True,
)


result = crew.kickoff(inputs={"ticker": "AAPL"})
print(result)

