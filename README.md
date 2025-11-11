Quant Studio Pro, a full-stack AI and Machine Learning–powered quantitative trading platform that integrates data engineering, statistical modeling, backtesting, and AI automation inside an interactive Streamlit application.

Project Overview
1. Data Collection & Cleaning
Collected historical BTC-USD data using the Yahoo Finance API (yfinance).
Cleaned, validated, and structured the dataset with Pandas for consistent time-series analysis.

2. Exploratory Data Analysis (EDA)
Analyzed raw BTC price trends, volatility patterns, and initial correlations to understand overall market behavior. 
Visualized RSI behavior and rolling Sharpe/Sortino ratios to identify profitable windows.

3. Feature Engineering & Data Preparation
Engineered indicators including SMA10, SMA20, SMA_gap, RSI14, Ret1/5/10, and Vol10/20.
Removed initialization NaNs, standardized inputs, and prepared aligned datasets for modeling.

4. Strategy 1 — Trend Pilot (SMA 10/20)
Implemented rule-based trading: Buy when SMA10 > SMA20, Sell on cross-down.
Integrated realistic constraints: 3% stop-loss, 10 bps trade fee, volatility gate, and 50/200 SMA trend filter.

5. Strategy 2 — Smart Filter AI (SMA + ML)
Built a Logistic Regression model using engineered features to predict next-day returns.
Executed trades only when SMA and ML signals agreed, improving precision and reducing false trades.

6. Systematic Investment Plan (SIP / DCA)
Designed a module for Systematic Investing across daily, weekly, and monthly frequencies.
Evaluated performance of passive compounding vs. active trading strategies.

7. AI Agent Integration
Integrated an AI Research Assistant powered by OpenAI GPT / Local LLM.
The agent can execute backtests, run SIP simulations, compare strategies, and explain performance interactively through natural language.

Tech Stack
Python · Pandas · NumPy · Scikit-Learn · Plotly · Streamlit · AI API · Ngrok

Domains Covered
Data Engineering · Feature Engineering · Machine Learning · Financial Modeling · Quantitative Research · AI Integration · Visualization · MLOps

Takeaway
Quant Studio Pro shows how data-driven trading, machine learning, and AI can make financial research more transparent and interactive. Strategy comparisons reveal that SMA provides steady trend-based growth, SMA + ML improves risk control with smaller drawdowns, while SIP outperforms all with higher long-term returns through disciplined compounding—highlighting the trade-off between active precision and passive consistency.

This project strengthened my skills in quantitative modeling, financial analytics, and applied AI, aligning perfectly with my long-term goal of working at top-tier quantitative trading and fintech firms
