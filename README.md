---
title: LinkedIn Post Analyzer
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
---

# LinkedIn Post Analyzer

A powerful tool for analyzing LinkedIn posts to gain insights into content performance and engagement patterns. Upload your LinkedIn posts data and get AI-powered insights about your content strategy.

## Features

- **Data Analysis**: Upload your LinkedIn posts data (CSV/Excel) and get comprehensive analysis
- **AI-Powered Insights**: Get detailed insights about post performance, engagement, and content patterns
- **Interactive Chat**: Ask questions about your data and get specific answers
- **Export Results**: Download enhanced data with additional metrics and detailed summaries

## How to Use

1. **Prepare Your Data**
   - Export your LinkedIn posts data to CSV or Excel
   - Required columns: Full Post, Reactions, Comment, Reposts
   - Optional columns: Media, Engagement_rate

2. **Upload and Analyze**
   - Click "Upload CSV or Excel"
   - Click "Analyze" to process your data
   - Wait for the analysis to complete

3. **Get Insights**
   - Download the enhanced Excel file and summary
   - Use the chat interface to ask specific questions
   - Try example questions or ask your own

## Example Questions

- "What are the characteristics of high-performing posts?"
- "Which content structures and formats work best?"
- "What will be your top 3 recommendations to improve my engagement?"
- "What is the breakdown of posts across topics for the top 10% performing posts?"

## Note

This app requires an OpenAI API key to function. Make sure to add your API key in the Hugging Face Space settings under "Secrets" with the key name `OPENAI_API_KEY`. 