import gradio as gr
import pandas as pd
from helpers import (
    process_dataframe,
    generate_summary,
    setup_langchain,
    batch_analyze_posts,
    word_count,
    calculate_reading_ease,
    setup_safe_analysis_agent,
    determine_media_type
)
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.llms import OpenAI
import os
from langchain_openai import ChatOpenAI
import numpy as np
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Add these imports to the top of your main file
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
import textwrap 
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool

# Make sure environment variables are loaded
load_dotenv()

# Define the data loading function FIRST, before it's used
def preprocess_data(df):
    """
    Preprocess the uploaded dataframe
    
    Args:
        df: pandas DataFrame
    
    Returns:
        pandas DataFrame: Preprocessed LinkedIn post data
    """
    # Convert engagement_rate to numeric if it exists
    if 'Engagement_rate' in df.columns:
        df['Engagement_rate'] = pd.to_numeric(df['Engagement_rate'], errors='coerce')
    elif 'Engagement Rate' in df.columns:
        df['Engagement_rate'] = pd.to_numeric(df['Engagement Rate'], errors='coerce')
        df = df.rename(columns={'Engagement Rate': 'Engagement_rate'})
    
    # Handle numeric columns
    numeric_columns = ['Reactions', 'Comment', 'Reposts', 'Word Count', 'Flesch Reading Ease']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN values with appropriate defaults
    df = df.fillna({
        'Reactions': 0,
        'Comment': 0,
        'Reposts': 0,
        'Engagement_rate': 0
    })
    
    # Print column information for debugging
    print(f"Available columns: {df.columns.tolist()}")
    
    return df

def analyze_file(file):
    try:
        if file.name.endswith('.csv'):
            try:
                df = pd.read_csv(file.name, encoding='latin-1')
            except Exception as e:
                df = pd.read_csv(file.name, encoding='iso-8859-1')
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            df = pd.read_excel(file.name)
        else:
            return "Unsupported file format. Please upload a CSV or Excel file.", None, None, None, None

        # Debug information
        original_row_count = len(df)
        print(f"Original file has {original_row_count} rows")
        
        # Process DataFrame using batch processing
        analysis_results = batch_analyze_posts(df['Full Post'].tolist())
        
        # Create a DataFrame from analysis results
        results_df = pd.DataFrame(analysis_results)
        
        # Combine with original DataFrame
        processed_df = pd.concat([df, results_df], axis=1)
        
        # Add Word Count and Reading Ease calculations
        processed_df['Word Count'] = processed_df['Full Post'].apply(word_count)
        processed_df['Flesch Reading Ease'] = processed_df['Full Post'].apply(calculate_reading_ease)
        
        # Add Media Creative analysis
        if 'Media' in processed_df.columns:
            processed_df['Media Creative'] = processed_df['Media'].apply(determine_media_type)
            print("Added Media Creative column")  # Debug print
        
        # Convert Engagement_rate to numeric if it exists
        if 'Engagement_rate' in processed_df.columns:
            try:
                processed_df['Engagement_rate'] = pd.to_numeric(processed_df['Engagement_rate'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
                print("Found and processed Engagement_rate column")
            except Exception as e:
                print(f"Warning: Could not process Engagement_rate column: {str(e)}")

        # Generate Markdown Summary
        markdown_summary = generate_summary(processed_df)

        # Save the updated Excel file
        updated_file = "updated_li_posts.xlsx"
        processed_df.to_excel(updated_file, index=False)

        # Save the markdown summary
        summary_file = "summary.md"
        with open(summary_file, 'w') as f:
            f.write(markdown_summary)

        return (
            f"Analysis complete! Processed {len(processed_df)} out of {original_row_count} posts.", 
            updated_file, 
            summary_file,
            processed_df,
            markdown_summary
        )

    except Exception as e:
        return f"An error occurred: {str(e)}", None, None, None, None

def chat_with_data(question, df, chat_history=None, summary=None):
    try:
        if df is None:
            return [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "Please analyze a file first."}
            ]
            
        # Create a copy to work with
        df = df.copy()
        
        # Ensure key columns are numeric
        for col in ['Reactions', 'Comment', 'Reposts', 'Word Count', 'Flesch Reading Ease', 'Engagement_rate']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
        # Create data context
        data_details = []
        data_details.append(f"## LinkedIn Post Analysis Data\n")
        data_details.append(f"General Statistics:")
        data_details.append(f"- Total posts analyzed: {len(df)}")
        
        # Add engagement metrics including Engagement_rate if it exists
        metrics = {
            'Reactions': 'reactions',
            'Comment': 'comments', 
            'Reposts': 'reposts',
            'Engagement_rate': 'Engagement rate'
        }
        
        for col, label in metrics.items():
            if col in df.columns:
                data_details.append(f"\n{label.title()} Metrics:")
                data_details.append(f"- Highest {label}: {df[col].max():.1f}")
                data_details.append(f"- Average {label}: {df[col].mean():.1f}")
                data_details.append(f"- Median {label}: {df[col].median():.1f}")
                
                # Top posts by this metric
                data_details.append(f"\nTop Posts by {label.title()}:")
                top_posts = df.nlargest(3, col)
                for i, row in top_posts.iterrows():
                    title = row.get('Title', 'Untitled post')
                    value = row.get(col, 0)
                    data_details.append(f"- {title} ({value:.1f} {label})")
        
        # Rest of your chat function...
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_message = f"Error processing question: {str(e)}"
        
        # Return error in the correct format
        user_message = {"role": "user", "content": question}
        assistant_message = {"role": "assistant", "content": error_message}
        
        if chat_history is None:
            return [user_message, assistant_message]
        else:
            return chat_history + [user_message, assistant_message]

def analyze_characteristics(df, column_name, cutoff=75):
    try:
        # Make sure the column exists
        if column_name not in df.columns:
            return f"Column '{column_name}' not found"
        
        # Create a copy to avoid modifying the original
        analysis_df = df.copy()
        
        # Ensure column is numeric
        analysis_df[column_name] = pd.to_numeric(analysis_df[column_name], errors='coerce')
        
        # Calculate percentiles first
        percentile_col = f"{column_name}_Percentile"
        if percentile_col not in analysis_df.columns:
            # Add the percentile column if it doesn't exist
            analysis_df[percentile_col] = analysis_df[column_name].rank(pct=True) * 100
        
        # Identify high-performing posts
        high_performing = analysis_df[analysis_df[percentile_col] > cutoff]
        
        if len(high_performing) == 0:
            return f"No posts above {cutoff}th percentile for {column_name}"
        
        # Generate insights
        insights = []
        
        # Analyze topics
        if 'Topic' in analysis_df.columns:
            top_topics = high_performing['Topic'].value_counts().head(3)
            insights.append(f"Top topics: {', '.join([f'{topic} ({count})' for topic, count in top_topics.items()])}")
        
        # Analyze tones
        if 'Writing Tone' in analysis_df.columns:
            top_tones = high_performing['Writing Tone'].value_counts().head(3)
            insights.append(f"Top tones: {', '.join([f'{tone} ({count})' for tone, count in top_tones.items()])}")
        
        # Analyze structure
        if 'Formatting' in analysis_df.columns:
            structure_counts = high_performing['Formatting'].str.split('|', expand=True)[0].str.strip().value_counts()
            top_structures = structure_counts.head(3)
            insights.append(f"Top structures: {', '.join([f'{struct} ({count})' for struct, count in top_structures.items()])}")
        
        # Return formatted insights
        return "\n".join(insights)
        
    except Exception as e:
        return f"Error in analyze_characteristics for {column_name}: {str(e)}"

# Add these imports to the top of your main file
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
import textwrap

def setup_langchain_agent(df):
    """Set up a LangChain agent that can perform calculations and analysis on the DataFrame"""
    try:
        # Create a Pandas DataFrame agent
        llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.2
        )
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True,
            allow_dangerous_code=True,  # Add this line to explicitly allow code execution
            handle_parsing_errors=True,
            include_df_in_prompt=True,
            prefix="""You are an expert data analyst specializing in LinkedIn post analytics.
            You have access to a DataFrame with LinkedIn post data and metrics.
            You can perform calculations, statistical analysis, and answer questions about the data.
            Analyze the data carefully and provide detailed insights backed by specific metrics.
            
            The DataFrame includes the following columns:
            - Full Post: The complete text of the LinkedIn post
            - Reactions: Number of reactions (likes etc.) received
            - Comment: Number of comments received
            - Reposts: Number of reposts/shares
            - Word Count: Number of words in the post
            - Flesch Reading Ease: Readability score
            - Topic: The main topic category of the post
            - Writing Tone: The tone used in the post
            - Intent: The purpose of the post (Inform, Convince, etc.)
            - Formatting: Structure and formatting characteristics
            """
        )
        return agent
    except Exception as e:
        print(f"Error setting up LangChain agent: {str(e)}")
        return None

def chat_with_data_enhanced(question, df, chat_history=None, summary=None):
    try:
        if df is None:
            return [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "Please analyze a file first."}
            ]
            
        # Create a copy to work with
        df = df.copy()
        print(f"Columns available for analysis: {df.columns.tolist()}")
        
        # Ensure key columns are numeric
        for col in ['Reactions', 'Comment', 'Reposts', 'Word Count', 'Flesch Reading Ease', 'Engagement_rate']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
                print(f"Converted {col} to numeric. Sample: {df[col].head(3).tolist()}")
        
        # For debugging - explicitly check if Engagement_rate is in the dataframe
        if 'Engagement_rate' in df.columns:
            print(f"Engagement rate column is present. Mean value: {df['Engagement_rate'].mean()}")
        else:
            print("Warning: Engagement_rate column is not present in the dataframe")
        
        # Create data context
        data_details = []
        data_details.append(f"## LinkedIn Post Analysis Data\n")
        data_details.append(f"General Statistics:")
        data_details.append(f"- Total posts analyzed: {len(df)}")
        
        # Add engagement metrics
        metrics = {
            'Reactions': 'reactions',
            'Comment': 'comments', 
            'Reposts': 'reposts',
            'Engagement_rate': 'engagement rate'
        }
        
        for col, label in metrics.items():
            if col in df.columns:
                data_details.append(f"\n{label.title()} Metrics:")
                data_details.append(f"- Highest {label}: {df[col].max():.1f}")
                data_details.append(f"- Average {label}: {df[col].mean():.1f}")
                data_details.append(f"- Median {label}: {df[col].median():.1f}")
                
                # Top posts by this metric
                data_details.append(f"\nTop Posts by {label.title()}:")
                top_posts = df.nlargest(3, col)
                for i, row in top_posts.iterrows():
                    title = row.get('Title', 'Untitled post')
                    value = row.get(col, 0)
                    data_details.append(f"- {title} ({value:.1f} {label})")
        
        # Specific analysis based on question
        specific_analysis = []
        
        # For questions about top performing posts
        if any(term in question.lower() for term in ["top", "best", "highest", "most"]):
            # If asking specifically about engagement rate
            if "engagement rate" in question.lower() or "Engagement_rate" in question.lower():
                if 'Engagement_rate' in df.columns:
                    top_posts = df.nlargest(3, 'Engagement_rate')
                    specific_analysis.append(f"\n## Posts with Highest Engagement Rate")
                    specific_analysis.append(f"Average Engagement Rate: {top_posts['Engagement_rate'].mean():.1f}")
                    specific_analysis.append(f"Average Reactions: {top_posts['Reactions'].mean():.1f}")
                    specific_analysis.append(f"Average Comments: {top_posts['Comment'].mean():.1f}")
                    
                    # Include topic and tone analysis for top engagement rate posts
                    if 'Topic' in top_posts.columns:
                        topics = top_posts['Topic'].value_counts()
                        specific_analysis.append(f"\nCommon Topics in High Engagement Posts:")
                        for topic, count in topics.items():
                            specific_analysis.append(f"- {topic}: {count} posts")
                            
                    if 'Writing Tone' in top_posts.columns:
                        tones = top_posts['Writing Tone'].value_counts()
                        specific_analysis.append(f"\nCommon Tones in High Engagement Posts:")
                        for tone, count in tones.items():
                            specific_analysis.append(f"- {tone}: {count} posts")
                else:
                    specific_analysis.append(f"\nEngagement rate data is not available in this dataset.")
            
            # If asking about reactions but want to know engagement rate too
            elif "reactions" in question.lower() and "engagement" in question.lower():
                top_posts = df.nlargest(3, 'Reactions')
                specific_analysis.append(f"\n## Posts with Highest Reactions and their Engagement")
                specific_analysis.append(f"Average Reactions: {top_posts['Reactions'].mean():.1f}")
                specific_analysis.append(f"Average Comments: {top_posts['Comment'].mean():.1f}")
                
                if 'Engagement_rate' in df.columns:
                    specific_analysis.append(f"Average Engagement Rate: {top_posts['Engagement_rate'].mean():.1f}")
                else:
                    specific_analysis.append(f"Engagement rate data is not available for these posts.")
            
            # For general top posts analysis (add engagement rate if available)
            elif "posts" in question.lower():
                top_posts = df.nlargest(5, 'Reactions')
                specific_analysis.append(f"\n## Analysis of Top Performing Posts")
                specific_analysis.append(f"Average Reactions: {top_posts['Reactions'].mean():.1f}")
                specific_analysis.append(f"Average Comments: {top_posts['Comment'].mean():.1f}")
                specific_analysis.append(f"Average Word Count: {top_posts['Word Count'].mean():.1f} words")
                specific_analysis.append(f"Average Reading Ease: {top_posts['Flesch Reading Ease'].mean():.2f}")
                
                if 'Engagement_rate' in df.columns:
                    specific_analysis.append(f"Average Engagement Rate: {top_posts['Engagement_rate'].mean():.1f}")
                
                # Include common characteristics
                if 'Topic' in top_posts.columns:
                    topics = top_posts['Topic'].value_counts()
                    specific_analysis.append(f"\nCommon Topics in Top Posts:")
                    for topic, count in topics.items():
                        specific_analysis.append(f"- {topic}: {count} posts")
                        
                if 'Writing Tone' in top_posts.columns:
                    tones = top_posts['Writing Tone'].value_counts()
                    specific_analysis.append(f"\nCommon Tones in Top Posts:")
                    for tone, count in tones.items():
                        specific_analysis.append(f"- {tone}: {count} posts")
        
        # Handle questions about specific engagement rate ranges or thresholds
        if "engagement rate" in question.lower() and any(term in question.lower() for term in ["above", "over", "higher than", "greater than"]):
            if 'Engagement_rate' in df.columns:
                # Try to extract a threshold value from the question
                threshold = 0
                for word in question.lower().split():
                    try:
                        threshold = float(word)
                        break
                    except ValueError:
                        continue
                
                # If no threshold found, use the average as default
                if threshold == 0:
                    threshold = df['Engagement_rate'].mean()
                
                high_engagement_posts = df[df['Engagement_rate'] > threshold]
                specific_analysis.append(f"\n## Posts with Engagement Rate > {threshold:.1f}")
                specific_analysis.append(f"Number of Posts: {len(high_engagement_posts)}")
                specific_analysis.append(f"Average Engagement Rate: {high_engagement_posts['Engagement_rate'].mean():.1f}")
                specific_analysis.append(f"Average Reactions: {high_engagement_posts['Reactions'].mean():.1f}")
                specific_analysis.append(f"Average Comments: {high_engagement_posts['Comment'].mean():.1f}")
                
                # Include characteristics of high engagement posts
                if len(high_engagement_posts) > 0:
                    if 'Topic' in high_engagement_posts.columns:
                        topics = high_engagement_posts['Topic'].value_counts()
                        specific_analysis.append(f"\nCommon Topics in High Engagement Posts:")
                        for topic, count in topics.items():
                            specific_analysis.append(f"- {topic}: {count} posts")
                            
                    if 'Writing Tone' in high_engagement_posts.columns:
                        tones = high_engagement_posts['Writing Tone'].value_counts()
                        specific_analysis.append(f"\nCommon Tones in High Engagement Posts:")
                        for tone, count in tones.items():
                            specific_analysis.append(f"- {tone}: {count} posts")
            else:
                specific_analysis.append(f"\nEngagement rate data is not available in this dataset.")
        
        # Data context and specific analysis
        prompt_parts = []
        prompt_parts.append("You are a professional LinkedIn content analyst providing insights based on post data analysis.")
        prompt_parts.append("\n".join(data_details))
        
        if specific_analysis:
            prompt_parts.append("\n".join(specific_analysis))
            
        if summary:
            prompt_parts.append(f"\nOverall Content Analysis Summary:\n{summary[:1500]}...")
            
        # Add chat history context (last 3 exchanges)
        if chat_history and len(chat_history) > 0:
            history_text = []
            for i in range(max(0, len(chat_history)-3), len(chat_history)):
                if chat_history[i]["role"] == "user":
                    history_text.append(f"Previous question: {chat_history[i]['content']}")
                else:
                    history_text.append(f"Previous answer: {chat_history[i]['content'][:100]}...")
            prompt_parts.append("\n".join(history_text))
        
        # Add the enhanced question
        enhanced_question = f"""
Based on the LinkedIn post data, please answer this question:
{question}

If calculations or statistical analysis are needed, perform them on the DataFrame.
Provide specific numbers and data-backed insights in your answer.
"""
        prompt_parts.append(enhanced_question)
        
        # Build the final prompt
        prompt = "\n\n".join(prompt_parts)
        
        # Use a higher temperature for more varied responses
        chat = ChatOpenAI(temperature=0.3)
        response = chat.predict(prompt)
        
        # Append to history and return
        if chat_history is None:
            chat_history = []
            
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": response})
        
        return chat_history
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_message = f"Error processing question: {str(e)}"
        
        # Return error in the correct format
        user_message = {"role": "user", "content": question}
        assistant_message = {"role": "assistant", "content": error_message}
        
        if chat_history is None:
            return [user_message, assistant_message]
        else:
            return chat_history + [user_message, assistant_message]

def analyze_correlations(df, metric1, metric2):
    """Analyze the correlation between two metrics in the DataFrame"""
    try:
        # Ensure the columns exist
        if metric1 not in df.columns or metric2 not in df.columns:
            return f"One or both metrics not found: {metric1}, {metric2}"
        
        # Create a copy to avoid modifying the original
        analysis_df = df.copy()
        
        # Ensure columns are numeric
        for col in [metric1, metric2]:
            analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
        
        # Drop rows with NaN values
        analysis_df = analysis_df.dropna(subset=[metric1, metric2])
        
        # Calculate correlation
        correlation = analysis_df[metric1].corr(analysis_df[metric2])
        
        # Interpret the correlation
        interpretation = ""
        if abs(correlation) < 0.3:
            interpretation = "weak or no linear relationship"
        elif abs(correlation) < 0.7:
            interpretation = "moderate linear relationship"
        else:
            interpretation = "strong linear relationship"
        
        # Direction
        direction = "positive" if correlation > 0 else "negative"
        
        return f"The correlation between {metric1} and {metric2} is {correlation:.3f}, indicating a {direction} {interpretation}."
    
    except Exception as e:
        return f"Error analyzing correlation: {str(e)}"

def segment_analysis(df, column, segments=3):
    """Perform segmented analysis on a specific metric"""
    try:
        # Ensure the column exists
        if column not in df.columns:
            return f"Column not found: {column}"
        
        # Create a copy to avoid modifying the original
        analysis_df = df.copy()
        
        # Ensure column is numeric
        analysis_df[column] = pd.to_numeric(analysis_df[column], errors='coerce')
        
        # Create segments
        analysis_df['Segment'] = pd.qcut(analysis_df[column], segments, labels=False)
        
        # Group by segments and calculate metrics
        segment_analysis = []
        
        for i in range(segments):
            segment_df = analysis_df[analysis_df['Segment'] == i]
            
            # Skip if segment is empty
            if len(segment_df) == 0:
                continue
                
            segment_stats = {
                'Segment': f"Segment {i+1}",
                'Range': f"{segment_df[column].min():.1f} - {segment_df[column].max():.1f}",
                'Count': len(segment_df),
                'Avg Reactions': segment_df['Reactions'].mean() if 'Reactions' in df.columns else None,
                'Avg Comments': segment_df['Comment'].mean() if 'Comment' in df.columns else None,
                'Avg Reposts': segment_df['Reposts'].mean() if 'Reposts' in df.columns else None,
            }
            
            # Add topic analysis if available
            if 'Topic' in df.columns:
                top_topics = segment_df['Topic'].value_counts().head(3)
                segment_stats['Top Topics'] = ', '.join([f"{t} ({c})" for t, c in top_topics.items()])
            
            # Add tone analysis if available
            if 'Writing Tone' in df.columns:
                top_tones = segment_df['Writing Tone'].value_counts().head(3)
                segment_stats['Top Tones'] = ', '.join([f"{t} ({c})" for t, c in top_tones.items()])
                
            segment_analysis.append(segment_stats)
        
        # Format results
        results = f"Segmented Analysis of {column}:\n\n"
        
        for segment in segment_analysis:
            results += f"### {segment['Segment']} ({segment['Range']})\n"
            results += f"- Posts: {segment['Count']}\n"
            
            if segment['Avg Reactions'] is not None:
                results += f"- Avg Reactions: {segment['Avg Reactions']:.1f}\n"
                
            if segment['Avg Comments'] is not None:
                results += f"- Avg Comments: {segment['Avg Comments']:.1f}\n"
                
            if segment['Avg Reposts'] is not None:
                results += f"- Avg Reposts: {segment['Avg Reposts']:.1f}\n"
                
            if 'Top Topics' in segment:
                results += f"- Top Topics: {segment['Top Topics']}\n"
                
            if 'Top Tones' in segment:
                results += f"- Top Tones: {segment['Top Tones']}\n"
                
            results += "\n"
        
        return results
    
    except Exception as e:
        return f"Error performing segment analysis: {str(e)}"
    
    

def chat_with_data_safe(question, df, chat_history=None, summary=None):
    """Safe version of chat_with_data that uses predefined analysis tools"""
    try:
        # Initialize chat history if None
        if chat_history is None:
            chat_history = []
        # Ensure chat_history is a list, not a DataFrame
        elif isinstance(chat_history, pd.DataFrame):
            chat_history = chat_history.values.tolist()

        if df is None:
            return chat_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "Please analyze a file first."}
            ]
        
        # Debug what we received
        print(f"\nAccessing dataframe with {len(df)} rows and {len(df.columns)} columns")
        print(f"Available columns: {df.columns.tolist()}")
        
        # Create the agent
        agent_executor = setup_safe_analysis_agent(df)
        if agent_executor is None:
            return chat_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "Failed to initialize analytics system. Please try again."}
            ]
        
        # Prepare context to include with the question
        context = ""
        
        # Add summary if available
        if summary:
            summary_brief = summary[:500] + "..." if len(summary) > 500 else summary
            context += f"Summary of previous analysis:\n{summary_brief}\n\n"
            
        # Enhanced question with context if needed
        enhanced_question = f"{context}Based on the LinkedIn post data, {question}"
        
        # Format chat history for the agent
        formatted_history = []
        for msg in chat_history:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    formatted_history.append(("human", content))
                elif role == "assistant":
                    formatted_history.append(("ai", content))
        
        try:
            # Use the agent to answer the question
            response = agent_executor.invoke({"input": enhanced_question, "chat_history": formatted_history})
            answer = response.get("output", "I couldn't generate a response. Please try asking your question differently.")
            
            # Ensure the answer is a string
            if not isinstance(answer, str):
                answer = str(answer)
            
            # Format for Gradio
            new_messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            
            # Ensure chat_history is a list of properly formatted messages
            if not isinstance(chat_history, list):
                chat_history = []
            
            # Validate each message in chat_history
            validated_history = []
            for msg in chat_history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    validated_history.append(msg)
            
            # Return the combined history
            return validated_history + new_messages
            
        except Exception as agent_error:
            print(f"Agent error: {str(agent_error)}")
            return chat_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"I encountered an error while processing your question: {str(agent_error)}"}
            ]
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_message = f"Error processing question: {str(e)}"
        
        # Return error in the correct format
        if not isinstance(chat_history, list):
            chat_history = []
            
        return chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": error_message}
        ]

    
    # Create the Gradio interface
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(show_label=False, type='messages')
        msg = gr.Textbox(label="Ask questions about the analysis results")
        clear = gr.Button("Clear")
        
        msg.submit(chat_with_agent, [msg, chatbot], [chatbot])
        clear.click(lambda: [], None, chatbot, queue=False)
    
    return demo

def main():
    demo = gr.Blocks(
        title="LinkedIn Post Analyzer",
        css="footer {display: none !important;}"
    )
    
    with demo:
        gr.Markdown("# LinkedIn Post Analyzer")
        gr.Markdown("Upload your LinkedIn posts data (CSV or Excel) to analyze content performance and get AI-powered insights.")

        processed_data_state = gr.State()
        summary_state = gr.State()

        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Upload CSV or Excel", file_types=['.csv', '.xlsx', '.xls'])
                analyze_button = gr.Button("Analyze")
            with gr.Column():
                status = gr.Textbox(label="Status", value="Ready to analyze...")
                updated_file_download = gr.File(label="Download Updated Excel")
                summary_download = gr.File(label="Download Summary Markdown")

        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(show_label=False, type='messages')
                msg = gr.Textbox(label="Ask questions about the analysis results")
                
                # Add a checkbox for switching between safe and enhanced mode
                enhanced_mode = gr.Checkbox(label="Use Enhanced Analysis Mode", value=False)
                
                gr.Markdown("### Example Questions:")
                with gr.Row():
                    q1 = gr.Button("What are the characteristics of high-performing posts?")
                    q2 = gr.Button("Which content structures and formats work best?")
                    q3 = gr.Button("What will be your top 3 recommendations to improve my engagement?")
                    q4 = gr.Button("What is the breakdown of posts across topics for the top 10% performing posts?")
                
                clear = gr.Button("Clear Chat")

        # Example question handlers with proper argument handling
        def set_message(question):
            return question, None  # Return both the message and None for the chatbot

        q1.click(
            fn=lambda: set_message("What are the characteristics of high-performing posts?"),
            inputs=None,
            outputs=[msg, chatbot]
        )
        q2.click(
            fn=lambda: set_message("Which content structures and formats work best?"),
            inputs=None,
            outputs=[msg, chatbot]
        )
        q3.click(
            fn=lambda: set_message("What will be your top 3 recommendations to improve my engagement?"),
            inputs=None,
            outputs=[msg, chatbot]
        )
        q4.click(
            fn=lambda: set_message("What is the breakdown of posts across topics for the top 10% performing posts?"),
            inputs=None,
            outputs=[msg, chatbot]
        )

        # Clear chat history
        clear.click(lambda: (None, None), outputs=[msg, chatbot])

        # File analysis handler
        analyze_button.click(
            analyze_file,
            inputs=[file_input],
            outputs=[status, updated_file_download, summary_download, processed_data_state, summary_state]
        )

        # Message handler - modified to handle both msg and chatbot outputs
        def chat_handler(message, df, history, summary, use_enhanced_mode=False):
            # Only process if there's a message
            if message:
                if use_enhanced_mode:
                    chat_response = chat_with_data_enhanced(message, df, history, summary)
                else:
                    chat_response = chat_with_data_safe(message, df, history, summary)
                return "", chat_response  # Clear the message box and update chat history
            return message, history  # Keep existing state if no message

        msg.submit(
            chat_handler,
            inputs=[msg, processed_data_state, chatbot, summary_state, enhanced_mode],
            outputs=[msg, chatbot]
        )

    return demo

if __name__ == "__main__":
    demo = main()
    if os.getenv('SPACE_ID'):
        # We're running on HF Spaces
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            favicon_path="https://huggingface.co/front/assets/huggingface_logo-noborder.svg"
        )
    else:
        # We're running locally
        demo.launch() 