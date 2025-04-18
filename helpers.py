import pandas as pd
import textstat
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os
from datetime import datetime
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.prompts import MessagesPlaceholder

# Ensure NLTK resources are downloaded
nltk.download('vader_lexicon')

# Load environment variables
load_dotenv()

def safe_text_analysis(text):
    """Helper function to ensure we're working with text"""
    if pd.isna(text):  # Check for NaN/None
        return "No content"
    return str(text)  # Convert to string if it's a number

def calculate_reading_ease(text):
    try:
        return textstat.flesch_reading_ease(str(text))
    except:
        return 0

def word_count(text):
    try:
        return len(str(text).split())
    except:
        return 0

def writing_tone(text):
    text = safe_text_analysis(text)
    if text == "No content":
        return "No tone"
    try:
        # Initialize OpenAI client
        llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Create a prompt for tone analysis
        prompt = PromptTemplate(
            input_variables=["post"],
            template="""
            Analyze this LinkedIn post and determine its primary tone.
            Choose exactly one tone from this list:
            - Formal (professional, structured, traditional)
            - Neutral (balanced, objective, straightforward)
            - Informal (casual, conversational, relaxed)
            - Optimistic (positive, hopeful, upbeat)
            - Worried (concerned, cautious, uncertain)
            - Friendly (warm, approachable, personable)
            - Curious (inquisitive, questioning, exploring)
            - Assertive (confident, direct, strong)
            - Encouraging (supportive, motivational, inspiring)
            - Surprised (amazed, unexpected, intrigued)
            - Cooperative (collaborative, inclusive, team-oriented)

            Post: {post}
            
            Return only the tone name (single word), nothing else.
            """
        )
        
        # Create and run the chain
        chain = prompt | llm
        tone = chain.invoke({"post": text}).strip()
        
        return tone
    except Exception as e:
        return "Tone Analysis Failed"

def formatting_analysis(text):
    text = safe_text_analysis(text)
    if text == "No content":
        return "No formatting"
    try:
        # Check for bullet points
        if re.search(r'(\* |- |\d+\.)\s+', text):
            return 'Bullet points present'
        return 'No bullet points'
    except Exception as e:
        return "Formatting Analysis Failed"

def analyze_intent(text):
    text = safe_text_analysis(text)
    if text == "No content":
        return "No intent"
    try:
        # Initialize OpenAI client
        llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Create a prompt for intent analysis
        prompt = PromptTemplate(
            input_variables=["post"],
            template="""
            Analyze this LinkedIn post and determine its primary intent.
            Choose exactly one intent from this list:
            - Inform (share knowledge, updates, or news)
            - Convince (persuade, influence, or call to action)
            - Describe (explain details, features, or processes)
            - Tell a story (share personal experiences or narratives)

            Post: {post}
            
            Return only the intent category name (Inform, Convince, Describe, or Tell a story), nothing else.
            """
        )
        
        # Create and run the chain
        chain = prompt | llm
        intent = chain.invoke({"post": text}).strip()
        
        return intent
    except Exception as e:
        return "Intent Analysis Failed"

def determine_media_type(media_content):
    """Determine media type from either direct string or URL"""
    if pd.isna(media_content):
        return "None"
    
    media_str = str(media_content).lower()
    
    # Direct string check
    if media_str in ["image", "video", "carousel", "none"]:
        return media_str.capitalize()
    
    # URL analysis
    if "http" in media_str:
        if "image" in media_str:
            return "Image"
        elif any(vid_term in media_str for vid_term in ["vid", "mp4", "video"]):
            return "Video"
        elif "carousel" in media_str:
            return "Carousel"
    
    return "None"

def process_dataframe(df):
    df['Flesch Reading Ease'] = df['Full Post'].apply(calculate_reading_ease)
    df['Word Count'] = df['Full Post'].apply(word_count)
    df['Writing Tone'] = df['Full Post'].apply(writing_tone)
    df['Formatting'] = df['Full Post'].apply(formatting_analysis)
    df['Intent'] = df['Full Post'].apply(analyze_intent)
    
    # Add Media Creative analysis
    if 'Media' in df.columns:
        df['Media Creative'] = df['Media'].apply(determine_media_type)
    
    return df

def generate_summary(df):
    try:
        # Debug prints
        print("Starting generate_summary function")
        print("DataFrame columns:", df.columns.tolist())
        print("DataFrame head:", df.head())
        
        # Calculate overall statistics with safe handling of None values
        total_posts = len(df)
        avg_reactions = df['Reactions'].fillna(0).mean()
        avg_comments = df['Comment'].fillna(0).mean()
        avg_reposts = df['Reposts'].fillna(0).mean()
        avg_word_count = df['Word Count'].fillna(0).mean()
        avg_reading_ease = df['Flesch Reading Ease'].fillna(0).mean()
        
        # Add Media Creative analysis if available
        media_distribution = None
        media_engagement = {}
        if 'Media Creative' in df.columns:
            media_distribution = df['Media Creative'].value_counts()
            # Calculate engagement metrics for each media type
            for media_type in media_distribution.index:
                media_posts = df[df['Media Creative'] == media_type]
                media_engagement[media_type] = {
                    'count': len(media_posts),
                    'avg_reactions': media_posts['Reactions'].fillna(0).mean(),
                    'avg_comments': media_posts['Comment'].fillna(0).mean(),
                    'avg_reposts': media_posts['Reposts'].fillna(0).mean(),
                    'avg_engagement': media_posts['Engagement_rate'].fillna(0).mean() if 'Engagement_rate' in df.columns else 0
                }
            
        # Add Engagement_rate (with error handling if column doesn't exist)
        avg_engagement_rate = df['Engagement_rate'].fillna(0).mean() if 'Engagement_rate' in df.columns else 0
        
        # Get high performing posts (top 25%)
        high_performing = df[df['Reactions'] > df['Reactions'].quantile(0.75)].copy()
        low_performing = df[df['Reactions'] < df['Reactions'].quantile(0.25)].copy()
        
        # Initialize summary sections
        summary = f"""# LinkedIn Post Analysis Summary

## Overall Statistics
- Total Posts Analyzed: {total_posts}
- Average Reactions: {avg_reactions:.1f}
- Average Comments: {avg_comments:.1f}
- Average Reposts: {avg_reposts:.1f}"""

        if avg_engagement_rate > 0:
            summary += f"""
- Average Engagement Rate: {avg_engagement_rate:.1f}%"""

        summary += f"""
- Average Word Count: {avg_word_count:.1f} words
- Average Reading Ease Score: {avg_reading_ease:.1f}"""

        # Add Media Creative analysis section if available
        if media_distribution is not None:
            summary += "\n\n## Media Creative Analysis"
            summary += "\n### Distribution by Media Type"
            for media_type, count in media_distribution.items():
                summary += f"\n- {media_type}: {count} posts ({(count/total_posts*100):.1f}%)"
            
            summary += "\n\n### Performance by Media Type"
            for media_type, metrics in media_engagement.items():
                summary += f"\n\n{media_type} Posts ({metrics['count']} posts):"
                summary += f"\n- Average Reactions: {metrics['avg_reactions']:.1f}"
                summary += f"\n- Average Comments: {metrics['avg_comments']:.1f}"
                summary += f"\n- Average Reposts: {metrics['avg_reposts']:.1f}"
                if metrics['avg_engagement'] > 0:
                    summary += f"\n- Average Engagement Rate: {metrics['avg_engagement']:.1f}%"

        # Add engagement analysis
        summary += f"""

## Engagement Analysis
### High Engagement Thresholds
- 75th Percentile: {df['Reactions'].quantile(0.75):.0f} reactions
- 90th Percentile: {df['Reactions'].quantile(0.90):.0f} reactions"""

        if 'Engagement_rate' in df.columns:
            summary += f"""
- 75th Percentile Engagement Rate: {df['Engagement_rate'].quantile(0.75):.1f}%
- 90th Percentile Engagement Rate: {df['Engagement_rate'].quantile(0.90):.1f}%"""

        # Add optional topic analysis if available
        if 'Topic' in df.columns:
            top_topics = df['Topic'].fillna('Unknown').value_counts().head(3)
            summary += "\n\n## Content Analysis"
            summary += "\n### Most Common Topics"
            for topic, count in top_topics.items():
                summary += f"\n- {topic}: {count} posts"

            if len(high_performing) > 0:
                high_perf_topics = high_performing['Topic'].fillna('Unknown').value_counts().head(3)
                summary += "\n\n### Top Topics in High-Performing Posts"
                for topic, count in high_perf_topics.items():
                    summary += f"\n- {topic}: {count} posts"

        # Add optional tone analysis if available
        if 'Writing Tone' in df.columns:
            top_tones = df['Writing Tone'].fillna('Unknown').value_counts().head(3)
            summary += "\n\n### Writing Tone Distribution"
            for tone, count in top_tones.items():
                summary += f"\n- {tone}: {count} posts"

        # Add optional intent analysis if available
        if 'Intent' in df.columns:
            top_intents = df['Intent'].fillna('Unknown').value_counts().head(3)
            summary += "\n\n### Content Intent"
            for intent, count in top_intents.items():
                summary += f"\n- {intent}: {count} posts"

        # Add readability analysis
        summary += f"""

## Readability Analysis
- Posts with High Readability (Score > 60): {len(df[df['Flesch Reading Ease'] > 60])} posts
- Posts with Medium Readability (Score 40-60): {len(df[(df['Flesch Reading Ease'] >= 40) & (df['Flesch Reading Ease'] <= 60)])} posts
- Posts with Low Readability (Score < 40): {len(df[df['Flesch Reading Ease'] < 40])} posts

## Length Analysis
- Short Posts (<100 words): {len(df[df['Word Count'] < 100])} posts
- Medium Posts (100-300 words): {len(df[(df['Word Count'] >= 100) & (df['Word Count'] <= 300)])} posts
- Long Posts (>300 words): {len(df[df['Word Count'] > 300])} posts"""

        # Add recommendations based on available data
        summary += "\n\n## Recommendations"
        summary += f"\n1. Content Length: Aim for {int(high_performing['Word Count'].mean())} words (average of top performing posts)"
        summary += f"\n2. Readability: Target a Flesch reading ease score of {int(high_performing['Flesch Reading Ease'].mean())}"

        if 'Topic' in df.columns:
            high_perf_topics = high_performing['Topic'].fillna('Unknown').value_counts().head(3)
            summary += f"\n3. Most Effective Topics: Focus on {', '.join(high_perf_topics.index)}"

        if 'Writing Tone' in df.columns:
            high_perf_tones = high_performing['Writing Tone'].fillna('Unknown').value_counts().head(2)
            summary += f"\n4. Recommended Tone: Primarily use {', '.join(high_perf_tones.index)}"

        if 'Engagement_rate' in df.columns:
            summary += f"\n5. Engagement Rate Target: Aim for {high_performing['Engagement_rate'].mean():.1f}% (average of top performers)"

        return summary

    except Exception as e:
        print(f"Error in generate_summary: {str(e)}")  # Debug print
        return f"Error generating summary: {str(e)}"

def format_list(series):
    """Helper function to format value counts into markdown list"""
    return '\n'.join([f"- {index}: {value} posts" for index, value in series.items()])

def setup_langchain():
    # Get API key from environment variable
    llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = PromptTemplate(
        input_variables=["question", "data"],
        template="""
        Based on the following data: {data}
        
        Please answer this question: {question}
        
        Provide a clear and concise answer based only on the data provided.
        """
    )
    
    chain = prompt | llm
    
    return chain

def run_chain(chain, question, data):
    return chain.invoke({"question": question, "data": data})

def analyze_file(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name, encoding='latin-1')
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            df = pd.read_excel(file.name)
        else:
            return ("Unsupported file format. Please upload a CSV or Excel file.", None, None)

        posts = df['Full Post'].tolist()
        analysis_results = batch_analyze_posts(posts)
        
        if analysis_results:
            # Create a new DataFrame from the analysis results
            results_df = pd.DataFrame(analysis_results)
            
            # Combine with original DataFrame
            df = pd.concat([df, results_df], axis=1)
            
            df['Flesch Reading Ease'] = df['Full Post'].apply(calculate_reading_ease)
            df['Word Count'] = df['Full Post'].apply(word_count)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_excel_path = f"analyzed_posts_{timestamp}.xlsx"
            output_markdown_path = f"analysis_summary_{timestamp}.md"
            
            df.to_excel(output_excel_path, index=False)
            
            summary = generate_summary(df)
            with open(output_markdown_path, 'w') as f:
                f.write(summary)
            
            return ("Analysis complete!", output_excel_path, output_markdown_path)

    except Exception as e:
        return (f"An error occurred: {str(e)}", None, None)

def batch_analyze_posts(posts):
    try:
        llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = PromptTemplate(
            input_variables=["post"],
            template="""
            Analyze this LinkedIn post and provide five pieces of information in exactly this format:
            TOPIC: [Choose from: 
            - Career Development
            - Career Development: Communications
            - Career Development: Networking
            - Career Development: Branding
            - Career Development: Writing
            - Career Development: Leadership
            - Industry Insights: AI
            - Industry Insights: Industrial Automation
            - Industry Insights: Marketing
            - Company News
            - Professional Tips: Communications
            - Professional Tips: Networking
            - Professional Tips: Branding
            - Professional Tips: Writing
            - Professional Tips: Marketing
            - Professional Tips: Leadership
            - Professional Tips: Personal growth
            - Personal Achievement
            - Event/Conference
            - Product/Service
            - Thought Leadership
            - Education/Learning: AI Fundamentals
            - Education/Learning: Machine Learning Basics
            - Education/Learning: Deep Learning
            - Education/Learning: Natural Language Processing
            - Education/Learning: Computer Vision
            - Education/Learning: AI Ethics and Safety
            - Education/Learning: AI Tools and Platforms
            - Job Opportunity
            - Market Trends
            - Team/Culture]
            TONE: [Choose from: Formal, Neutral, Informal, Optimistic, Worried, Friendly, Curious, Assertive, Encouraging, Surprised, Cooperative]
            INTENT: [Choose from: Inform, Convince, Describe, Tell a story]
            STRUCTURE: [Choose from: AIDA (Attention-Interest-Desire-Action), PAS (Problem-Agitation-Solution), Inverted Pyramid (most important first), No Clear Structure]
            BULLETS: [Choose from: Has Bullets, No Bullets]

            Post: {post}
            
            Return ONLY these five lines with your choices, nothing else.
            """
        )
        
        results = []
        for post in posts:
            try:
                # Single API call per post to get all 5 analyses
                chain = prompt | llm
                response = chain.invoke({"post": post}).strip()
                
                lines = response.split('\n')
                topic = lines[0].replace('TOPIC:', '').strip()
                tone = lines[1].replace('TONE:', '').strip()
                intent = lines[2].replace('INTENT:', '').strip()
                structure = lines[3].replace('STRUCTURE:', '').strip()
                bullets = lines[4].replace('BULLETS:', '').strip()
                
                results.append({
                    'Topic': topic,
                    'Writing Tone': tone,
                    'Intent': intent,
                    'Formatting': f"{structure} | {bullets}"
                })
            except Exception as e:
                results.append({
                    'Topic': 'Analysis Failed',
                    'Writing Tone': 'Analysis Failed',
                    'Intent': 'Analysis Failed',
                    'Formatting': 'Analysis Failed'
                })
                
        return results
    except Exception as e:
        return None

def setup_langchain_agent(df):
    """Set up a LangChain agent that can perform calculations and analysis on the DataFrame"""
    try:
        # Create a Pandas DataFrame agent
        llm = ChatOpenAI(
            model_name="gpt-4",  # Changed from gpt-4o to gpt-4
            temperature=0.2
        )
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True,
            allow_dangerous_code=True,
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

def setup_safe_analysis_agent(df):
    """Set up a safe LangChain agent that can perform calculations and analysis on the DataFrame"""
    try:
        # Create a Pandas DataFrame agent with safety constraints - with extra error handling
        try:
            # First try with basic parameters only
            llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.2
            )
        except Exception as llm_error:
            print(f"Error initializing ChatOpenAI: {str(llm_error)}")
            # Fall back to even simpler initialization
            try:
                llm = ChatOpenAI()
                print("Using default ChatOpenAI configuration")
            except Exception as fallback_error:
                print(f"Critical error with ChatOpenAI: {str(fallback_error)}")
                return None
        
        # Define safe analysis tools
        tools = [
            Tool(
                name="analyze_column_stats",
                func=lambda col: df[col].describe().to_dict() if col in df.columns else {"error": "Column not found"},
                description="Get statistical summary of a column. Input should be the column name."
            ),
            Tool(
                name="get_high_performing",
                func=lambda *args: df[df['Reactions'] > df['Reactions'].quantile(0.75)].to_dict('records'),
                description="Get high performing posts (top 25%). No input needed."
            ),
            Tool(
                name="get_correlation",
                func=lambda col1, col2: float(df[col1].corr(df[col2])) if col1 in df.columns and col2 in df.columns else None,
                description="Calculate correlation between two columns. Input should be two column names."
            )
        ]
        
        # Create the agent with the safe tools - added specific handling for different environments
        try:
            # Create a simpler prompt that works in all environments
            system_message = """You are an expert data analyst specializing in LinkedIn post analytics.
            You have access to a DataFrame with LinkedIn post data and metrics and can answer questions about it."""
            
            # Try using more basic prompt structure for compatibility
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", "{input}")
            ])
            
            # This simpler agent creation should work in more environments
            agent = create_openai_tools_agent(llm, tools, prompt)
            return AgentExecutor(agent=agent, tools=tools, verbose=True)
        except Exception as e:
            print(f"Error creating agent with standard prompt: {str(e)}")
            # Try an even simpler approach if the first fails
            try:
                # Create a direct agent instead of using the tool-based approach
                from langchain.agents import AgentType, initialize_agent
                return initialize_agent(
                    tools=tools,
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True
                )
            except Exception as alt_error:
                print(f"Error creating alternative agent: {str(alt_error)}")
                return None
                
    except Exception as e:
        print(f"Error setting up safe analysis agent: {str(e)}")
        return None

def setup_enhanced_analysis_agent(df):
    """Set up an enhanced LangChain agent that can perform calculations and analysis on the DataFrame"""
    try:
        # Create a Pandas DataFrame agent - with extra error handling
        try:
            # First try with basic parameters
            llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.2
            )
        except Exception as llm_error:
            print(f"Error initializing ChatOpenAI: {str(llm_error)}")
            # Fall back to even simpler initialization
            try:
                llm = ChatOpenAI()
                print("Using default ChatOpenAI configuration")
            except Exception as fallback_error:
                print(f"Critical error with ChatOpenAI: {str(fallback_error)}")
                return None
        
        # Safely attempt to create the agent with version detection
        # Try the most minimalist approach first
        try:
            from langchain.agents import AgentType, initialize_agent
            from langchain_community.agent_toolkits import create_pandas_dataframe_agent
            
            # Try to use the version that's most compatible with various environments
            return create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True
            )
        except Exception as pandas_error:
            print(f"Error with community pandas agent: {str(pandas_error)}")
            
            # If that fails, try with the experimental version but with minimal parameters
            try:
                return create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True
                )
            except Exception as final_err:
                print(f"Final error creating pandas agent: {str(final_err)}")
                
                # Last resort - try to use initialize_agent with custom tools
                try:
                    from langchain.tools.python.tool import PythonAstREPLTool
                    
                    # Create a Python REPL tool with the dataframe
                    python_tool = PythonAstREPLTool(locals={"df": df})
                    
                    # Initialize an agent with this tool
                    return initialize_agent(
                        tools=[python_tool],
                        llm=llm,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True
                    )
                except Exception as e:
                    print(f"Could not create any agent: {str(e)}")
                    return None
                    
    except Exception as e:
        print(f"Error setting up enhanced analysis agent: {str(e)}")
        return None 