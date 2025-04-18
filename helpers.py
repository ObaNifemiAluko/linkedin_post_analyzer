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
        
        # Calculate overall statistics
        total_posts = len(df)
        avg_reactions = df['Reactions'].mean()
        avg_comments = df['Comment'].mean()
        avg_reposts = df['Reposts'].mean()
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
                    'avg_reactions': media_posts['Reactions'].mean(),
                    'avg_comments': media_posts['Comment'].mean(),
                    'avg_reposts': media_posts['Reposts'].mean(),
                    'avg_engagement': media_posts['Engagement_rate'].mean() if 'Engagement_rate' in df.columns else None
                }
            
        # Add Engagement_rate (with error handling if column doesn't exist)
        avg_Engagement_rate = df['Engagement_rate'].mean() if 'Engagement_rate' in df.columns else None
        
        # Get high performing posts (top 25%)
        high_performing = df[df['Reactions'] > df['Reactions'].quantile(0.75)].copy()
        low_performing = df[df['Reactions'] < df['Reactions'].quantile(0.25)].copy()
        
        # Get top performing content characteristics
        top_topics = df['Topic'].value_counts().head(3)
        top_tones = df['Writing Tone'].value_counts().head(3)
        top_intents = df['Intent'].value_counts().head(3)
        
        # Analyze formatting and structure
        formats = df['Formatting'].str.split('|', expand=True)
        formats.columns = ['Structure', 'Bullets']
        structure_counts = formats['Structure'].value_counts()
        bullet_usage = formats['Bullets'].value_counts()
        
        # Calculate engagement percentiles
        percentile_75 = df['Reactions'].quantile(0.75)
        percentile_90 = df['Reactions'].quantile(0.90)
        
        # Add Engagement_rate percentiles if column exists
        engagement_percentile_75 = df['Engagement_rate'].quantile(0.75) if 'Engagement_rate' in df.columns else None
        engagement_percentile_90 = df['Engagement_rate'].quantile(0.90) if 'Engagement_rate' in df.columns else None
        
        # Analyze high-performing content
        high_perf_structures = high_performing['Formatting'].str.split('|', expand=True)[0].value_counts()
        high_perf_bullets = high_performing['Formatting'].str.split('|', expand=True)[1].value_counts()
        
        # Generate summary markdown
        summary = f"""# LinkedIn Post Analysis Summary

## Overall Statistics
- Total Posts Analyzed: {total_posts}
- Average Reactions: {avg_reactions:.1f}
- Average Comments: {avg_comments:.1f}
- Average Reposts: {avg_reposts:.1f}
- Average Engagement Rate: {avg_Engagement_rate:.1f}
- Average Word Count: {avg_word_count:.1f} words
- Average Reading Ease Score: {avg_reading_ease:.1f}"""

        # Add Media Creative analysis section
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
                if metrics['avg_engagement'] is not None:
                    summary += f"\n- Average Engagement Rate: {metrics['avg_engagement']:.1f}%"
            
            # Add media type analysis for high-performing posts
            if len(high_performing) > 0 and 'Media Creative' in high_performing.columns:
                high_perf_media = high_performing['Media Creative'].value_counts()
                summary += "\n\n### Media Types in High-Performing Posts"
                for media_type, count in high_perf_media.items():
                    summary += f"\n- {media_type}: {count} posts ({(count/len(high_performing)*100):.1f}%)"
                    media_posts = high_performing[high_performing['Media Creative'] == media_type]
                    summary += f"\n  - Average Reactions: {media_posts['Reactions'].mean():.1f}"
                    if 'Engagement_rate' in media_posts.columns:
                        summary += f"\n  - Average Engagement Rate: {media_posts['Engagement_rate'].mean():.1f}%"

        # Add Engagement_rate to Overall Statistics if it exists
        if avg_Engagement_rate is not None:
            summary += f"""
- Average Engagement Rate: {avg_Engagement_rate:.1f}"""

        summary += f"""
- Average Word Count: {avg_word_count:.1f} words
- Average Reading Ease Score: {avg_reading_ease:.1f}

## Engagement Analysis
### High Engagement Thresholds
- 75th Percentile: {percentile_75:.0f} reactions
- 90th Percentile: {percentile_90:.0f} reactions"""

        # Add Engagement_rate percentiles if data exists
        if engagement_percentile_75 is not None and engagement_percentile_90 is not None:
            summary += f"""
- 75th Percentile Engagement Rate: {engagement_percentile_75:.1f}
- 90th Percentile Engagement Rate: {engagement_percentile_90:.1f}"""

        summary += f"""

### Posts with High Engagement
- High Reactions (>75th percentile): {len(df[df['Reactions'] > percentile_75])} posts
- High Comments (>75th percentile): {len(df[df['Comment'] > df['Comment'].quantile(0.75)])} posts
- High Reposts (>75th percentile): {len(df[df['Reposts'] > df['Reposts'].quantile(0.75)])} posts"""

        # Add high Engagement_rate counts if column exists
        if 'Engagement_rate' in df.columns:
            summary += f"""
- High Engagement Rate (>75th percentile): {len(df[df['Engagement_rate'] > engagement_percentile_75])} posts"""

        summary += f"""

## Content Analysis
### Most Common Topics
{format_list(top_topics)}

### Top Topics in High-Performing Posts
{format_list(high_performing['Topic'].value_counts().head(3))}

### Writing Tone Distribution
{format_list(top_tones)}

### Content Intent
{format_list(top_intents)}

## Structure Analysis
### Content Structure Distribution
{format_list(structure_counts)}

### Bullet Point Usage
{format_list(bullet_usage)}

### Most Effective Structures (High-Performing Posts)
{format_list(high_perf_structures)}

### Bullet Point Impact
{format_list(high_perf_bullets)}

## Readability Analysis
- Posts with High Readability (Score > 60): {len(df[df['Flesch Reading Ease'] > 60])} posts
- Posts with Medium Readability (Score 40-60): {len(df[(df['Flesch Reading Ease'] >= 40) & (df['Flesch Reading Ease'] <= 60)])} posts
- Posts with Low Readability (Score < 40): {len(df[df['Flesch Reading Ease'] < 40])} posts

## Length Analysis
- Short Posts (<100 words): {len(df[df['Word Count'] < 100])} posts
- Medium Posts (100-300 words): {len(df[(df['Word Count'] >= 100) & (df['Word Count'] <= 300)])} posts
- Long Posts (>300 words): {len(df[df['Word Count'] > 300])} posts

## Top vs Bottom Analysis
### Top Performing Posts (Top 25%)
- Average Word Count: {high_performing['Word Count'].mean():.1f}
- Average Reading Ease: {high_performing['Flesch Reading Ease'].mean():.1f}"""

        # Add Engagement_rate for top performers if column exists
        if 'Engagement_rate' in df.columns:
            summary += f"""
- Average Engagement Rate: {high_performing['Engagement_rate'].mean():.1f}"""

        summary += f"""
- Most Common Topics: {', '.join(high_performing['Topic'].value_counts().head(3).index)}
- Most Common Tones: {', '.join(high_performing['Writing Tone'].value_counts().head(3).index)}

### Bottom Performing Posts (Bottom 25%)
- Average Word Count: {low_performing['Word Count'].mean():.1f}
- Average Reading Ease: {low_performing['Flesch Reading Ease'].mean():.1f}"""

        # Add Engagement_rate for bottom performers if column exists
        if 'Engagement_rate' in df.columns:
            summary += f"""
- Average Engagement Rate: {low_performing['Engagement_rate'].mean():.1f}"""

        summary += f"""
- Most Common Topics: {', '.join(low_performing['Topic'].value_counts().head(3).index)}
- Most Common Tones: {', '.join(low_performing['Writing Tone'].value_counts().head(3).index)}

## Recommendations
1. Content Length: Aim for {int(high_performing['Word Count'].mean())} words (average of top performing posts)
2. Readability: Target a Flesch reading ease score of {int(high_performing['Flesch Reading Ease'].mean())}"""

        # Add Engagement_rate recommendation if column exists
        if 'Engagement_rate' in df.columns:
            summary += f"""
3. Engagement Rate: Target an engagement rate of {high_performing['Engagement_rate'].mean():.1f} (average of top performers)
4. Most Effective Topics: Focus on {', '.join(high_performing['Topic'].value_counts().head(3).index)}
5. Recommended Tone: Primarily use {', '.join(high_performing['Writing Tone'].value_counts().head(2).index)}
6. Content Structure: Consider {high_perf_structures.index[0]} format with {high_perf_bullets.index[0].lower()}
7. Engagement Target: Aim for {int(percentile_75)} reactions to be in the top 25% of posts
"""
        else:
            summary += f"""
3. Most Effective Topics: Focus on {', '.join(high_performing['Topic'].value_counts().head(3).index)}
4. Recommended Tone: Primarily use {', '.join(high_performing['Writing Tone'].value_counts().head(2).index)}
5. Content Structure: Consider {high_perf_structures.index[0]} format with {high_perf_bullets.index[0].lower()}
6. Engagement Target: Aim for {int(percentile_75)} reactions to be in the top 25% of posts
"""

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

def setup_safe_analysis_agent(df):
    """Set up a LangChain agent that can perform calculations and analysis on the DataFrame"""
    from langchain_core.tools import Tool
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    
    # Define the analysis function
    def analyze_posts(query):
        """Analyze LinkedIn posts based on the query"""
        try:
            # Debug print to verify DataFrame contents
            print(f"DataFrame columns: {df.columns.tolist()}")
            print(f"Number of rows: {len(df)}")
            
            # Initialize response
            response = ""
            
            # First check if we have a summary to work with
            summary = None
            try:
                summary = generate_summary(df)
            except Exception as e:
                print(f"Could not generate summary: {str(e)}")
            
            # For queries about media types
            if any(term in query.lower() for term in ["media", "image", "video", "creative"]):
                if 'Media Creative' not in df.columns:
                    return "Media Creative data is not available in the dataset."
                
                # Get media distribution
                media_dist = df['Media Creative'].value_counts()
                response = "Media Creative Analysis:\n\n"
                
                # Overall distribution
                response += "Distribution of Media Types:\n"
                for media_type, count in media_dist.items():
                    response += f"- {media_type}: {count} posts ({(count/len(df)*100):.1f}%)\n"
                
                # Engagement analysis by media type
                response += "\nEngagement Analysis by Media Type:\n"
                for media_type in media_dist.index:
                    media_posts = df[df['Media Creative'] == media_type]
                    avg_reactions = media_posts['Reactions'].mean()
                    avg_comments = media_posts['Comment'].mean()
                    avg_engagement = media_posts['Engagement_rate'].mean() if 'Engagement_rate' in df.columns else None
                    
                    response += f"\n{media_type} Posts:\n"
                    response += f"- Average Reactions: {avg_reactions:.1f}\n"
                    response += f"- Average Comments: {avg_comments:.1f}\n"
                    if avg_engagement is not None:
                        response += f"- Average Engagement Rate: {avg_engagement:.1f}%\n"
                
                return response
            
            # For basic statistics, try to extract from summary first
            if summary and ("average" in query.lower() or "mean" in query.lower()):
                # Only use summary for general average queries without specific subsets
                if not any(term in query.lower() for term in ["highest", "lowest", "top", "bottom", "worst", "best"]):
                    summary_lines = summary.split('\n')
                    if "word" in query.lower() or "count" in query.lower():
                        for line in summary_lines:
                            if "Average Word Count:" in line:
                                return line.strip().replace('- ', '')
                    elif "reaction" in query.lower():
                        for line in summary_lines:
                            if "Average Reactions:" in line:
                                return line.strip().replace('- ', '')
                    elif "comment" in query.lower():
                        for line in summary_lines:
                            if "Average Comments:" in line:
                                return line.strip().replace('- ', '')
                    elif "repost" in query.lower():
                        for line in summary_lines:
                            if "Average Reposts:" in line:
                                return line.strip().replace('- ', '')
                    elif "reading" in query.lower() or "flesch" in query.lower():
                        for line in summary_lines:
                            if "Average Reading Ease Score:" in line:
                                return line.strip().replace('- ', '')
                    elif "engagement" in query.lower():
                        for line in summary_lines:
                            if "Average Engagement Rate:" in line:
                                return line.strip().replace('- ', '')
            
            # For queries about word count of specific subsets
            if "word" in query.lower() or "count" in query.lower():
                # Try to extract the number of posts to analyze
                num_posts = 3  # default for specific subsets
                for word in query.lower().split():
                    try:
                        num = int(word)
                        num_posts = num
                        break
                    except ValueError:
                        continue
                
                # For lowest performing posts
                if any(term in query.lower() for term in ["lowest", "worst", "bottom"]):
                    bottom_posts = df.nsmallest(num_posts, 'Reactions')
                    avg_words = bottom_posts['Word Count'].mean()
                    response = f"Analysis of word count for {num_posts} posts with lowest reactions:\n"
                    response += f"- Average word count: {avg_words:.1f} words\n"
                    response += f"- Individual post details:\n"
                    
                    for _, post in bottom_posts.iterrows():
                        response += f"  - Post with {post['Reactions']} reactions has {post['Word Count']} words"
                        if 'Topic' in df.columns:
                            response += f" (Topic: {post['Topic']})"
                        response += "\n"
                    
                    return response
                
                # For highest performing posts
                elif any(term in query.lower() for term in ["highest", "top", "best"]):
                    top_posts = df.nlargest(num_posts, 'Reactions')
                    avg_words = top_posts['Word Count'].mean()
                    response = f"Analysis of word count for {num_posts} posts with highest reactions:\n"
                    response += f"- Average word count: {avg_words:.1f} words\n"
                    response += f"- Individual post details:\n"
                    
                    for _, post in top_posts.iterrows():
                        response += f"  - Post with {post['Reactions']} reactions has {post['Word Count']} words"
                        if 'Topic' in df.columns:
                            response += f" (Topic: {post['Topic']})"
                        response += "\n"
                    
                    return response
                
                # For general word count query
                else:
                    if "Word Count" not in df.columns:
                        return "Word count data is not available in the dataset."
                    return f"The average word count across all posts is {df['Word Count'].mean():.1f} words."
            
            # For queries about reading scores
            elif "flesch" in query.lower() or "reading" in query.lower() or "readability" in query.lower():
                if 'Flesch Reading Ease' not in df.columns:
                    return "Flesch Reading Ease scores are not available in the dataset."
                
                # Convert to numeric and handle any non-numeric values
                df['Flesch Reading Ease'] = pd.to_numeric(df['Flesch Reading Ease'], errors='coerce')
                valid_scores = df.dropna(subset=['Flesch Reading Ease'])
                
                if len(valid_scores) == 0:
                    return "No valid Flesch Reading Ease scores found in the dataset."
                
                response = f"Analysis of Flesch Reading Ease scores across {len(valid_scores)} posts:\n\n"
                response += f"Overall metrics:\n"
                response += f"- Average Reading Ease Score: {valid_scores['Flesch Reading Ease'].mean():.1f}\n"
                response += f"- Highest Score: {valid_scores['Flesch Reading Ease'].max():.1f}\n"
                response += f"- Lowest Score: {valid_scores['Flesch Reading Ease'].min():.1f}\n"
                response += f"- Median Score: {valid_scores['Flesch Reading Ease'].median():.1f}\n\n"
                
                # Add readability distribution
                response += "Readability Distribution:\n"
                response += f"- Very Easy (90-100): {len(valid_scores[valid_scores['Flesch Reading Ease'] >= 90])} posts\n"
                response += f"- Easy (80-89): {len(valid_scores[(valid_scores['Flesch Reading Ease'] >= 80) & (valid_scores['Flesch Reading Ease'] < 90)])} posts\n"
                response += f"- Fairly Easy (70-79): {len(valid_scores[(valid_scores['Flesch Reading Ease'] >= 70) & (valid_scores['Flesch Reading Ease'] < 80)])} posts\n"
                response += f"- Standard (60-69): {len(valid_scores[(valid_scores['Flesch Reading Ease'] >= 60) & (valid_scores['Flesch Reading Ease'] < 70)])} posts\n"
                response += f"- Fairly Difficult (50-59): {len(valid_scores[(valid_scores['Flesch Reading Ease'] >= 50) & (valid_scores['Flesch Reading Ease'] < 60)])} posts\n"
                response += f"- Difficult (30-49): {len(valid_scores[(valid_scores['Flesch Reading Ease'] >= 30) & (valid_scores['Flesch Reading Ease'] < 50)])} posts\n"
                response += f"- Very Difficult (0-29): {len(valid_scores[valid_scores['Flesch Reading Ease'] < 30])} posts\n\n"
                
                # Analyze correlation with engagement
                if 'Reactions' in df.columns:
                    valid_data = valid_scores.dropna(subset=['Reactions'])
                    if len(valid_data) > 0:
                        corr_reactions = valid_data['Flesch Reading Ease'].corr(valid_data['Reactions'])
                        response += f"\nCorrelation with engagement:\n"
                        response += f"- Correlation with Reactions: {corr_reactions:.3f}\n"
                
                # Top performing posts and their readability
                if 'Reactions' in df.columns:
                    top_posts = valid_scores.nlargest(5, 'Reactions')
                    response += f"\nReadability of top 5 performing posts:\n"
                    for _, post in top_posts.iterrows():
                        response += f"- Post with {post['Reactions']} reactions has reading ease score of {post['Flesch Reading Ease']:.1f}\n"
                
                return response
            
            # For general analysis or high-performing posts
            elif any(term in query.lower() for term in ["high", "best", "top", "perform"]):
                response = f"Analysis based on all {len(df)} posts:\n\n"
                response += f"Overall metrics:\n"
                response += f"- Average Reactions: {df['Reactions'].mean():.1f}\n"
                response += f"- Average Comments: {df['Comment'].mean():.1f}\n"
                response += f"- Average Reposts: {df['Reposts'].mean():.1f}\n"
                
                if 'Engagement_rate' in df.columns:
                    response += f"- Average Engagement Rate: {df['Engagement_rate'].mean():.1f}%\n"
                
                # Top performing posts analysis
                top_posts = df.nlargest(5, 'Reactions')
                response += f"\nTop 5 Performing Posts:\n"
                for _, post in top_posts.iterrows():
                    response += f"- Post with {post['Reactions']} reactions"
                    if 'Topic' in df.columns:
                        response += f", Topic: {post['Topic']}"
                    if 'Writing Tone' in df.columns:
                        response += f", Tone: {post['Writing Tone']}"
                    response += "\n"
                
                # Add topic distribution if available
                if 'Topic' in df.columns:
                    response += "\nTopic Distribution in Top Posts:\n"
                    topic_dist = top_posts['Topic'].value_counts()
                    for topic, count in topic_dist.items():
                        response += f"- {topic}: {count} posts\n"
                
                return response
            
            # For queries about engagement rate
            elif "engagement" in query.lower() or "engagement_rate" in query.lower():
                if 'Engagement_rate' not in df.columns:
                    return "Engagement rate data is not available in the dataset."
                
                response = f"Engagement Rate Analysis:\n"
                response += f"- Average Engagement Rate: {df['Engagement_rate'].mean():.1f}%\n"
                response += f"- Highest Engagement Rate: {df['Engagement_rate'].max():.1f}%\n"
                response += f"- Lowest Engagement Rate: {df['Engagement_rate'].min():.1f}%\n"
                
                # Top engaging posts
                top_engaging = df.nlargest(3, 'Engagement_rate')
                response += f"\nTop 3 Posts by Engagement Rate:\n"
                for _, post in top_engaging.iterrows():
                    response += f"- {post['Engagement_rate']:.1f}% engagement rate"
                    if 'Topic' in df.columns:
                        response += f", Topic: {post['Topic']}"
                    response += "\n"
                
                return response
            
            # Default analysis
            else:
                response = f"Analysis based on all {len(df)} posts:\n\n"
                response += f"Overall metrics:\n"
                response += f"- Total Posts: {len(df)}\n"
                response += f"- Average Reactions: {df['Reactions'].mean():.1f}\n"
                response += f"- Average Comments: {df['Comment'].mean():.1f}\n"
                response += f"- Average Reposts: {df['Reposts'].mean():.1f}\n"
                
                if 'Engagement_rate' in df.columns:
                    response += f"- Average Engagement Rate: {df['Engagement_rate'].mean():.1f}%\n"
                
                return response

        except Exception as e:
            print(f"Error analyzing posts: {str(e)}")
            return f"Error analyzing posts: {str(e)}"
    
    # Create a tool from the function
    tools = [
        Tool(
            name="analyze_posts",
            func=analyze_posts,
            description="Analyze LinkedIn posts based on user queries"
        )
    ]
    
    # Create the prompt template with agent_scratchpad included
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You're an assistant that analyzes LinkedIn posts data. Use the tools available to answer questions.
        Important guidelines:
        1. Always calculate metrics directly from the data, don't refer to previous analysis
        2. Provide specific numbers and statistics when available
        3. If a calculation is needed, perform it using the current data
        4. Don't mention previous conversations or analyses
        5. Be direct and concise in your responses"""),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])
    
    # Create the model and agent
    llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Set temperature to 0 for consistent responses
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor 