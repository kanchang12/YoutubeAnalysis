from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway
from openai import OpenAI
import json
import pandas as pd
from flask_cors import CORS
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
# Set up OpenAI API
import os

# Set up OpenAI API
api_key = os.getenv('OPENAI_API_KEY')  # Get the API key from the environment variable
openai_client = OpenAI(api_key=api_key)

# Chat history to maintain context
chat_history = []

# Looker Studio iframe URLs
LOOKER_STUDIO_IFRAMES = {
    'basic_stats': 'https://public.tableau.com/views/YoutubeAnalyis/Sheet1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link',
    'category_wise': 'https://lookerstudio.google.com/embed/reporting/61bd5267-ba4b-408c-b873-ec342207fc0e/page/your_page_id',
    'keywords_research': 'https://lookerstudio.google.com/embed/reporting/61bd5267-ba4b-408c-b873-ec342207fc0e/page/your_page_id',
    'channel_growth': 'https://lookerstudio.google.com/embed/reporting/61bd5267-ba4b-408c-b873-ec342207fc0e/page/your_page_id'
}

import pandas as pd
import requests
import csv
from io import StringIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(filepath):
    try:
        logger.info(f"Starting to load data from {filepath}")
        
        # Download content if it's a URL
        if filepath.startswith('http'):
            logger.info("Downloading data from URL")
            response = requests.get(filepath)
            content = StringIO(response.text)
        else:
            logger.info("Reading local file")
            content = open(filepath, 'r', encoding='utf-8')

        # First, detect the dialect and delimiter
        sample = content.read(1024)
        content.seek(0)  # Reset file pointer
        
        try:
            dialect = csv.Sniffer().sniff(sample)
            logger.info(f"Detected delimiter: {dialect.delimiter}")
        except:
            logger.warning("Could not detect CSV dialect, using comma as delimiter")
            dialect = csv.excel

        # Read the headers first
        reader = csv.reader(content, dialect)
        headers = next(reader)
        logger.info(f"Headers found: {headers}")

        # Initialize list for valid rows
        valid_rows = []
        row_count = 0
        error_count = 0

        # Process each row
        for row_num, row in enumerate(reader, start=2):  # start=2 because we already read headers
            try:
                # Skip empty rows
                if not row or all(cell.strip() == '' for cell in row):
                    continue

                # If row has correct number of fields, process it
                if len(row) == len(headers):
                    # Clean the row data
                    cleaned_row = [cell.strip() for cell in row]
                    valid_rows.append(cleaned_row)
                    row_count += 1
                else:
                    logger.warning(f"Line {row_num}: Found {len(row)} fields, expected {len(headers)}")
                    error_count += 1

            except Exception as e:
                logger.error(f"Error processing row {row_num}: {str(e)}")
                error_count += 1
                continue

        logger.info(f"Processed {row_count} valid rows, encountered {error_count} errors")

        # Create DataFrame from valid rows
        df = pd.DataFrame(valid_rows, columns=headers)

        # Convert numeric columns
        numeric_cols = ['view_count', 'like_count', 'comment_count']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
                logger.info(f"Converted {col} to numeric, found {df[col].isna().sum()} missing values")

        # Handle text columns
        text_cols = ['title', 'description']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')

        # Close the file if it's not a StringIO object
        if not isinstance(content, StringIO):
            content.close()

        if df.empty:
            logger.warning("No valid data was loaded")
            return None

        logger.info(f"Successfully loaded {len(df)} rows of data")
        return df

    except Exception as e:
        logger.error(f"Fatal error in load_data: {str(e)}", exc_info=True)
        return None
    finally:
        # Ensure file is closed
        if 'content' in locals() and not isinstance(content, StringIO):
            content.close()


# Hypothesis testing
def dynamic_hypothesis_testing(df, group_condition, metric='relevance_score', min_sample_size=5):
    try:
        if callable(group_condition):
            group1 = df[group_condition(df)][metric]
        elif isinstance(group_condition, dict):
            condition = ' & '.join([f"{k} == '{v}'" if isinstance(v, str)
                                    else f"{k} == {v}" for k, v in group_condition.items()])
            group1 = df.query(condition)[metric]
        else:
            group1 = df[df.eval(group_condition)][metric]
        if len(group1) < min_sample_size:
            return {'error': f'Insufficient data for group 1. Found {len(group1)} samples, need at least {min_sample_size}.'}
        group2 = df[~df.index.isin(group1.index)][metric]
        if len(group2) < min_sample_size:
            group2 = df.sample(n=max(min_sample_size, len(group1)), random_state=42)[metric]
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
        f_stat, anova_p_value = f_oneway(group1, group2)
        pooled_std = np.sqrt(((len(group1) - 1) * group1.std()**2 +
                              (len(group2) - 1) * group2.std()**2) /
                             (len(group1) + len(group2) - 2))
        cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std != 0 else 0
        return {
            'status': 'success',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'f_statistic': float(f_stat),
            'anova_p_value': float(anova_p_value),
            'cohens_d': float(cohens_d),
            'group1_stats': {'mean': float(group1.mean()), 'std': float(group1.std()), 'size': int(len(group1))},
            'group2_stats': {'mean': float(group2.mean()), 'std': float(group2.std()), 'size': int(len(group2))},
            'significance_level': 0.05,
            'is_significant': float(p_value) < 0.05
        }
    except Exception as e:
        return {'error': str(e), 'status': 'failed'}

# Interpret hypothesis results
def interpret_results(results, hypothesis):
    if results['status'] == 'failed':
        return f"Analysis failed: {results.get('error', 'Unknown error')}"
    interpretation = f"Testing hypothesis: {hypothesis}\n\n"
    if results['is_significant']:
        interpretation += f"There is statistical evidence to support the hypothesis (p-value: {results['p_value']:.4f}). "
    else:
        interpretation += f"There is not enough statistical evidence to support the hypothesis (p-value: {results['p_value']:.4f}). "
    effect_size = abs(results['cohens_d'])
    if effect_size < 0.2:
        effect_description = "negligible"
    elif effect_size < 0.5:
        effect_description = "small"
    elif effect_size < 0.8:
        effect_description = "medium"
    else:
        effect_description = "large"
    interpretation += f"\nThe effect size is {effect_description} (Cohen's d: {results['cohens_d']:.2f})."
    return interpretation

# Get top videos
def get_top_videos(df, n=5):
    try:
        df['engagement_score'] = (df['view_count'] * 0.3 + 
                                   df['like_count'] * 0.3 + 
                                   df['comment_count'] * 0.2 +
                                   df['relevance_score'] * 0.2)
        if len(df) > 0:
            df['engagement_score'] = (df['engagement_score'] - df['engagement_score'].min()) / \
                                     (df['engagement_score'].max() - df['engagement_score'].min() + 1e-10)
        top_videos = df.sort_values('engagement_score', ascending=False).head(n)
        return top_videos[['title', 'view_count', 'like_count', 'comment_count', 
                            'relevance_score', 'engagement_score']].to_dict('records')
    except Exception as e:
        print(f"Error getting top videos: {e}")
        return []

# Generate insights using OpenAI
def generate_insights(data):
    try:
        prompt = (
            "Hypothesis Testing Results:\n"
            + (f"- T-statistic: {data['t_statistic']}\n" if 't_statistic' in data else '')
            + (f"- P-value: {data['p_value']}\n" if 'p_value' in data else '')
            + (f"- F-statistic: {data['f_statistic']}\n" if 'f_statistic' in data else '')
            + (f"- ANOVA P-value: {data['anova_p_value']}\n" if 'anova_p_value' in data else '')
            + (f"- Group 1 Mean (Similar Videos): {data['group1_mean']}\n" if 'group1_mean' in data else '')
            + (f"- Group 2 Mean (Random Videos): {data['group2_mean']}" if 'group2_mean' in data else '')
        )

        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst. Your job is to find answers and hidden patterns in the data"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating insights: {e}")
        return "Unable to generate insights at this time. Please try again later."

# Get Looker Studio visualization based on user query
def get_looker_studio_viz(user_query):
    try:
        # AI logic to determine which iframe to show based on user intent
        prompt = f"""
        User query: "{user_query}"

        Determine which Looker Studio visualization to show:
        - basic_stats: for general statistics and overview
        - category_wise: for category-specific data
        - keywords_research: for keyword analysis
        - channel_growth: for channel growth and performance

        Respond with only one of these words: basic_stats, category_wise, keywords_research, channel_growth
        """

        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a data visualization expert. Respond with a single word only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )
        viz_type = response.choices[0].message.content.strip().lower()
        return LOOKER_STUDIO_IFRAMES.get(viz_type, LOOKER_STUDIO_IFRAMES['basic_stats'])
    except Exception as e:
        print(f"Error selecting visualization: {e}")
        return LOOKER_STUDIO_IFRAMES['basic_stats']

# Chat with data
def chat_with_data(user_query, analysis_data, df):
    try:
        hypothesis_detection_prompt = f"""
        The user has asked: "{user_query}"

        Determine if this can be formulated as a testable hypothesis using YouTube video data.
        If yes, respond with this exact JSON format:
        {{
            "is_hypothesis": true,
            "hypothesis": "clear statement if applicable",
            "test_metric": "primary metric to test",
            "condition": "how to split the data for comparison"
        }}

        If it's not a testable hypothesis, respond with:
        {{
            "is_hypothesis": false
        }}

        IMPORTANT: Respond ONLY with the JSON object, nothing else.
        """

        detection_response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst. Respond only with the specified JSON format."},
                {"role": "user", "content": hypothesis_detection_prompt}
            ],
            max_tokens=200
        )
        response_content = detection_response.choices[0].message.content.strip()
        try:
            hypothesis_info = json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response_content}")
            return chat_with_data_regular(user_query, analysis_data, df)

        return chat_with_data_regular(user_query, analysis_data, df)
    except Exception as e:
        print(f"Error in chat processing: {e}")
        return {"response": "I encountered an error while processing your request. Please try again.", 
                "requires_confirmation": False}

def chat_with_data_regular(user_query, analysis_data, df):
    try:
        chat_history.append({"role": "user", "content": user_query})
        video_stats = {
            "total_videos": len(df),
            "avg_views": float(df['view_count'].mean()),
            "avg_likes": float(df['like_count'].mean()),
            "avg_comments": float(df['comment_count'].mean()),
            "top_videos": get_top_videos(df, 3),
            "topic": df['topic'].unique().tolist(),  # Add topics
            "tone": df['tone'].unique().tolist(),    # Add tones
            "type": df['type'].unique().tolist()     # Add types
        }
        prompt = f"""
        You are analyzing YouTube video data.

        User question: "{user_query}"

        Video statistics summary:
        - Total videos: {video_stats['total_videos']}
        - Avg views: {int(video_stats['avg_views'])}
        - Avg likes: {int(video_stats['avg_likes'])}
        - Avg comments: {int(video_stats['avg_comments'])}
        - Topics: {', '.join(video_stats['topic'])}
        - Tones: {', '.join(video_stats['tone'])}
        - Types: {', '.join(video_stats['type'])}

        Provide a DIRECT, CONCISE answer (maximum 3 sentences).
        Your response should be:
        - brief
        - Direct to the point
        - Include exact numbers when requested
        - No explanations unless explicitly asked

        IMPORTANT: If the user is asking for a "top 5" or any ranking, simply provide the exact data requested without extra commentary.
        """

        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst assistant specializing in YouTube analytics."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )
        ai_response = response.choices[0].message.content
        chat_history.append({"role": "assistant", "content": ai_response})
        return {"response": ai_response, "requires_confirmation": False}
    except Exception as e:
        print(f"Error in regular chat processing: {e}")
        return {"response": "I encountered an error while processing your request. Please try again.", 
                "requires_confirmation": False}

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_chart_data', methods=['POST'])
def get_chart_data():
    try:
        data = request.json
        user_query = data.get('query', '')
        
        filepath = "https://www.dropbox.com/scl/fi/zvxl6im53o7aoy290530g/youtube_videos_Final_with_classification_and_similarity.csv?rlkey=b76k2j3ou74vj30l7m4egeidq&st=vh573jaz&dl=1"
        df = load_data(filepath)
        
        if df is None:
            return jsonify({'error': 'Failed to load data'}), 500
        
        # Determine chart type using AI
        chart_prompt = f"""
        User query: "{user_query}"
        
        Based on this query, determine the most appropriate chart type to visualize YouTube data.
        Choose from: pie, bar, line
        
        If the query is about proportions, categories, or distribution among different groups, use 'pie'.
        If the query is about comparing quantities across categories, use 'bar'.
        If the query is about trends over time or sequences, use 'line'.
        
        Respond with ONLY one word: pie, bar, or line
        """
        
        chart_response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a data visualization expert. Respond with only one word."},
                {"role": "user", "content": chart_prompt}
            ],
            max_tokens=10
        )
        
        chart_type = chart_response.choices[0].message.content.strip().lower()
        
        # Determine what data to extract based on the query
        data_prompt = f"""
        User query: "{user_query}"
        Available columns in the dataframe: {', '.join(df.columns.tolist())}
        
        Determine what data should be extracted to answer this query.
        Return a JSON object with:
        1. "category_column": column name to use for grouping/categories
        2. "value_column": column name to use for values (if applicable, otherwise just use "count")
        3. "operation": aggregation operation (count, sum, mean, max, min)
        4. "title": a short, descriptive title for the chart
        
        ONLY respond with valid JSON. No other text.
        """
        
        data_extraction_response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a data analysis expert. Respond with only JSON."},
                {"role": "user", "content": data_prompt}
            ],
            max_tokens=150
        )
        
        try:
            extraction_info = json.loads(data_extraction_response.choices[0].message.content.strip())
            category_col = extraction_info.get('category_column')
            value_col = extraction_info.get('value_column')
            operation = extraction_info.get('operation', 'count')
            chart_title = extraction_info.get('title', 'YouTube Data Analysis')
            
            # Generate chart data
            if operation == 'count':
                chart_data = df[category_col].value_counts().reset_index()
                chart_data.columns = ['name', 'value']
            elif operation == 'sum':
                chart_data = df.groupby(category_col)[value_col].sum().reset_index()
                chart_data.columns = ['name', 'value']
            elif operation == 'mean':
                chart_data = df.groupby(category_col)[value_col].mean().reset_index()
                chart_data.columns = ['name', 'value']
            elif operation == 'max':
                chart_data = df.groupby(category_col)[value_col].max().reset_index()
                chart_data.columns = ['name', 'value']
            elif operation == 'min':
                chart_data = df.groupby(category_col)[value_col].min().reset_index()
                chart_data.columns = ['name', 'value']
            else:
                chart_data = df[category_col].value_counts().reset_index()
                chart_data.columns = ['name', 'value']
            
            # Convert to list of dicts for JSON serialization
            chart_data_list = chart_data.to_dict('records')
            
            # Limit to top 10 values for better visualization
            if len(chart_data_list) > 10:
                chart_data_list = sorted(chart_data_list, key=lambda x: x['value'], reverse=True)[:10]
            
            return jsonify({
                'chartType': chart_type,
                'chartData': chart_data_list,
                'chartTitle': chart_title
            })
            
        except json.JSONDecodeError:
            return jsonify({
                'error': 'Failed to parse AI response',
                'chartType': 'bar',
                'chartData': []
            }), 500
            
    except Exception as e:
        print(f"Error generating chart data: {e}")
        return jsonify({
            'error': 'An error occurred while generating chart data',
            'message': str(e)
        }), 500

        
@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        filepath = "your_csv_url_here"
        df = load_data(filepath)
        
        if df is None or df.empty:
            return jsonify({
                'error': 'Failed to load data',
                'summary_stats': {
                    'total_videos': 0,
                    'avg_views': 0,
                    'avg_likes': 0,
                    'avg_comments': 0
                },
                'top_videos': [],
                'insights': 'No data available',
                'viz_url': None
            }), 500

        # Calculate statistics safely
        summary_stats = {
            "total_videos": len(df),
            "avg_views": int(df['view_count'].mean()) if 'view_count' in df.columns else 0,
            "avg_likes": int(df['like_count'].mean()) if 'like_count' in df.columns else 0,
            "avg_comments": int(df['comment_count'].mean()) if 'comment_count' in df.columns else 0
        }

        try:
            results = dynamic_hypothesis_testing(df, lambda df: df['engagement_score'] > df['engagement_score'].median())
        except Exception as e:
            logger.error(f"Error in hypothesis testing: {str(e)}")
            results = {}

        try:
            insights = generate_insights(results) if results else "No insights available"
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights = "Error generating insights"

        try:
            top_videos = get_top_videos(df) if not df.empty else []
        except Exception as e:
            logger.error(f"Error getting top videos: {str(e)}")
            top_videos = []

        return jsonify({
            'error': None,
            'analysis_results': results,
            'insights': insights,
            'top_videos': top_videos,
            'summary_stats': summary_stats,
            'viz_url': LOOKER_STUDIO_IFRAMES.get('basic_stats', '')
        })

    except Exception as e:
        logger.error(f"Error in analyze route: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An error occurred during analysis',
            'message': str(e),
            'summary_stats': {
                'total_videos': 0,
                'avg_views': 0,
                'avg_likes': 0,
                'avg_comments': 0
            },
            'top_videos': [],
            'analysis_results': {},
            'insights': 'Error generating insights',
            'viz_url': None
        }), 500



@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_query = data.get('message', '')
        is_confirmation = data.get('is_confirmation', False)
        hypothesis_info = data.get('hypothesis_info', None)
        filepath = "https://www.dropbox.com/scl/fi/zvxl6im53o7aoy290530g/youtube_videos_Final_with_classification_and_similarity.csv?rlkey=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        df = load_data(filepath)  # Removed on_bad_lines parameter
        if df is None:
            return jsonify({'error': 'Failed to load data'}), 500
        
        initial_analysis = dynamic_hypothesis_testing(df, lambda df: df['engagement_score'] > df['engagement_score'].median())
        
        if is_confirmation and hypothesis_info:
            results = dynamic_hypothesis_testing(
                df,
                lambda df: df[hypothesis_info['test_metric']] > df[hypothesis_info['test_metric']].median(),
                hypothesis_info['test_metric']
            )
            interpretation = interpret_results(results, hypothesis_info['hypothesis'])
            viz_url = get_looker_studio_viz(user_query)
            insights = generate_insights(results)
            return jsonify({
                'response': interpretation,
                'analysis_results': results,
                'type': 'hypothesis_test',
                'viz_url': viz_url,
                'insights': insights
            })
        
        chat_response = chat_with_data(user_query, initial_analysis, df)
        viz_url = get_looker_studio_viz(user_query)
        chat_response['viz_url'] = viz_url
        query_insights = generate_insights(initial_analysis)
        chat_response['insights'] = query_insights
        return jsonify(chat_response)
    
    except Exception as e:
        print(f"Error in chat route: {e}")
        return jsonify({
            'response': 'I encountered an error while processing your request. Please try again.',
            'type': 'error',
            'viz_url': LOOKER_STUDIO_IFRAMES['basic_stats'],
            'insights': 'Unable to generate insights due to an error.'
        }), 500



if __name__ == '__main__':
    app.run(debug=True)
