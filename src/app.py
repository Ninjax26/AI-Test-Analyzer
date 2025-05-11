import os
import json
import threading
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.exceptions import BadRequest
from sentiment_analyzer import SentimentAnalyzer

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'))

# Thread-safe model initialization
_analyzer = None
_analyzer_lock = threading.Lock()

# Custom analyzer configuration
ANALYZER_CONFIG = {
    'emotion_model': "bhadresh-savani/bert-base-uncased-emotion",
    'toxicity_model': "unitary/toxic-bert",
    'summarizer_model': "facebook/bart-large-cnn",
    'thresholds': {
        'emotion_confidence': 0.25,  # More sensitive than default
        'toxicity': 0.65,            # More sensitive than default
        'summarization_min_length': 30,
        'summarization_max_length': 150
    }
}

def get_analyzer():
    """
    Get the sentiment analyzer instance with all features enabled.
    Initializes the models if they haven't been initialized yet.
    Uses thread-safe initialization.
    """
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:  # Double-check pattern for thread safety
                print("Initializing enhanced sentiment analyzer models...")
                _analyzer = SentimentAnalyzer(
                    emotion_model=ANALYZER_CONFIG['emotion_model'],
                    toxicity_model=ANALYZER_CONFIG['toxicity_model'],
                    summarizer_model=ANALYZER_CONFIG['summarizer_model'],
                    thresholds=ANALYZER_CONFIG['thresholds']
                )
                print("Models initialized and ready for inference")
                print(f"Available models: {_analyzer.get_available_models()}")
    return _analyzer

@app.route('/')
def index():
    """Render the main web interface."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Handle comprehensive text analysis requests from the web interface.
    
    Expects form data with a 'text' field containing the text to analyze.
    Returns the analysis results rendered in HTML.
    """
    try:
        text = request.form.get('text', '')
        include_all = request.form.get('include_all', 'false').lower() == 'true'
        
        if not text:
            return render_template('index.html', error="Please provide some text to analyze.")
        
        # Get analyzer instance
        analyzer = get_analyzer()
        
        # Analyze the text with comprehensive analysis
        analysis = analyzer.analyze(text, include_all=True)  # Always include all analysis
        
        # Extract necessary data
        dominant_emotion = analysis.get('dominant_emotion', {}).get('name', 'unknown')
        confidence = analysis.get('dominant_emotion', {}).get('score', 0) * 100  # Convert to percentage
        emotions = analysis.get('emotions', {})
        
        # Extract language information
        language = analysis.get('language', {'code': 'en', 'name': 'English'})
        
        # Get toxicity information if available
        toxicity = analysis.get('toxicity', {'is_toxic': False, 'score': 0})
        
        # Get sentiment analysis if available
        sentiment = analysis.get('sentiment', {})
        if sentiment:
            sentiment = {
                'positive': sentiment.get('pos', 0.0),
                'negative': sentiment.get('neg', 0.0),
                'neutral': sentiment.get('neu', 0.0),
                'compound': sentiment.get('compound', 0.0)
            }
        
        # Get summary if available
        summary = analysis.get('summary', '')
        
        return render_template('index.html', 
                              text=text,
                              analysis=analysis,
                              results=emotions,
                              dominant_emotion=dominant_emotion,
                              confidence=confidence,
                              language=language,
                              toxicity=toxicity,
                              sentiment=sentiment,
                              summary=summary,
                              include_all=include_all)
    
    except Exception as e:
        app.logger.error(f"Error analyzing text: {str(e)}")
        return render_template('index.html', error=f"An error occurred: {str(e)}")

@app.route('/analyze-batch', methods=['POST'])
def analyze_batch():
    """
    Handle batch text analysis requests from the web interface.
    
    Expects form data with a 'texts' field containing multiple lines of text to analyze.
    Returns the comprehensive analysis results rendered in HTML.
    """
    try:
        text_batch = request.form.get('texts', '')
        include_all = request.form.get('include_all', 'false').lower() == 'true'
        
        if not text_batch:
            return render_template('index.html', error="Please provide some texts to analyze.")
        
        # Split the text by lines
        texts = [t.strip() for t in text_batch.split('\n') if t.strip()]
        
        # Get analyzer instance
        analyzer = get_analyzer()
        
        # Analyze the batch of texts with comprehensive analysis if requested
        batch_results = analyzer.analyze_batch(texts, include_all=include_all)
        
        return render_template('index.html', 
                              batch_results=batch_results,
                              texts=text_batch,
                              include_all=include_all)
    
    except Exception as e:
        app.logger.error(f"Error analyzing batch: {str(e)}")
        return render_template('index.html', error=f"An error occurred: {str(e)}")

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    API endpoint for comprehensive text analysis.
    
    Expects JSON with a 'text' field and optional 'include_all' boolean.
    Returns JSON with comprehensive analysis results.
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            raise BadRequest("Missing 'text' field in request")
        
        text = data['text']
        include_all = data.get('include_all', True)  # Default to full analysis
        
        # Get analyzer instance
        analyzer = get_analyzer()
        
        # Analyze the text with comprehensive analysis
        analysis = analyzer.analyze(text, include_all=include_all)
        
        return jsonify({
            'analysis': analysis,
            'metadata': {
                'models': analyzer.get_available_models(),
                'thresholds': analyzer.get_thresholds()
            }
        })
    
    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        app.logger.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-batch', methods=['POST'])
def api_analyze_batch():
    """
    API endpoint for batch comprehensive text analysis.
    
    Expects JSON with a 'texts' field containing an array of texts and optional 'include_all' boolean.
    Returns JSON with comprehensive analysis results for each text.
    """
    try:
        data = request.get_json()
        if not data or 'texts' not in data or not isinstance(data['texts'], list):
            raise BadRequest("Missing or invalid 'texts' field in request")
        
        texts = data['texts']
        include_all = data.get('include_all', True)  # Default to full analysis
        
        # Get analyzer instance
        analyzer = get_analyzer()
        
        # Analyze the batch of texts with comprehensive analysis
        results = analyzer.analyze_batch(texts, include_all=include_all)
        
        return jsonify({
            'batch_results': results,
            'metadata': {
                'models': analyzer.get_available_models(),
                'thresholds': analyzer.get_thresholds()
            }
        })
    
    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        app.logger.error(f"API batch error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Enable CORS for API endpoints
@app.after_request
def add_cors_headers(response):
    """Add CORS headers to allow cross-origin requests to the API."""
    if request.path.startswith('/api/'):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint for health check."""
    global _analyzer
    
    # Basic health check
    health_info = {'status': 'ok', 'model_loaded': _analyzer is not None}
    
    # Add detailed model status if analyzer is initialized
    if _analyzer is not None:
        health_info['models'] = _analyzer.get_available_models()
        health_info['thresholds'] = _analyzer.get_thresholds()
        
    return jsonify(health_info)

# Additional routes for new features

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    """API endpoint for text summarization."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            raise BadRequest("Missing 'text' field in request")
        
        text = data['text']
        min_length = data.get('min_length')
        max_length = data.get('max_length')
        
        analyzer = get_analyzer()
        summary = analyzer.summarize_text(text, min_length=min_length, max_length=max_length)
        
        return jsonify({
            'original_text': text,
            'summary': summary,
            'original_length': len(text.split()),
            'summary_length': len(summary.split()) if summary else 0
        })
    
    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        app.logger.error(f"API error in summarization: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/toxicity', methods=['POST'])
def api_toxicity():
    """API endpoint for toxicity detection."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            raise BadRequest("Missing 'text' field in request")
        
        text = data['text']
        analyzer = get_analyzer()
        toxicity = analyzer.detect_toxicity(text)
        
        return jsonify({
            'text': text,
            'toxicity': toxicity
        })
    
    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        app.logger.error(f"API error in toxicity detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)

