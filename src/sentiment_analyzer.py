import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    pipeline,
    AutoModelForSeq2SeqLM
)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect, LangDetectException
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    An enhanced class for text analysis using multiple models and techniques:
    - Emotion detection using pre-trained transformer models
    - Sentiment intensity analysis using VADER
    - Language detection for multi-language support
    - Text summarization
    - Toxicity detection
    - Configurable thresholds and parameters
    """
    
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ru': 'Russian',
        'ar': 'Arabic',
        'hi': 'Hindi'
    }
    
    # Default thresholds
    DEFAULT_THRESHOLDS = {
        'emotion_confidence': 0.3,  # Minimum confidence to consider an emotion
        'toxicity': 0.7,            # Threshold for toxic content
        'summarization_min_length': 30,
        'summarization_max_length': 100
    }
    
    def __init__(self, 
                 emotion_model: str = "bhadresh-savani/bert-base-uncased-emotion",
                 toxicity_model: str = "unitary/toxic-bert",
                 summarizer_model: str = "facebook/bart-large-cnn",
                 thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the sentiment analyzer with multiple models and capabilities.
        
        Args:
            emotion_model: Model for emotion detection
            toxicity_model: Model for toxicity detection
            summarizer_model: Model for text summarization
            thresholds: Custom thresholds for analysis parameters
        """
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize thresholds with defaults and any custom values
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)
        
        # Initialize emotion detection model
        self._init_emotion_model(emotion_model)
        
        # Initialize VADER for sentiment intensity
        try:
            import nltk
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon')
            self.vader = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("NLTK VADER not available. Sentiment intensity analysis disabled.")
            self.vader = None
            
        # Initialize toxicity detection
        try:
            self.toxicity_pipeline = pipeline(
                "text-classification", 
                model=toxicity_model,
                device=0 if self.device.type == "cuda" else -1
            )
        except Exception as e:
            logger.warning(f"Could not initialize toxicity model: {e}")
            self.toxicity_pipeline = None
            
        # Initialize summarizer
        try:
            self.summarizer = pipeline(
                "summarization", 
                model=summarizer_model,
                device=0 if self.device.type == "cuda" else -1
            )
        except Exception as e:
            logger.warning(f"Could not initialize summarizer: {e}")
            self.summarizer = None
    
    def _init_emotion_model(self, model_name: str):
        """Initialize the emotion detection model."""
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get emotion labels from the model config
            self.labels = self.model.config.id2label
            logger.info(f"Initialized emotion model with labels: {self.labels}")
        except Exception as e:
            logger.error(f"Error initializing emotion model: {e}")
            raise
    
    def detect_language(self, text: str) -> Tuple[str, str]:
        """
        Detect the language of the input text.
        
        Args:
            text: The text to analyze
            
        Returns:
            A tuple of (language_code, language_name)
        """
        try:
            lang_code = detect(text)
            lang_name = self.SUPPORTED_LANGUAGES.get(lang_code, "Unknown")
            return (lang_code, lang_name)
        except LangDetectException:
            return ('en', 'English (default)')
    
    def analyze_sentiment_intensity(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment intensity using VADER.
        
        Args:
            text: The text to analyze
            
        Returns:
            A dictionary with positive, negative, neutral scores and a compound score
        """
        if not self.vader:
            return {
                'positive': 0.0,
                'negative': 0.0, 
                'neutral': 1.0,
                'compound': 0.0
            }
            
        return self.vader.polarity_scores(text)
    
    def detect_toxicity(self, text: str) -> Dict[str, Any]:
        """
        Detect toxicity in the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            A dictionary with toxicity analysis results
        """
        if not self.toxicity_pipeline:
            return {'is_toxic': False, 'score': 0.0}
            
        try:
            result = self.toxicity_pipeline(text)[0]
            score = result['score']
            is_toxic = score >= self.thresholds['toxicity']
            label = result['label']
            
            return {
                'is_toxic': is_toxic,
                'score': score,
                'label': label
            }
        except Exception as e:
            logger.error(f"Error in toxicity detection: {e}")
            return {'is_toxic': False, 'score': 0.0, 'error': str(e)}
    
    def summarize_text(self, text: str, max_length: int = None, min_length: int = None) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text: The text to summarize
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary
            
        Returns:
            A summarized version of the input text
        """
        if not self.summarizer:
            return ""
            
        if len(text.split()) < 30:  # Don't summarize short texts
            return text
            
        try:
            min_length = min_length or self.thresholds['summarization_min_length']
            max_length = max_length or self.thresholds['summarization_max_length']
            
            summary = self.summarizer(
                text, 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=False
            )[0]['summary_text']
            
            return summary
        except Exception as e:
            logger.error(f"Error in text summarization: {e}")
            return ""
    
    def analyze(self, text: str, include_all: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the given text.
        
        Args:
            text: The text to analyze
            include_all: Whether to include all available analyses
            
        Returns:
            A dictionary with comprehensive analysis results
        """
        result = {'text': text}
        
        # Detect language
        lang_code, lang_name = self.detect_language(text)
        result['language'] = {'code': lang_code, 'name': lang_name}
        
        # Basic emotion analysis
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            
            # Convert to dictionary of emotion -> probability
            emotions = {self.labels[i]: float(probabilities[i]) for i in range(len(self.labels))}
            result['emotions'] = emotions
            
            # Get dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            result['dominant_emotion'] = {
                'name': dominant_emotion[0],
                'score': dominant_emotion[1]
            }
            
            # Filter significant emotions
            threshold = self.thresholds['emotion_confidence']
            result['significant_emotions'] = {
                emotion: score for emotion, score in emotions.items() 
                if score >= threshold
            }
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            result['emotions_error'] = str(e)
        
        # Include additional analyses if requested
        if include_all:
            # Sentiment intensity
            result['sentiment'] = self.analyze_sentiment_intensity(text)
            
            # Toxicity detection
            result['toxicity'] = self.detect_toxicity(text)
            
            # Text summarization (for longer texts)
            if len(text.split()) > 30:
                result['summary'] = self.summarize_text(text)
        
        return result
    
    def analyze_batch(self, texts: List[str], include_all: bool = False) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts with comprehensive analysis.
        
        Args:
            texts: A list of texts to analyze
            include_all: Whether to include all available analyses
            
        Returns:
            A list of dictionaries with analysis results
        """
        if not texts:
            return []
            
        results = []
        
        # First get the emotion predictions in batch for efficiency
        try:
            # Tokenize inputs
            inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                
            # Process each text individually for complete analysis
            for i, text in enumerate(texts):
                result = {'text': text}
                
                # Emotion analysis from batch prediction
                emotions = {self.labels[j]: float(probabilities[i][j]) for j in range(len(self.labels))}
                result['emotions'] = emotions
                
                # Get dominant emotion
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                result['dominant_emotion'] = {
                    'name': dominant_emotion[0],
                    'score': dominant_emotion[1]
                }
                
                # Filter significant emotions
                threshold = self.thresholds['emotion_confidence']
                result['significant_emotions'] = {
                    emotion: score for emotion, score in emotions.items() 
                    if score >= threshold
                }
                
                # Detect language
                lang_code, lang_name = self.detect_language(text)
                result['language'] = {'code': lang_code, 'name': lang_name}
                
                # Include additional analyses if requested
                if include_all:
                    # Sentiment intensity
                    result['sentiment'] = self.analyze_sentiment_intensity(text)
                    
                    # Toxicity detection
                    result['toxicity'] = self.detect_toxicity(text)
                    
                    # Text summarization (for longer texts)
                    if len(text.split()) > 30:
                        result['summary'] = self.summarize_text(text)
                
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            # Return simplified results with error information
            for text in texts:
                results.append({
                    'text': text,
                    'error': str(e)
                })
        
        return results
    
    def get_dominant_emotion(self, text: str) -> Tuple[str, float]:
        """
        Get the dominant emotion in the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            A tuple containing the dominant emotion and its probability
        """
        try:
            analysis = self.analyze(text)
            return (analysis['dominant_emotion']['name'], analysis['dominant_emotion']['score'])
        except Exception as e:
            logger.error(f"Error getting dominant emotion: {e}")
            return ("unknown", 0.0)
    
    def set_threshold(self, threshold_name: str, value: float) -> bool:
        """
        Update a threshold setting.
        
        Args:
            threshold_name: Name of the threshold to update
            value: New threshold value
            
        Returns:
            True if successful, False otherwise
        """
        if threshold_name not in self.thresholds:
            logger.warning(f"Unknown threshold: {threshold_name}")
            return False
            
        if not isinstance(value, (int, float)):
            logger.warning(f"Threshold value must be numeric, got {type(value)}")
            return False
            
        self.thresholds[threshold_name] = value
        logger.info(f"Updated threshold '{threshold_name}' to {value}")
        return True
    
    def get_thresholds(self) -> Dict[str, float]:
        """
        Get current threshold settings.
        
        Returns:
            Dictionary of current thresholds
        """
        return self.thresholds.copy()
    
    def reset_thresholds(self) -> None:
        """Reset all thresholds to their default values."""
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        logger.info("Reset all thresholds to default values")
    
    def get_available_models(self) -> Dict[str, str]:
        """
        Get information about currently loaded models.
        
        Returns:
            Dictionary with model information
        """
        models = {
            'emotion': "Active" if hasattr(self, 'model') else "Not loaded",
            'toxicity': "Active" if self.toxicity_pipeline else "Not loaded",
            'summarization': "Active" if self.summarizer else "Not loaded",
            'sentiment': "Active" if self.vader else "Not loaded"
        }
        return models
    
    def handle_multilingual_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process text based on its detected language.
        For non-English text, provides translation information.
        
        Args:
            text: The input text
            
        Returns:
            Tuple of (processed_text, language_info)
        """
        lang_code, lang_name = self.detect_language(text)
        
        language_info = {
            'detected': {
                'code': lang_code,
                'name': lang_name
            },
            'is_supported_for_emotion': lang_code == 'en'  # Most emotion models work best with English
        }
        
        # If not English and needs translation (could integrate translation service here)
        if lang_code != 'en':
            language_info['needs_translation'] = True
            
        return (text, language_info)

