

# AI-Text-Analyzer
Analyze emotions, sentiment, and toxicity in text with a beautiful, modern web app.

>>>>>>> c27c8aad513bc48497e688168a0cd5a99c254b13
# AI Text Analyzer

A modern, dynamic web app for analyzing text using AI. Detects emotions, sentiment, toxicity, and more—powered by BERT, Transformers, and advanced NLP models.

## 🚀 Features
- Emotion detection (joy, sadness, anger, etc.)
- Sentiment analysis (positive, neutral, negative, compound)
- Toxicity detection
- Text summarization
- Batch analysis
- Animated, modern UI with dark/light mode
- API status badge
- Feedback modal

## 🛠️ Setup
1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```
2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download NLTK data:**
   ```bash
   python download_nltk_data.py
   ```
5. **Run the app locally:**
   ```bash
   python src/app.py
   ```
   Visit [http://localhost:5001](http://localhost:5001)

## 🌐 Deployment

### Deploy Free on Render
1. Push your code to GitHub.
2. Go to [https://render.com/](https://render.com/) and create a new Web Service.
3. Connect your repo, set build command:
   ```
   pip install -r requirements.txt
   ```
   and start command:
   ```
   gunicorn -w 4 -b 0.0.0.0:10000 src.app:app
   ```
4. Deploy and get your public URL!

### Deploy Free on Railway
1. Go to [https://railway.app/](https://railway.app/) and create a new project.
2. Connect your GitHub repo.
3. Set the start command:
   ```
   gunicorn -w 4 -b 0.0.0.0:10000 src.app:app
   ```
4. Deploy and get your public URL!

## 📦 API Usage
- See `/api/health` for status
- `/api/analyze` and `/api/analyze-batch` for programmatic access

## 🙏 Credits
- Built with [Flask](https://flask.palletsprojects.com/), [Transformers](https://huggingface.co/transformers/), [NLTK](https://www.nltk.org/), and more.
- UI inspired by Bolt AI and modern web design best practices.

---

<<<<<<< HEAD
**Feel free to fork, star, and contribute!** 
=======
**Feel free to fork, star, and contribute!**

>>>>>>> c27c8aad513bc48497e688168a0cd5a99c254b13
