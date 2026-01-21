# 📚 Book Recommendation System - SBERT + GenAI

An intelligent book recommendation system using semantic analysis (SBERT) and generative AI (Google Gemini) to provide personalized literary suggestions.

## 🌟 Features

- **Hybrid Questionnaire**: Open-ended questions + Likert scale preferences
- **Semantic Analysis**: SBERT embeddings for zero-cost similarity matching
- **Smart Scoring**: Weighted similarity calculation (80% semantic + 20% preferences)
- **GenAI Enhancement**: Conditional query enrichment and personalized summaries
- **Web Interface**: User-friendly Flask-based UI

## 🏗️ Architecture

```
User Input → SBERT Encoding → Cosine Similarity → Top 3 Books → GenAI Summary
```

## 📦 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/Simone-Me/suggest_book.git
cd suggest_book
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Set up API key** (optional, for GenAI features):

```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

Get a free Gemini API key: https://makersuite.google.com/app/apikey

## 🚀 Usage

### Command Line Version

```bash
python book_recommendation_system.py
```

### Web Interface

```bash
python app.py
```

Then open http://localhost:5000 in your browser.

## 📊 Dataset

The system uses `Book_Dataset_1.csv` containing 700+ books with:

- Title
- Genre
- Description
- Price & ratings

## 🧠 How It Works

1. **Data Collection** (EF1): User answers 7 questions (3 open + 4 Likert)
2. **Semantic Modeling** (EF2): SBERT converts text to 384D vectors
3. **Scoring** (EF3): Calculates weighted similarity scores
4. **Recommendations** (EF3.2): Returns top 3 matches
5. **GenAI Summary** (EF4): Optional personalized explanation

## 📁 Project Structure

```
.
├── book_recommendation_system.py   # Main CLI script
├── app.py                          # Web application
├── Book_Dataset_1.csv              # Book database
├── templates/
│   └── index.html                  # Web UI
├── analysis.py                     # Analysis scripts
├── GUIDE_COMPLET.md                # Complete guide (French)
├── README_WEB.md                   # Web interface guide
└── requirements.txt                # Dependencies
```

## 🔍 Key Technologies

- **SentenceTransformers**: Local semantic embeddings (all-MiniLM-L6-v2)
- **Flask**: Web framework
- **Google Gemini**: GenAI for text enrichment
- **scikit-learn**: Cosine similarity calculation
- **Pandas/NumPy**: Data processing

## 📈 Performance

- **Dataset Size**: 700+ books
- **Embedding Model**: 384 dimensions
- **Similarity Metric**: Cosine similarity
- **Weighting**: 80% semantic + 20% Likert preferences

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is developed as part of EFREI M1 Data Engineering - Generative AI course.

## 👥 Authors

- **Maharo** - **Simone** - EFREI M1 Data Engineering

## 🙏 Acknowledgments

- EFREI Paris for the educational framework
- Google Gemini for GenAI capabilities
- SentenceTransformers community

---

**Note**: This is an educational project demonstrating NLP and GenAI techniques for personalized recommendations.
