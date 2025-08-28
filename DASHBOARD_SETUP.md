# 🎛️ Dashboard Setup Instructions

## Quick Start (Streamlit - Recommended for Demo)

1. **Install dependencies:**
   ```bash
   pip install -r dashboard_requirements.txt
   ```

2. **Run the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

3. **Open in browser:** 
   - Automatically opens at `http://localhost:8501`

## Alternative: Flask Dashboard

1. **Install Flask:**
   ```bash
   pip install flask
   ```

2. **Run the Flask app:**
   ```bash
   python flask_dashboard.py
   ```

3. **Open in browser:**
   - Visit `http://localhost:5000`

## Features

### 🏠 **Overview Page**
- Key metrics and statistics
- Action distribution charts
- Recent activity timeline

### 📝 **Process Reviews**
- Single review processing
- Batch upload from CSV
- Real-time results

### 📊 **Analytics**
- Performance metrics
- Confidence distributions
- Quality trends

### ⚙️ **Model Information**
- Model performance stats
- Feature importance
- Configuration details

### 👥 **Manual Review Queue**
- Flagged reviews needing human attention
- Approve/Remove actions
- Priority queue management

## Demo Data

The dashboard includes sample data generation for demo purposes. In production, connect it to your actual model pipeline:

```python
# Replace the simulation with your actual pipeline
result = production_pipeline.process_single_review(
    review_text=data['text'],
    rating=data['rating'],
    author_name=data.get('author', ''),
    business_name=data.get('business', '')
)
```

## Hackathon Tips

1. **Start with Streamlit** - Faster setup, more interactive
2. **Use demo mode** - Shows pipeline capabilities without model file
3. **Prepare sample reviews** - Have interesting examples ready
4. **Customize metrics** - Add your specific KPIs
5. **Mobile responsive** - Works on judge's tablets/phones

## Production Deployment

For production deployment, consider:
- Database integration (PostgreSQL/MongoDB)
- User authentication
- API rate limiting
- Real-time notifications
- Model monitoring
- A/B testing framework
