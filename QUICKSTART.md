# 🚀 Quick Start Guide - Fashion Shopping Assistant

Get up and running with the Fashion Shopping Assistant in under 5 minutes!

## Prerequisites

- Python 3.8+ installed
- 4GB+ RAM
- Internet connection for dataset download

## Step-by-Step Setup

### 1️⃣ Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

### 2️⃣ Download Dataset (2-3 minutes)

```bash
python download_dataset.py
```

**Options:**
- **Option 1**: Full dataset from Kaggle (requires account)
- **Option 2**: Sample dataset for quick testing (no account needed)

For quick testing, choose Option 2.

### 3️⃣ Run the Application (30 seconds)

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 🎯 First Time Usage

### What to Expect on First Run

1. **Feature Extraction** (5-10 minutes on first run)
   - The app will extract visual features from all images
   - This happens only once; features are cached for future use
   - Progress bar will show extraction status

2. **Index Building** (< 1 minute)
   - Creates fast similarity search index
   - Automatic and one-time process

### Quick Test Workflow

1. **Browse Catalog**
   - Click "Browse Catalog" in sidebar
   - Apply filters (Gender, Category, Color)
   - Navigate through product pages

2. **Find Similar Items**
   - Select "Similar Items" mode
   - Click "🎲 Random Product" for quick selection
   - View 8 visually similar products

3. **Get Recommendations**
   - Choose "Smart Recommendations"
   - Try different recommendation types:
     - Diverse Recommendations
     - Complementary Items
     - Complete the Outfit

4. **Upload Your Image**
   - Select "Upload Image" mode
   - Upload a fashion item photo
   - Find similar items in catalog

## 🔧 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Dataset not found" | Run `python download_dataset.py` |
| Import errors | Run `pip install -r requirements.txt` |
| Out of memory | Use smaller dataset (Option 2 in download) |
| Slow performance | Enable GPU if available or reduce batch size |

### Verify Installation

Run the test script to check everything is working:

```bash
python test_installation.py
```

## 📊 Sample Dataset Info

If using the sample dataset (Option 2):
- 100 products with metadata
- 20 sample images
- Perfect for testing and development
- Full functionality available

## 🎨 Features to Try

### Filters
- **Gender**: Men, Women, Boys, Girls
- **Category**: Apparel, Footwear, Accessories
- **Color**: 14+ color options
- **Season**: Summer, Winter, Spring, Fall

### Recommendation Types
- **Similar Items**: Visual similarity search
- **Diverse**: Balanced variety in recommendations
- **Complementary**: Different categories, similar style
- **Outfit**: Complete outfit suggestions

## 📱 Interface Navigation

```
┌─────────────────────────────────────┐
│         Fashion Assistant           │
├──────────┬──────────────────────────┤
│          │                          │
│ Sidebar  │    Main Content Area     │
│          │                          │
│ • Mode   │    • Product Grid        │
│ • Filters│    • Recommendations     │
│ • Stats  │    • Upload Area         │
│          │                          │
└──────────┴──────────────────────────┘
```

## 🔥 Pro Tips

1. **Faster Loading**: Pre-compute features by running the app once and letting it extract all features
2. **Better Results**: Use full dataset (Option 1) for more diverse recommendations
3. **GPU Acceleration**: If you have NVIDIA GPU, PyTorch will automatically use it
4. **Custom Images**: For best results, use clear product images with plain backgrounds

## 📝 Next Steps

After getting familiar with the basic features:

1. **Explore the Code**
   - `app.py`: Main Streamlit application
   - `similarity_search.py`: Recommendation algorithms
   - `feature_extractor.py`: Deep learning models

2. **Customize Settings**
   - Edit `config.py` to adjust parameters
   - Change batch size, model type, or UI settings

3. **Use Jupyter Notebook**
   - Open `fashion_assistant_notebook.py` in Jupyter
   - Interactive exploration and experimentation

## 🆘 Need Help?

- Check the full README.md for detailed documentation
- Run `python test_installation.py` to diagnose issues
- Review error messages in the terminal

## 🎉 Ready to Go!

You're all set! The Fashion Shopping Assistant is ready to help you explore and discover fashion items using AI-powered recommendations.

**Happy Shopping! 🛍️**