# Frontend Quick Start Guide

## 🚀 Getting Started

### 1. Navigate to Frontend Directory
```bash
cd frontend
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Start Development Server
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 📁 Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   └── page.js              # Main home page
│   ├── components/
│   │   ├── SearchBar.js         # Search input component
│   │   ├── PredictionCard.js     # Results display
│   │   └── LoadingSpinner.js    # Loading animation
│   └── lib/
│       └── api.js               # API integration
├── public/                       # Static assets
└── package.json
```

---

## 🎯 Features Implemented

✅ **Search Bar**
- Input HuggingFace model URL or ID
- Example quick links
- Validation and error handling

✅ **Loading State**
- Animated spinner
- Progress message
- Disabled interactions during prediction

✅ **Results Display**
- Predicted popularity badge (High/Low)
- Confidence score with progress bar
- Feature breakdown
- Link to HuggingFace model page
- Responsive design

✅ **Error Handling**
- Clear error messages
- Validation feedback
- Graceful fallbacks

---

## 🔧 How It Works

### 1. User enters model URL
```javascript
// Example: google/gemma-2-2b
// Or: https://huggingface.co/google/gemma-2-2b
```

### 2. Frontend sends API request
```javascript
POST http://localhost:8000/api/v1/predict
Body: { "huggingface_url": "..." }
```

### 3. Backend processes
- Extracts features
- Runs ML model
- Returns prediction

### 4. Frontend displays results
- Shows prediction (High/Low)
- Displays confidence score
- Lists key features
- Provides metadata

---

## 🧪 Testing

### Test with High Popularity Model
```bash
# In browser, enter:
google/gemma-2-2b
```

### Test with Low Popularity Model
```bash
# In browser, enter:
JunhaoZhuang/FlashVSR
```

---

## 🎨 Customization

### Change API URL
Edit `src/lib/api.js`:
```javascript
const API_BASE_URL = 'http://your-api-url:8000';
```

### Modify Colors
The components use Tailwind CSS. Update classes in:
- `SearchBar.js`
- `PredictionCard.js`
- `page.js`

### Add More Features
Create new components in `src/components/` and import them.

---

## 🐛 Troubleshooting

### "Failed to fetch"
- Make sure backend is running: `cd ../server && python3 app.py`
- Check API URL in `src/lib/api.js`
- Verify CORS is enabled in backend

### "Module not found"
- Run `npm install`
- Check file paths in imports

### No results showing
- Check browser console for errors
- Verify API endpoint is working: `curl http://localhost:8000/health`

---

## 📝 Next Steps

1. ✅ Start the frontend: `npm run dev`
2. ✅ Start the backend: `cd ../server && python3 app.py`
3. ✅ Test with example models
4. 🔄 Customize styling (if needed)
5. 🔄 Add more features from PRD
6. 🚀 Deploy to production

---

## 🚀 Deployment

### Vercel (Recommended)
```bash
npm run build
vercel deploy
```

### Netlify
```bash
npm run build
# Netlify will auto-deploy from git
```

### Manual Build
```bash
npm run build
npm start  # Runs on port 3000
```

---

## 📚 Resources

- **Backend API**: `server/app.py`
- **API Docs**: http://localhost:8000/docs
- **PRD**: `PRD.md`
- **Integration Guide**: `API_INTEGRATION_GUIDE.md`

