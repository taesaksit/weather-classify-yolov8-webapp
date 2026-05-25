import React, { useState, useEffect, useRef } from 'react';

// Access 'ort' from window (loaded via script tag in index.html)
const ort = (window as any).ort;

// Configuration Constants
const MODEL_URL = '/models/best_model_v3.onnx';
const INPUT_SIZE = 224;

// Metadata for the weather classes
const WEATHER_CLASSES = [
  { label: 'Cloudy', icon: '☁️', desc: 'Overcast sky with heavy clouds', color: 'bg-slate-400' },
  { label: 'Rain', icon: '🌧️', desc: 'Rainy conditions detected', color: 'bg-blue-400' },
  { label: 'Sunrise', icon: '🌅', desc: 'Sunrise or sunset lighting detected', color: 'bg-orange-400' },
  { label: 'Shine', icon: '☀️', desc: 'Clear weather with bright sunlight', color: 'bg-yellow-400' },
];

const App: React.FC = () => {
  // --- States ---
  const [session, setSession] = useState<any>(null);
  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<any[] | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isDarkMode, setIsDarkMode] = useState(true);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- Initialize Model ---
  useEffect(() => {
    const initModel = async () => {
      if (!ort) {
        setErrorMessage('Could not load ONNX Runtime. Please check your internet connection.');
        setModelStatus('error');
        return;
      }

      try {
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
        ort.env.wasm.numThreads = 1;
        
        const newSession = await ort.InferenceSession.create(MODEL_URL, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
        
        setSession(newSession);
        setModelStatus('ready');
      } catch (err: any) {
        console.error(err);
        setModelStatus('error');
        setErrorMessage('Model loading failed: ' + err.message);
      }
    };

    initModel();
  }, []);

  // --- Theme Management ---
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  // --- Handlers ---
  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file only.');
      return;
    }
    
    setSelectedFile(file);
    setResults(null);
    setErrorMessage(null);

    const reader = new FileReader();
    reader.onload = (e) => setPreviewUrl(e.target?.result as string);
    reader.readAsDataURL(file);
  };

  const runPrediction = async () => {
    if (!previewUrl || !session) return;

    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      const img = await new Promise<HTMLImageElement>((resolve, reject) => {
        const i = new Image();
        i.onload = () => resolve(i);
        i.onerror = reject;
        i.src = previewUrl;
      });

      const tensor = preprocess(img);
      const inputName = session.inputNames[0];
      const output = await session.run({ [inputName]: tensor });
      const rawData = Array.from(output[session.outputNames[0]].data as Float32Array);
      const probs = calculateSoftmax(rawData);

      const predictionResults = WEATHER_CLASSES.map((item, idx) => ({
        ...item,
        probability: probs[idx] || 0,
      })).sort((a, b) => b.probability - a.probability);

      setResults(predictionResults);
    } catch (err: any) {
      setErrorMessage('Analysis failed: ' + err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetApp = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResults(null);
    setErrorMessage(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // --- Helper Functions ---
  const preprocess = (img: HTMLImageElement) => {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = INPUT_SIZE;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);

    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const floatData = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
      floatData[i] = imageData[i * 4] / 255.0; // R
      floatData[INPUT_SIZE * INPUT_SIZE + i] = imageData[i * 4 + 1] / 255.0; // G
      floatData[2 * INPUT_SIZE * INPUT_SIZE + i] = imageData[i * 4 + 2] / 255.0; // B
    }

    return new (window as any).ort.Tensor('float32', floatData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  };

  const calculateSoftmax = (logits: number[]) => {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
  };

  // --- UI Components ---
  return (
    <div className={`min-h-screen transition-colors duration-300 ${isDarkMode ? 'bg-gray-950 text-white' : 'bg-gray-50 text-gray-900'}`}>
      <div className="max-w-3xl mx-auto px-8 py-16 font-sans">
        
        {/* Header */}
        <header className="mb-16 flex justify-between items-start">
          <div>
            <h1 className="text-6xl font-black tracking-tighter uppercase mb-3">SkyLens</h1>
            <p className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'} text-base tracking-widest uppercase font-medium`}>Weather Vision AI</p>
          </div>
          <div className="flex flex-col items-end gap-6">
            <button 
              onClick={() => setIsDarkMode(!isDarkMode)}
              aria-label="Toggle Theme"
              className={`group relative w-16 h-8 rounded-full transition-all duration-500 border p-1
                ${isDarkMode ? 'bg-gray-900 border-gray-800' : 'bg-gray-100 border-gray-200'}`}
            >
              <div className={`absolute top-1 left-1 w-6 h-6 rounded-full transition-all duration-500 flex items-center justify-center shadow-sm
                ${isDarkMode ? 'translate-x-8 bg-gray-800 text-yellow-400' : 'translate-x-0 bg-white text-gray-400'}`}>
                {isDarkMode ? (
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707M16.95 16.95l.707.707M7.05 7.05l.707.707M12 8a4 4 0 100 8 4 4 0 000-8z" />
                  </svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                  </svg>
                )}
              </div>
            </button>
            <div className={`px-4 py-2 rounded-lg border text-xs font-bold uppercase tracking-wider flex items-center gap-3 
              ${modelStatus === 'ready' 
                ? 'border-green-500/50 text-green-500 bg-green-500/5' 
                : isDarkMode ? 'border-gray-800 text-gray-500 bg-gray-900' : 'border-gray-200 text-gray-400 bg-white'}`}>
              <div className={`w-2 h-2 rounded-full animate-pulse ${modelStatus === 'ready' ? 'bg-green-500' : 'bg-gray-500'}`} />
              {modelStatus === 'ready' ? 'Online' : 'Loading Model'}
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="space-y-10">
          
          {/* Error Display */}
          {errorMessage && (
            <div className="bg-red-500/10 border border-red-500/50 text-red-500 p-5 rounded-xl text-base font-medium">
              {errorMessage}
            </div>
          )}

          {/* Upload Box */}
          {!previewUrl && (
            <div 
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all
                ${isDarkMode 
                  ? 'border-gray-800 bg-gray-900/50 hover:border-gray-600' 
                  : 'border-gray-200 bg-white hover:border-blue-400 shadow-sm'}`}
            >
              <div className="text-6xl mb-6">🌤️</div>
              <h3 className="text-2xl font-bold mb-2">Upload Sky Image</h3>
              <p className={`${isDarkMode ? 'text-gray-500' : 'text-gray-400'} text-base mb-8`}>Click to browse or drag and drop your photo here</p>
              <button className={`${isDarkMode ? 'bg-white text-black' : 'bg-gray-900 text-white'} px-8 py-3 rounded-full text-sm font-bold uppercase tracking-wider transition-transform active:scale-95`}>
                Browse Files
              </button>
              <input 
                type="file" 
                className="hidden" 
                ref={fileInputRef} 
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              />
            </div>
          )}

          {/* Preview and Analysis Button */}
          {previewUrl && !results && (
            <div className={`border rounded-2xl p-8 flex items-center gap-8 transition-all
              ${isDarkMode ? 'bg-gray-900 border-gray-800' : 'bg-white border-gray-100 shadow-lg'}`}>
              <img src={previewUrl} className="w-32 h-32 object-cover rounded-xl shadow-inner" alt="Preview" />
              <div className="flex-1">
                <h4 className="text-lg font-bold mb-1 truncate">{selectedFile?.name}</h4>
                <p className={`${isDarkMode ? 'text-gray-500' : 'text-gray-400'} text-sm mb-6 uppercase tracking-tight font-mono`}>
                  {(selectedFile!.size / 1024).toFixed(1)} KB
                </p>
                <button 
                  onClick={runPrediction}
                  disabled={isAnalyzing || modelStatus !== 'ready'}
                  className="bg-green-500 hover:bg-green-400 text-black px-8 py-3 rounded-full text-sm font-bold uppercase tracking-wider disabled:opacity-50 transition-colors shadow-lg shadow-green-500/20"
                >
                  {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
                </button>
              </div>
              <button onClick={resetApp} className="text-gray-500 hover:text-red-500 p-2 text-2xl transition-colors">✕</button>
            </div>
          )}

          {/* Results Display */}
          {results && (
            <div className={`border rounded-3xl overflow-hidden transition-all
              ${isDarkMode ? 'bg-gray-900 border-gray-800' : 'bg-white border-gray-100 shadow-2xl'}`}>
              <div className="relative h-64 w-full">
                <img src={previewUrl!} className="w-full h-full object-cover" alt="Result" />
                <div className="absolute inset-0 bg-gradient-to-t from-gray-900/80 to-transparent" />
              </div>
              
              <div className="p-10">
                <div className="mb-10">
                  <span className="text-5xl mb-3 block">{results[0].icon}</span>
                  <h2 className="text-5xl font-black uppercase mb-2 tracking-tighter leading-none">{results[0].label}</h2>
                  <p className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'} text-lg mb-6 leading-relaxed`}>{results[0].desc}</p>
                  <div className="inline-block px-4 py-1.5 rounded-full bg-green-500/10 border border-green-500/30 text-green-500 text-xs font-bold tracking-[0.2em] uppercase">
                    Confidence: {(results[0].probability * 100).toFixed(1)}%
                  </div>
                </div>

                {/* Probability Bars */}
                <div className="space-y-6 mb-10">
                  {results.map((item, idx) => (
                    <div key={item.label} className="flex items-center gap-6">
                      <div className="w-28 text-sm font-bold uppercase tracking-tighter truncate opacity-70">
                        {item.icon} {item.label}
                      </div>
                      <div className={`flex-1 h-2 rounded-full overflow-hidden ${isDarkMode ? 'bg-gray-800' : 'bg-gray-100'}`}>
                        <div 
                          className={`h-full transition-all duration-1000 ease-out rounded-full ${idx === 0 ? item.color : isDarkMode ? 'bg-gray-700' : 'bg-gray-300'}`}
                          style={{ width: `${(item.probability * 100).toFixed(1)}%` }}
                        />
                      </div>
                      <div className="w-12 text-right text-sm font-mono font-bold opacity-60">
                        {(item.probability * 100).toFixed(0)}%
                      </div>
                    </div>
                  ))}
                </div>

                <button 
                  onClick={resetApp}
                  className={`w-full py-4 rounded-xl text-sm font-bold uppercase tracking-widest transition-all
                    ${isDarkMode 
                      ? 'border border-gray-800 hover:bg-gray-800' 
                      : 'bg-gray-900 text-white hover:bg-black shadow-lg shadow-gray-900/20'}`}
                >
                  Analyze New Image
                </button>
              </div>
            </div>
          )}

        </main>

        <footer className={`mt-24 pt-10 border-t text-center text-[10px] uppercase tracking-[0.3em] font-bold
          ${isDarkMode ? 'border-gray-900 text-gray-600' : 'border-gray-200 text-gray-400'}`}>
          SkyLens AI — Professional Weather Diagnostics
        </footer>

        {/* Loading Overlay */}
        {isAnalyzing && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex flex-col items-center justify-center">
            <div className="w-12 h-12 border-4 border-gray-800 border-t-green-500 rounded-full animate-spin mb-6" />
            <p className="text-xs uppercase tracking-[0.4em] font-black text-white animate-pulse">Processing Conditions</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
