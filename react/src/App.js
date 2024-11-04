// src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { ResultProvider } from './context/ResultContext';
import Home from './components/Home';
import Upload from './components/detect/Upload';
import Result from './components/detect/Result';
import Predict from './components/detect/Predict';
import Main from './components/disrupt/Main';
import Compare from './components/disrupt/Compare';
import './styles/App.css';

function App() {
  return (
    <ResultProvider>
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />

          <Route path="/disrupt/upload" element={<Main />} />
          <Route path="/disrupt/compare" element={<Compare />} />

          <Route path="/detect/upload" element={<Upload />} />
          <Route path="/detect/predict" element={<Predict />} />
          <Route path="/detect/result" element={<Result />} />
        </Routes>
      </Router>
    </ResultProvider>
  );
}

export default App;
