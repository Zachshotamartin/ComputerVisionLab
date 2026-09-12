import { createRoot } from 'react-dom/client';
import VisionLab from '../web/index.jsx';
import './style.css';
createRoot(document.getElementById('root')).render(<main><header><a href="https://github.com/Zachshotamartin/ComputerVisionLab">Zach Martin / Computer Vision Lab</a><h1>See what the image contains.</h1><p>Choose an operation, change the inputs, and inspect the result. All images and camera frames stay in your browser.</p></header><VisionLab assetBase="/" /></main>);
