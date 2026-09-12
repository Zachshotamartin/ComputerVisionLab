import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({
  root: 'demo',
  publicDir: '../web/assets',
  plugins: [react()],
  build: { outDir: '../dist', emptyOutDir: true },
});
