import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'./e2e',timeout:90000,expect:{timeout:30000},workers:1,reporter:'list',use:{baseURL:'http://127.0.0.1:4196',viewport:{width:1280,height:850},reducedMotion:'reduce',screenshot:'only-on-failure'},webServer:{command:'npm run dev -- --port 4196 --strictPort',url:'http://127.0.0.1:4196',reuseExistingServer:false}});
