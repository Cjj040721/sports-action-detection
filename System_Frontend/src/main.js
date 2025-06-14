import './assets/main.css'


import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import ElementPlus from 'element-plus'  // 完整引入Element Plus
import 'element-plus/dist/index.css'   // 引入全局样式
import 'echarts'
import VChart from 'vue-echarts'

const app = createApp(App)

app.use(router)
app.use(ElementPlus)
app.component('v-chart', VChart);
app.mount('#app')

