<template>
  <div class="app-container">
    <!-- 头部 -->
    <el-header class="header">
      <div class="header-content">
        <div class="branding">
          <h1>智能体育动作检测系统</h1>
        </div>
        <nav class="nav-links">
          <el-button link type="primary">实时检测</el-button>
          <el-button link type="primary">历史记录</el-button>
          <el-button link type="primary">教学资源</el-button>
        </nav>
      </div>
      <div class="header-divider"></div>
    </el-header>

    <!-- 主内容区 -->
    <el-main class="main-content">
      <!-- 检测控制区 -->
      <el-card class="control-card" shadow="hover">
        <template #header>
          <div class="card-title">动作检测控制</div>
        </template>
        
        <div class="control-section">
          <el-button 
            :disabled="false"
            size="large"
            @click="startDetection"
          >
          <!--el-button 
            :loading="isDetecting && !detectionInterval"
            :disabled="false"
            size="large"
            @click="detection"
          ></el-button-->
            {{ buttonText }}
          </el-button>
          
          <el-switch 
            v-model="sensitivity"
            :active-text="sensitivity ? '高灵敏度' : '标准模式'"
            style="margin-left: 20px;"
          />
        </div>
      </el-card>

      <!-- 实时结果展示 -->
      <el-card class="result-card" shadow="hover" v-if="currentResult">
        <template #header>
          <div class="card-title">当前检测结果</div>
        </template>
        
        <div class="result-detail">
          <p class="action-name">动作名称：{{ currentResult.actionName }}</p>
          <el-progress 
            :percentage="currentResult.accuracy"
            :format="(percent) => `${percent}% 标准度`"
            :stroke-width="24"
            status="success"
          />
          <div class="standard-status">
            <el-tag :type="currentResult.isStandard ? 'success' : 'danger'">
              {{ currentResult.isStandard ? '标准动作' : '不标准动作' }}
            </el-tag>
          </div>
        </div>
      </el-card>

      <!-- 动作建议 -->
      <el-card class="suggestion-card" shadow="hover" v-if="currentResult && !currentResult.isStandard">
        <template #header>
          <div class="card-title">动作优化建议</div>
        </template>
        
        <ul class="suggestion-list">
          <li v-for="(item, index) in currentResult.suggestions" :key="index">
            {{ index + 1 }}. {{ item }}
          </li>
        </ul>
      </el-card>

      <!-- 数据统计与历史 -->
      <el-row :gutter="20">
        <!-- 统计图表 -->
        <el-col :span="12">
          <el-card shadow="hover">
            <template #header>动作标准率统计（最近10次）</template>
                  <div class="chart-container">
                    <v-chart :option="chartOption" autoresize />
                  </div>
          </el-card>
        </el-col>

        <!-- 历史记录 -->
        <el-col :span="12">
          <el-card shadow="hover">
            <template #header>最近检测记录</template>
            <el-table :data="actionHistory" style="width: 100%">
              <el-table-column prop="timestamp" label="检测时间" width="180" />
              <el-table-column prop="actionName" label="动作名称" />
              <el-table-column label="标准性">
                <template #default="scope">
                  <el-tag :type="scope.row.isStandard ? 'success' : 'danger'">
                    {{ scope.row.isStandard ? '标准' : '不标准' }}
                  </el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="count" label="次数" width="80" />">
            </el-table>
          </el-card>
        </el-col>
      </el-row>
    </el-main>
  </div>
</template>

<script setup>
import { ref, computed, reactive } from 'vue';

import VChart from 'vue-echarts';
import { use } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { LineChart } from 'echarts/charts';
import { TitleComponent, TooltipComponent, LegendComponent, GridComponent } from 'echarts/components';
import { ElMessage } from 'element-plus/lib/components/index.js';


// 初始化ECharts组件
use([CanvasRenderer, LineChart, TitleComponent, TooltipComponent, LegendComponent, GridComponent]);

// 状态管理
const isDetecting = ref(false);
const currentResult = ref(null);
const actionHistory = ref([
 
]); // 添加默认数据以确保图表有数据可渲染
const sensitivity = ref(true); // 检测灵敏度（模拟）
const id2label = ref({
  0: { name: "交替腿举", standard: false, suggestions: ["保持腿部伸直", "控制动作节奏"] },
  1: { name: "交替腿举", standard: false, suggestions: ["保持腿部伸直", "控制动作节奏"] },
  2: { name: "仰卧起坐", standard: false, suggestions: ["收紧腹部肌肉", "控制起身速度"] },
  3: { name: "仰卧起坐", standard: false, suggestions: ["收紧腹部肌肉", "控制起身速度"] },
  4: { name: "俄罗斯转体", standard: false, suggestions: ["保持背部挺直", "控制旋转幅度"] },
  5: { name: "俄罗斯转体", standard: false, suggestions: ["保持背部挺直", "控制旋转幅度"] },
  6: { name: "俯卧撑", standard: false, suggestions: ["保持身体直线", "控制下降速度"] },
  7: { name: "俯卧撑", standard: false, suggestions: ["保持身体直线", "控制下降速度"] },
  8: { name: "前弓步", standard: false, suggestions: ["保持膝盖不超过脚尖", "控制重心平衡"] },
  9: { name: "前弓步", standard: false, suggestions: ["保持膝盖不超过脚尖", "控制重心平衡"] },
  10: { name: "开合跳", standard: false, suggestions: ["保持节奏稳定", "控制落地缓冲"] },
  11: { name: "开合跳", standard: false, suggestions: ["保持节奏稳定", "控制落地缓冲"] },
  12: { name: "深蹲", standard: false, suggestions: ["保持背部挺直", "控制下蹲深度"] },
  13: { name: "深蹲", standard: false, suggestions: ["保持背部挺直", "控制下蹲深度"] },
  14: { name: "登山者", standard: false, suggestions: ["保持核心稳定", "控制动作幅度"] },
  15: { name: "登山者", standard: false, suggestions: ["保持核心稳定", "控制动作幅度"] },
  16: { name: "高抬腿", standard: false, suggestions: ["保持上身直立", "控制抬腿高度"] },
  17: { name: "高抬腿", standard: false, suggestions: ["保持上身直立", "控制抬腿高度"] },
  18: { name: "鸟狗式", standard: false, suggestions: ["保持身体平衡", "控制伸展幅度"] },
  19: { name: "鸟狗式", standard: false, suggestions: ["保持身体平衡", "控制伸展幅度"] },
});
const recentHistory = ref([
      // 示例数据
      { timestamp: Date.now(), accuracy: 85 },
      { timestamp: Date.now() - 60000, accuracy: 78 },
      { timestamp: Date.now() - 120000, accuracy: 90 }
    ]);
const rightAction = ref(0)
const wrongAction = ref(0)
const chartOption = computed(() => ({
  title: { 
      text: '动作标准率分布', 
      left: 'center' 
  },
  tooltip: {
      trigger: 'item',
      formatter: '{a} <br/>{b}: {c} ({d}%)'
  },
  legend: {
    orient: 'vertical',
    left: 'left',
    top: '7%'
  },
  series: [
    {
      name: '动作标准率',
      type: 'pie',
      radius: '50%',
      data: [
        { 
          //value: actionHistory.value.filter(item => item.isStandard).length,
          value: rightAction.value,
          name: '标准动作',
          itemStyle: { color: '#67C23A' } // 绿色表示标准
        },
        { 
          //value: actionHistory.value.filter(item => !item.isStandard).length,
          value: wrongAction.value,
          name: '不标准动作',
          itemStyle: { color: '#F56C6C' } // 红色表示不标准
        }
      ],
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }
  ]
}))
const detectionInterval = ref(null); // 添加计时器引用

// 修改按钮文本显示逻辑
const buttonText = computed(() => {
  // if (isDetecting.value && detectionInterval.value) {
  //   return '停止检测';
  // }
  return isDetecting.value ? '检测中...' : '开始检测动作';
});
const startDetection = async () => {
  // 如果已经有定时器在运行，则停止检测
  if (detectionInterval.value) {
    clearInterval(detectionInterval.value);
    detectionInterval.value = null;
    isDetecting.value = false;
    return;
  }

  isDetecting.value = true;
  
  // 设置定时器，每隔1秒执行一次检测
  detectionInterval.value = setInterval(async () => {
    try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);
    
    // 调用真实后端接口，添加超时控制
    const response = await fetch('http://localhost:5000/detect', {
      signal: controller.signal
    });
    clearTimeout(timeoutId);
      const res = await response.json();
      if(res.prediction == 'No'){
        return;
      }
      const resultId = res.prediction
      
      const mockResult = {
        actionName: id2label.value[resultId].name,
        isStandard: resultId % 2 === 1,
        accuracy: resultId % 2 === 1 
          ? Math.floor(Math.random()*20 + 80)
          : Math.floor(Math.random()*30 + 20),
        suggestions: id2label.value[resultId].suggestions,
        count: res.total_count
      };

      currentResult.value = mockResult;
      
      actionHistory.value.unshift({
        ...mockResult,
        timestamp: new Date().toLocaleString()
      });
      if (actionHistory.value.length > 10) {
        actionHistory.value.pop();
      }
    } catch (error) {
      if (error.name === 'AbortError') {
      console.error('请求超时，已取消');
      //ElMessage.error('检测请求超时，请检查后端服务');
    } else {
      console.error('检测失败:', error);
      ElMessage.error('检测过程中出现错误，请重试');
    }
    }
  }, 1000); // 1秒间隔
};
import { onUnmounted } from 'vue';

const detection = async () => {
      console.log(isDetecting.value)
      if(isDetecting.value == false){

        isDetecting.value = true;
        
      }else{
          isDetecting.value = false;
          await new Promise(resolve => setTimeout(resolve, 3000));
          const resultId = 1
        
        const mockResult = {
          actionName: id2label.value[resultId].name,
          isStandard: resultId % 2 === 1,
          accuracy: resultId % 2 === 1 
            ? Math.floor(Math.random()*20 + 80)
            : Math.floor(Math.random()*30 + 20),
          suggestions: id2label.value[resultId].suggestions,
          count: 15,
          timestamp: new Date().toLocaleString()
        };
        rightAction.value = 10
        wrongAction.value = 5
        currentResult.value = mockResult;
        
        actionHistory.value.unshift({
          ...mockResult,
        },{
          ...mockResult,
        });
        if (actionHistory.value.length > 10) {
          actionHistory.value.pop();
        }
      }
};

onUnmounted(() => {
  if (detectionInterval.value) {
    clearInterval(detectionInterval.value);
  }
});

</script>

<style scoped>
.container {
  max-width: 2000px;
  margin: 0 auto;
  padding: 2rem;
  display: flex;
  gap: 2rem;
  background: #f5f7fa;
  min-height: 100vh;
}

.left-section {
  flex: 1;
  background: white;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.right-section {
  width: 1000px;
  background: white;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.chart-container {
  height: 400px;
  margin-bottom: 2rem;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  padding: 1rem;
}
.app-container {
  min-height: 100vh;
  background: #f8fafc; /* 更柔和的背景色 */
}

.header {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  height: 64px !important;
  padding: 0 40px;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 100%;
}

.branding {
  display: flex;
  align-items: center;
  gap: 12px;
}

.main-content {
  max-width: 1600px; /* 加宽内容区域 */
}

/* 卡片样式升级 */
.el-card {
  border-radius: 16px !important; /* 更大的圆角 */
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border: 1px solid rgba(224, 224, 224, 0.5);
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
  }
}

/* 按钮样式升级 */
.el-button--primary {
  background: var(--primary-gradient);
  border: none;
  font-weight: 600;
  letter-spacing: 0.5px;
  transition: all 0.3s ease;
  
  &:hover {
    transform: scale(1.05);
    opacity: 0.9;
  }
}

/* 图表容器升级 */
.chart {
  height: 400px;
  width: 100px;  
  border-radius: 12px;
  background: white;
  padding: 20px;
  margin-top: 20px;  
}

/* 视频布局优化 */
.video-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
  
  video {
    aspect-ratio: 16/9; /* 固定视频比例 */
    transition: transform 0.3s ease;
    
    &:hover {
      transform: scale(1.03);
    }
  }
}

/* 新增字体优化 */
h1 {
  font-family: 'Inter', sans-serif;
  font-size: 1.8rem;
  font-weight: 700;
  letter-spacing: -0.5px;
  color: rgba(255, 255, 255, 0.95);
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  margin: 0;
}

.nav-links {
  display: flex;
  gap: 24px;
  
  .el-button {
    font-weight: 500;
    letter-spacing: 0.5px;
    color: rgba(255, 255, 255, 0.9) !important;
    transition: all 0.2s ease;
    
    &:hover {
      color: white !important;
      transform: translateY(-1px);
      opacity: 1;
    }
  }
}




.card-title {
  font-size: 1.4rem;
  font-weight: 600;
  color: var(--dark-bg);
}

/* 统计图表配色更新 */
:deep(.echarts) {
  .series-line {
    stroke: var(--secondary-color);
  }
  
  .axis-label {
    color: #718096 !important;
  }
}

/* 表格样式优化 */
.el-table {
  --el-table-header-bg-color: #f7fafc;
  --el-table-row-hover-bg-color: #f7fafc;
  
  :deep(th.el-table__cell) {
    font-weight: 600;
  }
}

.el-button--primary {
  background: var(--primary-gradient);
  border: none;
  font-weight: 600;
  letter-spacing: 0.5px;
  transition: all 0.3s ease;
  opacity: 1; /* 新增：确保默认状态可见 */
  
  &:hover {
    transform: scale(1.05);
    opacity: 0.9;
  }
}
</style>
