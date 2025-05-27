# 智能学习推荐系统 - 前端界面

这是一个基于React、TypeScript、shadcn/ui和Recharts构建的现代化前端界面，用于与智能学习推荐系统的后端API进行交互。

## 功能特性

### 🧠 学习预测
- 基于SAINT+模型的序列学习预测
- 支持输入题目ID、响应时间和题目类别序列
- 实时可视化预测概率结果
- 交互式折线图展示预测趋势

### 📚 题目推荐
- **基础推荐**: 基于标签重叠度的推荐算法
- **高级推荐**: 基于知识掌握度和题目难度的智能推荐
- 支持自定义基准标签和推荐数量
- 柱状图可视化推荐结果

### 📊 数据可视化
- 使用Recharts库提供交互式图表
- 实时API状态监控
- 响应式设计，支持移动端

## 技术栈

- **React 18** - 前端框架
- **TypeScript** - 类型安全
- **Tailwind CSS** - 样式框架
- **shadcn/ui** - 现代化UI组件库
- **Recharts** - 数据可视化
- **Axios** - HTTP客户端
- **Lucide React** - 图标库

## 安装和运行

### 前提条件
- Node.js 16+ 
- npm 或 yarn

### 安装依赖
```bash
cd frontend
npm install
```

### 启动开发服务器
```bash
npm start
```

应用将在 `http://localhost:3000` 启动。

### 构建生产版本
```bash
npm run build
```

## API配置

默认API地址配置在 `src/services/api.ts` 文件中：

```typescript
const API_BASE_URL = 'http://localhost:8000';
```

如果后端API运行在不同的地址，请修改此配置。

## 使用说明

### 学习预测功能
1. 切换到"学习预测"标签页
2. 输入题目ID序列（用逗号分隔），例如：`1,2,3,4,5`
3. 输入响应时间序列（毫秒），例如：`1000,2000,1500,3000,2500`
4. 输入题目类别序列，例如：`1,2,1,3,2`
5. 点击"开始预测"按钮
6. 查看预测结果的可视化图表和数值

### 题目推荐功能
1. 切换到"题目推荐"标签页
2. 输入用户ID，例如：`12345`
3. （可选）输入基准标签，例如：`1,2,3`
4. 设置推荐数量，例如：`10`
5. 选择推荐算法：
   - 点击"基础推荐"使用标签重叠度算法
   - 点击"高级推荐"使用知识追踪算法
6. 查看推荐结果的可视化图表

## 项目结构

```
frontend/
├── public/                 # 静态资源
├── src/
│   ├── components/         # React组件
│   │   ├── ui/            # shadcn/ui基础组件
│   │   ├── PredictTab.tsx # 预测功能组件
│   │   └── RecommendTab.tsx # 推荐功能组件
│   ├── services/          # API服务
│   │   └── api.ts         # API接口定义
│   ├── lib/               # 工具函数
│   │   └── utils.ts       # 通用工具
│   ├── App.tsx            # 主应用组件
│   └── index.tsx          # 应用入口
├── package.json           # 项目配置
└── tailwind.config.js     # Tailwind配置
```

## 自定义和扩展

### 添加新的UI组件
shadcn/ui组件位于 `src/components/ui/` 目录下，可以根据需要添加更多组件。

### 修改样式主题
在 `tailwind.config.js` 和 `src/index.css` 中修改颜色主题和样式变量。

### 扩展API功能
在 `src/services/api.ts` 中添加新的API接口定义。

## 故障排除

### API连接问题
- 确保后端API服务正在运行
- 检查API地址配置是否正确
- 查看浏览器控制台的网络错误信息

### 样式问题
- 确保Tailwind CSS正确配置
- 检查CSS类名是否正确

### 依赖问题
- 删除 `node_modules` 文件夹并重新安装依赖
- 确保Node.js版本兼容

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

MIT License
