# auto-train-agent

<p>构建自动训练模型的智能体</p>
<p>技术栈
  <ul>
    <li>语言: python</li>
    <li>前端页面框架: gradio</li>
    <li>智能体框架: smolagents</li>
    <li>模型: Qwen/Qwen2.5-Coder-32B-Instruct</li>
  </ul>
</p>
<p>项目启动步骤
  <ol>
    <li>创建虚拟环境: python -m venv .venv</li>
    <li>激活虚拟环境: source .venv/bin/activate</li>
    <li>安装依赖: pip install -r requirements.txt</li>
    <li>将.env文件中的token配置替换为生成的token值</li>
    <li>启动监控服务: python -m phoenix.server.main serve</li>
    <li>本地启动服务: python code/UI.py</li>
  </ol>
</p>
