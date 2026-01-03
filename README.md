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
    <li>将.env文件中的class_column配置替换为数据文件中的预测列的名称</li>
    <li>将.env文件中的positive_class配置替换为数据文件中表示好客户的值</li>
    <li>将.env文件中的negative_class配置替换为数据文件中表示坏客户的值</li>
    <li>安装中文字体库: sudo apt-get install fonts-noto-cjk</li>
    <li>清除matplotlib缓存: rm -rf ~/.cache/matplotlib</li>
    <li>将.env文件中的font_path配置替换为本地的中文字体库地址, 比如: font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"</li>
    <li>启动监控服务: python -m phoenix.server.main serve</li>
    <li>本地启动服务: python code/UI.py</li>
  </ol>
</p>
