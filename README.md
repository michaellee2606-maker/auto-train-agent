# auto-train-agent

<h3>UI</h3>
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/bd316fe8-4e0c-44d3-8de9-9990f7afa39e" />

<h3>报表内容</h3>
<img width="600" height="887" alt="image" src="https://github.com/user-attachments/assets/f481aaa9-ccfc-4f50-96d0-e62fea3d98f0" />

---

<p>技术栈
  <ul>
    <li>前端页面框架: gradio</li>
    <li>智能体框架: langChain</li>
    <li>语言模型: Qwen/Qwen2.5-7B-Instruct</li>
    <li>多模态模型: moonshotai/Kimi-K2.5</li>
  </ul>
</p>
<p>项目启动步骤
  <ol>
    <li>创建虚拟环境: python -m venv .venv</li>
    <li>激活虚拟环境: source .venv/bin/activate</li>
    <li>安装依赖: pip install -r requirements.txt</li>
    <li>执行命令: env | grep -i proxy</li>
    <li>根据上一步返回的代理IP和端口替换.env文件中HTTP_PROXY和HTTPS_PROXY的配置</li>
    <li>下载langfuse代码: git clone https://github.com/langfuse/langfuse.git</li>
    <li>执行命令: sudo mkdir -p /etc/systemd/system/docker.service.d</li>
    <li>执行命令: sudo vi /etc/systemd/system/docker.service.d/http-proxy.conf</li>
    <li>将HTTP_PROXY和HTTPS_PROXY的值替换为.env文件中的配置，然后将下面的内容添加到http-proxy.conf文件中:<br/>
[Service]<br/>
Environment="HTTP_PROXY=http://proxy_ip:proxy_port"<br/>
Environment="HTTPS_PROXY=http://proxy_ip:proxy_port"<br/>
Environment="NO_PROXY=localhost,127.0.0.1"
    </li>
    <li>执行命令: systemctl start docker</li>
    <li>在langfuse文件夹下执行命令: sudo docker compose up</li>
    <li>Langfuse启动成功后，
      <ol>
        <li>创建对应的项目</li>
        <li>Settings面板中创建API Keys</li>
        <li>将.env文件中的LANGFUSE_SECRET_KEY配置替换为生成的LANGFUSE_SECRET_KEY值</li>
        <li>将.env文件中的LANGFUSE_PUBLIC_KEY配置替换为生成的LANGFUSE_PUBLIC_KEY值</li>
      </ol>
    </li>
    <li>将.env文件中的token配置替换为生成的token值</li>
    <li>将.env文件中的class_column配置替换为数据文件中的预测列的名称</li>
    <li>将.env文件中的positive_class配置替换为数据文件中表示坏客户的值</li>
    <li>将.env文件中的negative_class配置替换为数据文件中表示好客户的值</li>
    <li>安装中文字体库: sudo apt-get install fonts-noto-cjk</li>
    <li>清除matplotlib缓存: rm -rf ~/.cache/matplotlib</li>
    <li>将.env文件中的font_path配置替换为本地的中文字体库地址, 比如: font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"</li>
    <li>本地启动服务: python code/UI.py</li>
  </ol>
</p>



