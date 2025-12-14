# 使用一个轻量级的 Nginx 官方镜像作为基础
FROM nginx:stable-alpine

# 将我们自定义的 index.html 文件复制到 Nginx 的默认网站根目录
# 这会覆盖掉 Nginx 默认的欢迎页面
COPY index.html /usr/share/nginx/html/index.html

# 暴露容器的 80 端口（Nginx 默认监听的端口）
EXPOSE 80