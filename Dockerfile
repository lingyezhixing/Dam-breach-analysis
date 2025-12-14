# 使用一个轻量级的 Nginx 官方镜像作为基础
FROM nginx:stable-alpine

# 将我们自定义的 index.html 文件复制到 Nginx 的默认网站根目录
COPY index.html /usr/share/nginx/html/index.html

# 移除 Nginx 默认的配置
RUN rm /etc/nginx/conf.d/default.conf

# 复制我们自定义的 Nginx 配置文件
COPY default.conf /etc/nginx/conf.d/

# 创建存放证书的目录
RUN mkdir -p /etc/nginx/ssl

# 复制 SSL 证书和私钥 - USE YOUR FILENAMES HERE
COPY dam-breach-analysis.lingyezhixing.top.crt /etc/nginx/ssl/
COPY dam-breach-analysis.lingyezhixing.top.key /etc/nginx/ssl/

# 暴露 HTTP (80) 和 HTTPS (443) 端口
EXPOSE 80
EXPOSE 443