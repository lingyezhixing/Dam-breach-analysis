# 使用一个轻量级的 Nginx 官方镜像作为基础
FROM nginx:stable-alpine

# 移除 Nginx 默认的配置
RUN rm /etc/nginx/conf.d/default.conf

# 复制我们自定义的配置文件和网站文件
COPY default.conf /etc/nginx/conf.d/
COPY index.html /usr/share/nginx/html/index.html

# 复制 SSL 证书和私钥 - USE YOUR FILENAMES HERE
COPY dam-breach-analysis.lingyezhixing.top.crt /etc/nginx/ssl/
COPY dam-breach-analysis.lingyezhixing.top.key /etc/nginx/ssl/

# 在一个 RUN 指令中设置所有文件的正确权限，以优化镜像层级
#   - 644: 对网站文件、配置文件和证书是安全的标准权限 (所有人可读)
#   - 600: 对私钥是必要的安全权限 (仅文件所有者可读)
RUN chmod 644 /etc/nginx/conf.d/default.conf \
    && chmod 644 /usr/share/nginx/html/index.html \
    && chmod 644 /etc/nginx/ssl/dam-breach-analysis.lingyezhixing.top.crt \
    && chmod 600 /etc/nginx/ssl/dam-breach-analysis.lingyezhixing.top.key

# 暴露 HTTP (80) 和 HTTPS (443) 端口
EXPOSE 80
EXPOSE 443

# Nginx 镜像的默认命令会自动启动服务，所以不需要再加 CMD