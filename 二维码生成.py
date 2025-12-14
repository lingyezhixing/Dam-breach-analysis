import qrcode

# 第一个网址和对应的文件名
url1 = "https://frp-act.com:52754/"
filename1 = "frp_act_qrcode.png"

# 第二个网址和对应的文件名
url2 = "https://dam-breach-analysis.lingyezhixing.top:45000"
filename2 = "dam_breach_analysis_qrcode.png"

try:
    # --- 生成第一个二维码 ---
    # 创建一个二维码对象，可以进行更详细的设置
    qr1 = qrcode.QRCode(
        version=1,  # 控制二维码的大小 (1-40)
        error_correction=qrcode.constants.ERROR_CORRECT_L,  # 错误纠正级别
        box_size=10,  # 每个格子的像素数
        border=4,  # 边框的格子数
    )
    # 添加数据到二维码
    qr1.add_data(url1)
    qr1.make(fit=True)
    # 创建二维码图片
    img1 = qr1.make_image(fill_color="black", back_color="white")
    # 保存图片
    img1.save(filename1)
    print(f"成功为 '{url1}' 生成二维码，并保存为 '{filename1}'")

    # --- 生成第二个二维码 ---
    # 这里使用一个更简洁的方法 qrcode.make()
    img2 = qrcode.make(url2)
    # 保存图片
    img2.save(filename2)
    print(f"成功为 '{url2}' 生成二维码，并保存为 '{filename2}'")

except Exception as e:
    print(f"生成二维码时发生错误: {e}")