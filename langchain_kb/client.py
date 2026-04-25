import requests
import json
import time

SERVER_URL = "http://10.63.242.62:8001"


def chat(query):
    url = f"{SERVER_URL}/chat"
    try:
        # 记录开始时间
        start_time = time.time()
        print("Waiting for AI response...", end="", flush=True)

        resp = requests.post(url, json={"query": query})

        if resp.status_code == 200:
            data = resp.json()
            print(f"\r", end="")  # 清除等待提示

            print(f"\n=== AI ({time.time() - start_time:.2f}s) ===")
            print(data["answer"])

            print("\n=== 参考资料 ===")
            for idx, src in enumerate(data["sources"]):
                print(f"[{idx + 1}] {src}")
            print("-" * 40)
        else:
            print(f"\nError: {resp.text}")
    except Exception as e:
        print(f"\nConnection failed: {e}")


def rebuild_kb():
    url = f"{SERVER_URL}/rebuild"
    print("\n>>> 发送重建指令到服务器...")
    try:
        # 在这里修改 urls 列表，添加新链接
        payload = {
            # pdf_files 设为 None，让服务器自动去扫描文件夹
            "pdf_files": None,

            # 在这里添加新的网址
            "urls": [
                "https://baike.baidu.com/item/%E9%BB%91%E8%84%B8%E7%90%B5%E9%B9%AD/347612",
                "https://datazone.birdlife.org/species/factsheet/black-faced-spoonbill-platalea-minor",
            ]
        }
        resp = requests.post(url, json=payload)
        if resp.status_code == 200:
            print(f"Server: {resp.json()['message']}")
        else:
            print(f"Error: {resp.text}")
    except Exception as e:
        print(f"Connection failed: {e}")


if __name__ == "__main__":
    print(f"连接到 RAG 服务器: {SERVER_URL}")
    print("指令说明:")
    print(" - 直接输入文字: 进行提问")
    print(" - 输入 /rebuild: 重建知识库")
    print(" - 输入 /quit: 退出")
    print("-" * 40)

    while True:
        user_input = input("\n用户: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["/quit", "exit"]:
            break

        if user_input.lower() == "/rebuild":
            rebuild_kb()
            continue

        chat(user_input)